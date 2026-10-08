package model

import (
	"fmt"
	"io"
	"sync"
)

// Prefix sharing: sessions whose prompts share a prefix hold its KV pages
// once.
//
// A sealed page (one the sequence has written past) never changes, and under
// a namespace it is named by every token up to its end (kvCache.pageKey), so
// two sessions whose prompts agree that far would compute the same bytes.
// SharedStore keeps one copy of each such page and hands every session that
// reaches it the SAME memory: a session's page table (kvPages.k and .v) points
// at it, refcounted, instead of at a page of its own. Nothing else about the
// cache changes -- the attention kernels read a page wherever it is.
//
// The rules that keep it correct:
//
//   - Only sealed, whole pages are shared. The page a session is writing into
//     is its own; the first page where two prompts diverge is computed by
//     each (its key covers the diverging tokens, so it never matches).
//   - A shared page is never written in place. Every write path (the token's
//     append, a migration scattering rows home) copies a shared page into a
//     private one first and drops its reference: copy-on-write.
//   - A reference is dropped wherever the session lets a page go (eviction, a
//     restore's truncation, a window's release, Close), never handed to the
//     session's page pool, which would recycle memory others still read.
//   - A shared page does not move to a device for one of its holders: the
//     block stays on the host (State.migrateKV refuses, counted by
//     KVSharedRefusals). A device page shared between sessions is not
//     implemented; a session with blocks on a device does not share
//     (State.ShareKV).
//   - A recurrent layer's summary is not paged by position, so sharing is
//     refused on a model that has one (State.ShareKV), as it is for a batch,
//     whose pages hold every row.
//
// SharedStore is also an ordinary KVStore: Get and Set copy, so a session that
// does not share (SetKVStore alone) reads and writes it as it would a
// MemStore, and a sharing session finds the pages it published.
type SharedStore struct {
	mu    sync.Mutex
	pages map[kvStoreKey]*sharedPage
	held  map[*float32]*sharedPage
	bytes uint64
}

// sharedPage is one page's single copy: refs sessions alias it.
type sharedPage struct {
	key  kvStoreKey
	b    []float32
	refs int
	// gone means Drop or Prune took it out of the index: no session can
	// acquire it again, and it is forgotten once the last holder lets go.
	gone bool
}

// NewSharedStore returns an empty SharedStore.
func NewSharedStore() *SharedStore {
	return &SharedStore{pages: map[kvStoreKey]*sharedPage{}, held: map[*float32]*sharedPage{}}
}

// Get writes a copy of the held page into page, or returns ErrNoPage.
func (m *SharedStore) Get(cacheId string, layer, index int, page io.Writer) error {
	m.mu.Lock()
	e := m.pages[kvStoreKey{cacheId, layer, index}]
	m.mu.Unlock()
	if e == nil {
		return ErrNoPage
	}
	// The bytes of a held page never change (no session writes one in place),
	// so they are read outside the lock.
	_, err := page.Write(kvBytesOf(e.b))
	return err
}

// Set holds a copy of the page under its key, replacing one no session holds.
// A key a session aliases keeps its page: those bytes are what it reads.
func (m *SharedStore) Set(cacheId string, layer, index int, page io.Reader) error {
	b, err := io.ReadAll(page)
	if err != nil {
		return err
	}
	if len(b)%4 != 0 {
		return fmt.Errorf("model: a kv page of %d bytes is not float32 slots", len(b))
	}
	f := make([]float32, len(b)/4)
	copy(kvBytesOf(f), b)
	m.mu.Lock()
	defer m.mu.Unlock()
	k := kvStoreKey{cacheId, layer, index}
	if e := m.pages[k]; e != nil {
		if e.refs > 0 {
			return nil
		}
		m.forget(e)
	}
	m.add(k, f)
	return nil
}

// Drop forgets the pages held under one cache id. A page a session still
// aliases stays with it until it lets go.
func (m *SharedStore) Drop(cacheId string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	for k, e := range m.pages {
		if k.cache == cacheId {
			m.retire(e)
		}
	}
	return nil
}

// Prune forgets every page no session holds, which is what bounds the store:
// a page is kept for the next session to find until Prune (or Drop) lets it
// go.
func (m *SharedStore) Prune() {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, e := range m.pages {
		if e.refs == 0 {
			m.retire(e)
		}
	}
}

// Bytes is what the store holds: every page once, however many sessions
// alias it.
func (m *SharedStore) Bytes() uint64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.bytes
}

// Refs is how many session references page `index` of `layer` under cacheId
// has, -1 when the store does not hold it.
func (m *SharedStore) Refs(cacheId string, layer, index int) int {
	m.mu.Lock()
	defer m.mu.Unlock()
	if e := m.pages[kvStoreKey{cacheId, layer, index}]; e != nil {
		return e.refs
	}
	return -1
}

// publish hands the store a sealed page a session computed. When the store
// already holds that page, the session gets the held copy (and keeps
// ownership of b, which it gives back to its pool); otherwise b becomes the
// held copy. Either way the session holds one reference to what is returned.
func (m *SharedStore) publish(k kvStoreKey, b []float32) []float32 {
	m.mu.Lock()
	defer m.mu.Unlock()
	if e := m.pages[k]; e != nil {
		if len(e.b) == len(b) {
			e.refs++
			return e.b
		}
		// Another shape under one key is a stale page from another geometry
		// (the key digests it, so only a collision): the new one replaces it.
		m.retire(e)
	}
	e := m.add(k, b)
	e.refs++
	return b
}

// acquire is the held page under k with one more reference, or nil.
func (m *SharedStore) acquire(k kvStoreKey, n int) []float32 {
	m.mu.Lock()
	defer m.mu.Unlock()
	e := m.pages[k]
	if e == nil || len(e.b) != n {
		return nil
	}
	e.refs++
	return e.b
}

// release drops one session reference to the held page b.
func (m *SharedStore) release(b []float32) {
	if len(b) == 0 {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	e := m.held[&b[0]]
	if e == nil {
		panic("model: a shared kv page released that the store does not hold")
	}
	e.refs--
	if e.refs < 0 {
		panic("model: a shared kv page released more often than it was taken")
	}
	if e.refs == 0 && e.gone {
		m.forget(e)
	}
}

// add indexes a new held page. Callers hold m.mu.
func (m *SharedStore) add(k kvStoreKey, b []float32) *sharedPage {
	e := &sharedPage{key: k, b: b}
	m.pages[k] = e
	m.held[&b[0]] = e
	m.bytes += 4 * uint64(len(b))
	return e
}

// retire takes e out of the index, forgetting it now if nobody holds it.
// Callers hold m.mu.
func (m *SharedStore) retire(e *sharedPage) {
	if m.pages[e.key] == e {
		delete(m.pages, e.key)
	}
	e.gone = true
	if e.refs == 0 {
		m.forget(e)
	}
}

// forget drops e entirely. Callers hold m.mu.
func (m *SharedStore) forget(e *sharedPage) {
	if m.pages[e.key] == e {
		delete(m.pages, e.key)
	}
	if m.held[&e.b[0]] == e {
		delete(m.held, &e.b[0])
		m.bytes -= 4 * uint64(len(e.b))
	}
}

// ShareKV makes this session hold the sealed pages of its history once with
// every other session sharing st: what it seals is published to st, and a
// prefix PrefillCached finds there is aliased, not copied (SharedStore has
// the rules). It sets st as the session's store; SetCacheKey must name the
// namespace first, since a page is named by its tokens only inside one.
//
// Refused by name: a batch (its pages hold every row's slots), a model with a
// recurrent layer (its summary is not paged by position), DeepSeek V4 (its
// compressed entries are a second history keyed by entry), and a session with
// blocks on a device (a device page shared between sessions is not
// implemented).
func (s *State) ShareKV(st *SharedStore) error {
	switch {
	case st == nil:
		return fmt.Errorf("model: ShareKV wants a store")
	case s.nseq > 1:
		return fmt.Errorf("model: a %d-row batch cannot share kv pages: one page holds every row's slots", s.nseq)
	case s.kv.recurrent:
		return fmt.Errorf("model: %s has recurrent layers, whose summary is not paged by position: "+
			"kv page sharing is refused on it", s.c.Arch)
	case len(s.kv.ent) > 0:
		return fmt.Errorf("model: %s keeps compressed entries beside its pages: kv page sharing is refused on it", s.c.Arch)
	case s.kv.ns == "":
		return fmt.Errorf("model: ShareKV wants a namespace (SetCacheKey): a page is named by its tokens only inside one")
	case s.ld != nil && s.devCount() > 0:
		return fmt.Errorf("model: %d block(s) of this session are on a device: a device kv page shared "+
			"between sessions is not implemented", s.devCount())
	case s.pos > 0:
		return fmt.Errorf("model: ShareKV wants a fresh sequence, this one is at position %d", s.pos)
	}
	s.kv.store, s.kv.share = st, st
	for li := s.kv.lo; li < len(s.kv.layers); li++ {
		s.kv.layers[li].sh = st
	}
	return nil
}

// KVSharedBytes is what this session reads from pages it shares: held once
// in its SharedStore, and not counted by KVBytes.
func (s *State) KVSharedBytes() uint64 {
	var n uint64
	for li := s.kv.lo; li < len(s.kv.layers); li++ {
		n += s.kv.layers[li].sharedBytes()
	}
	return n
}

// KVSharedRefusals is how many times a block holding shared pages was kept
// off a device: a shared page does not move for one of its holders.
func (s *State) KVSharedRefusals() int64 { return s.kv.shareRefused }

// isShared says page n aliases a SharedStore's page.
func (s *kvPages) isShared(n int) bool { return n < len(s.shared) && s.shared[n] }

// anyShared says some page of this layer is shared.
func (s *kvPages) anyShared() bool {
	for _, x := range s.shared {
		if x {
			return true
		}
	}
	return false
}

// markShared records that page n aliases the store's.
func (s *kvPages) markShared(n int, on bool) {
	for len(s.shared) <= n {
		s.shared = append(s.shared, false)
	}
	s.shared[n] = on
}

// unshare drops this session's references to page n's shared memory and
// leaves the slots for the caller to fill or nil. The bytes are untouched:
// other sessions read them.
func (s *kvPages) unshare(n int) {
	if !s.isShared(n) {
		return
	}
	s.sh.release(s.k[n])
	if !s.latent {
		s.sh.release(s.v[n])
	}
	s.shared[n] = false
}

// cow gives page n private copies of its shared memory before a write: the
// write lands in this session's page and every other holder keeps reading
// what it read.
func (s *kvPages) cow(n int) {
	if !s.isShared(n) {
		return
	}
	if s.cowFault {
		return // the violation: write into the shared page in place
	}
	k := s.pool.get(s.pp)
	copy(k, s.k[n])
	v := k
	if !s.latent {
		v = s.pool.get(s.pp)
		copy(v, s.v[n])
	}
	s.unshare(n)
	s.k[n], s.v[n] = k, v
}

// sharedBytes is what this layer's shared pages hold.
func (s *kvPages) sharedBytes() uint64 {
	var n uint64
	for i := range s.shared {
		if s.shared[i] && i < len(s.k) && s.k[i] != nil {
			n += uint64(s.pp) * 4
			if !s.latent {
				n += uint64(s.pp) * 4
			}
		}
	}
	return n
}

// publishSealed hands page n, sealed, to the store: it becomes the store's
// copy, or the store's existing copy replaces it and the session's buffers go
// back to its pool.
func (kc *kvCache) publishSealed(li, n int, key string) {
	s := &kc.layers[li]
	if s.isShared(n) {
		return // restored from the store: already the held copy
	}
	k := kc.share.publish(kvStoreKey{key, li * 2, n}, s.k[n])
	if &k[0] != &s.k[n][0] {
		s.pool.give(s.k[n])
	}
	v := k
	if !s.latent {
		v = kc.share.publish(kvStoreKey{key, li*2 + 1, n}, s.v[n])
		if &v[0] != &s.v[n][0] {
			s.pool.give(s.v[n])
		}
	}
	s.k[n], s.v[n] = k, v
	s.markShared(n, true)
}

// faultShared aliases page n of layer li from the store, or reports a miss.
func (kc *kvCache) faultShared(li, n int, key string) error {
	s := &kc.layers[li]
	k := kc.share.acquire(kvStoreKey{key, li * 2, n}, s.pp)
	if k == nil {
		return ErrNoPage
	}
	v := k
	if !s.latent {
		if v = kc.share.acquire(kvStoreKey{key, li*2 + 1, n}, s.pp); v == nil {
			kc.share.release(k)
			return ErrNoPage
		}
	}
	s.k[n], s.v[n] = k, v
	s.out[n] = false
	s.markShared(n, true)
	kc.fetched += 2
	return nil
}

// unshareAll drops every shared page's reference, for Close: the pool must
// never take a page others read.
func (kc *kvCache) unshareAll() {
	for li := kc.lo; li < len(kc.layers); li++ {
		s := &kc.layers[li]
		for n := range s.shared {
			if s.shared[n] {
				s.unshare(n)
				s.k[n], s.v[n] = nil, nil
			}
		}
	}
}
