package model

import (
	"container/heap"
	"io"
	"sync"

	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// MemStore keeps pages in host memory, keyed by (cache, layer, index).
//
// NoStore and MemStore must produce the same tokens, which is the equality
// gate. It locks because a store exists to be shared between sessions.
//
// A limit (SetLimit) bounds it in bytes. Past it pages go one at a time,
// least recently used first and, among pages used in the same pass, the one
// furthest into its prompt first. Plain LRU would be the wrong order for a
// prefix cache: a prompt's first page is written first, so it would be the
// first to go, and a restore stops at the first page it misses -- every page
// after it would be held for nothing. Dropping the tail first shortens the
// prefix a restore reaches by one page at a time instead (vLLM and SGLang
// evict a radix tree's leaves first for the same reason). A pass is the run of
// operations until a layer's first page is touched a second time: a prompt
// touches each layer's first page once, whatever order it walks the layers
// in. Dropping any
// page is safe -- the rest is computed -- so the bound costs only hits. Zero,
// the default, holds everything.
//
// Pages live off the Go heap (kernels.MapOffHeap) where the platform maps
// anonymous memory, so a store sized near the memory limit does not leave the
// collector a goal it cannot reach (RULE 2f); a mapping is whole OS pages, and
// that is what Bytes counts.
type MemStore struct {
	mu    sync.Mutex
	pages map[kvStoreKey]*memPage
	order memOrder
	bytes uint64
	limit uint64

	// pass is the clock eviction orders by; heads is the layers whose first
	// page this pass has touched.
	pass  uint64
	heads map[int]struct{}

	hits, misses, sets, evicted int64
}

type kvStoreKey struct {
	cache        string
	layer, index int
}

type memPage struct {
	key    kvStoreKey
	b      []byte
	mapped bool
	pass   uint64
	at     int // its place in order; -1 while pinned
	// pins is how many Pinned views hold the page: two parked sessions
	// sharing a prefix pin its pages twice, and the first Release must not
	// free them under the second.
	pins int
}

// memOrder is a heap whose top is the next page to evict.
type memOrder []*memPage

func (o memOrder) Len() int { return len(o) }
func (o memOrder) Less(i, j int) bool {
	if o[i].pass != o[j].pass {
		return o[i].pass < o[j].pass
	}
	return o[i].key.index > o[j].key.index
}
func (o memOrder) Swap(i, j int) {
	o[i], o[j] = o[j], o[i]
	o[i].at, o[j].at = i, j
}
func (o *memOrder) Push(x any) {
	p := x.(*memPage)
	p.at = len(*o)
	*o = append(*o, p)
}
func (o *memOrder) Pop() any {
	old := *o
	p := old[len(old)-1]
	old[len(old)-1] = nil
	*o = old[:len(old)-1]
	return p
}

// NewMemStore returns an empty, unbounded MemStore.
func NewMemStore() *MemStore { return &MemStore{pages: map[kvStoreKey]*memPage{}} }

// NewBoundedMemStore returns an empty MemStore that holds at most limit bytes.
func NewBoundedMemStore(limit uint64) *MemStore {
	m := NewMemStore()
	m.limit = limit
	return m
}

// SetLimit bounds the store at limit bytes, evicting until it fits; zero
// removes the bound.
func (m *MemStore) SetLimit(limit uint64) {
	m.mu.Lock()
	m.limit = limit
	m.trim()
	m.mu.Unlock()
}

// Bytes is what the store holds now.
func (m *MemStore) Bytes() uint64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.bytes
}

// MemStoreStats is what a MemStore has served: pages found and missed by
// Get, taken by Set, and dropped to stay under its limit.
type MemStoreStats struct {
	Hits, Misses, Sets, Evicted int64
	Pages                       int
	Bytes, Limit                uint64
	// Pinned is the pages a Pinned view holds out of eviction.
	Pinned int
}

// Stats reports the store's counters.
func (m *MemStore) Stats() MemStoreStats {
	m.mu.Lock()
	defer m.mu.Unlock()
	pinned := len(m.pages) - len(m.order)
	return MemStoreStats{Hits: m.hits, Misses: m.misses, Sets: m.sets, Evicted: m.evicted,
		Pages: len(m.pages), Bytes: m.bytes, Limit: m.limit, Pinned: pinned}
}

// tick advances the pass clock when an operation starts a prompt over. m.mu
// held.
func (m *MemStore) tick(layer, index int) uint64 {
	if index != 0 {
		return m.pass
	}
	if _, again := m.heads[layer]; again || m.heads == nil {
		m.pass++
		m.heads = map[int]struct{}{}
	}
	m.heads[layer] = struct{}{}
	return m.pass
}

// Get writes the held page into page, or returns ErrNoPage. The copy is made
// under the lock: an eviction unmaps the page's memory.
func (m *MemStore) Get(cacheId string, layer, index int, page io.Writer) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	pass := m.tick(layer, index)
	p, ok := m.pages[kvStoreKey{cacheId, layer, index}]
	if !ok {
		m.misses++
		return ErrNoPage
	}
	m.hits++
	p.pass = pass
	if p.pins == 0 {
		heap.Fix(&m.order, p.at)
	}
	_, err := page.Write(p.b)
	return err
}

// Set reads the page and holds a copy, replacing one held under the same key.
// A page larger than the whole limit is not held.
func (m *MemStore) Set(cacheId string, layer, index int, page io.Reader) error {
	return m.set(cacheId, layer, index, page, false)
}

func (m *MemStore) set(cacheId string, layer, index int, page io.Reader, pin bool) error {
	src, err := io.ReadAll(page)
	if err != nil {
		return err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	pass := m.tick(layer, index)
	if !pin && m.limit > 0 && cost(len(src), true) > m.limit {
		return nil
	}
	b, mapped, err := kernels.MapOffHeap(len(src))
	if err != nil || len(b) < len(src) {
		b, mapped = make([]byte, len(src)), false
	}
	b = b[:len(src)]
	copy(b, src)
	k := kvStoreKey{cacheId, layer, index}
	pins := 0
	if old, ok := m.pages[k]; ok {
		// A page offered again keeps the pins it had: the same name is the
		// same bytes, and a parked session still needs them.
		pins = old.pins
		m.remove(old)
	}
	if pin {
		pins++
	}
	p := &memPage{key: k, b: b, mapped: mapped, pass: pass, at: -1, pins: pins}
	m.pages[k] = p
	if pins == 0 {
		heap.Push(&m.order, p)
	}
	m.bytes += cost(len(b), mapped)
	m.sets++
	m.trim()
	return nil
}

// Drop forgets every page held for cacheId.
func (m *MemStore) Drop(cacheId string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	for k, p := range m.pages {
		if k.cache == cacheId {
			m.remove(p)
		}
	}
	return nil
}

// Close gives back every page's memory. The store is empty afterwards and
// still usable.
func (m *MemStore) Close() {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, p := range m.pages {
		m.remove(p)
	}
}

// trim evicts until the store fits its limit. m.mu held.
func (m *MemStore) trim() {
	for m.limit > 0 && m.bytes > m.limit && len(m.order) > 0 {
		m.remove(m.order[0])
		m.evicted++
	}
}

// remove forgets one page and frees its memory. m.mu held.
func (m *MemStore) remove(p *memPage) {
	if p.pins == 0 {
		heap.Remove(&m.order, p.at)
	}
	delete(m.pages, p.key)
	m.bytes -= cost(len(p.b), p.mapped)
	if p.mapped {
		kernels.UnmapOffHeap(p.b)
	}
}

// cost is what a page of n bytes holds: a mapping is whole OS pages, so a
// small page (a tail, a logits row) is charged the page it occupies.
func cost(n int, mapped bool) uint64 {
	if mapped {
		const osPage = 4096
		n = (n + osPage - 1) / osPage * osPage
	}
	return uint64(n)
}

// PinnedStore is a view of a MemStore whose pages are held out of eviction
// until Release: what a parked session's history goes into, since a bound
// that dropped one of them would leave the session nothing to resume from. A
// pinned page counts against the store's bytes like any other, so a store
// holding parked sessions evicts more of everything else; it may exceed its
// limit when pinned pages alone do.
type PinnedStore struct {
	m    *MemStore
	mu   sync.Mutex
	keys []kvStoreKey
	done bool
}

// Pinned returns a view whose Sets are pinned until Release.
func (m *MemStore) Pinned() *PinnedStore { return &PinnedStore{m: m} }

// Get reads the page from the store.
func (p *PinnedStore) Get(cacheId string, layer, index int, page io.Writer) error {
	return p.m.Get(cacheId, layer, index, page)
}

// Set holds the page pinned, or, after Release, as an ordinary page.
func (p *PinnedStore) Set(cacheId string, layer, index int, page io.Reader) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if err := p.m.set(cacheId, layer, index, page, !p.done); err != nil {
		return err
	}
	if !p.done {
		p.keys = append(p.keys, kvStoreKey{cacheId, layer, index})
	}
	return nil
}

// Drop forgets every page held for cacheId.
func (p *PinnedStore) Drop(cacheId string) error { return p.m.Drop(cacheId) }

// Release unpins every page this view pinned: they stay in the store as
// ordinary pages, evicted in their turn, so a later prompt with the same
// prefix may still restore them.
func (p *PinnedStore) Release() {
	p.mu.Lock()
	keys := p.keys
	p.keys, p.done = nil, true
	p.mu.Unlock()
	m := p.m
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, k := range keys {
		if pg, ok := m.pages[k]; ok && pg.pins > 0 {
			if pg.pins--; pg.pins == 0 {
				heap.Push(&m.order, pg)
			}
		}
	}
	m.trim()
}
