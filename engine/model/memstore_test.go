package model

import (
	"bytes"
	"errors"
	"testing"

	"github.com/samyfodil/jitllm/internal/testmodels"
)

// TestBoundedMemStoreEvictsTheOldestTailFirst holds a MemStore to its limit:
// it never holds more than it was given, it drops from the pass used longest
// ago (a Get counts as use) and, within it, the page furthest into the
// prompt, and its byte count goes back to zero when everything is dropped.
func TestBoundedMemStoreEvictsTheOldestTailFirst(t *testing.T) {
	page := bytes.Repeat([]byte{7}, 4096)
	m := NewBoundedMemStore(3 * 4096)
	defer m.Close()
	for i := 0; i < 3; i++ {
		if err := m.Set("c", 0, i, bytes.NewReader(page)); err != nil {
			t.Fatal(err)
		}
	}
	var sink bytes.Buffer
	// Page 0 is used again in a new pass, so pages 1 and 2 are the oldest
	// pass, and 2 is its tail.
	if err := m.Get("c", 0, 0, &sink); err != nil {
		t.Fatal(err)
	}
	if err := m.Set("c", 0, 3, bytes.NewReader(page)); err != nil {
		t.Fatal(err)
	}
	if got := m.Bytes(); got > 3*4096 {
		t.Fatalf("the store holds %d bytes past its %d limit", got, 3*4096)
	}
	if _, ok := m.pages[kvStoreKey{"c", 0, 2}]; ok {
		t.Fatal("page 2, the oldest pass's tail, is still held")
	}
	for _, i := range []int{0, 1, 3} {
		sink.Reset()
		if err := m.Get("c", 0, i, &sink); err != nil {
			t.Fatalf("page %d was evicted instead of the oldest pass's tail: %v", i, err)
		}
		if !bytes.Equal(sink.Bytes(), page) {
			t.Fatalf("page %d came back changed", i)
		}
	}
	st := m.Stats()
	if st.Evicted != 1 || st.Pages != 3 {
		t.Fatalf("stats %+v: want one eviction and three pages", st)
	}
	m.SetLimit(4096)
	if st := m.Stats(); st.Pages != 1 || st.Bytes != 4096 {
		t.Fatalf("lowering the limit left %+v", st)
	}
	if _, ok := m.pages[kvStoreKey{"c", 0, 0}]; !ok {
		t.Fatal("lowering the limit kept a page other than the prompt's head")
	}
	if err := m.Drop("c"); err != nil {
		t.Fatal(err)
	}
	if m.Bytes() != 0 {
		t.Fatalf("an emptied store still counts %d bytes", m.Bytes())
	}
	// A page bigger than the whole limit is not held, rather than evicting
	// everything for nothing.
	if err := m.Set("c", 0, 9, bytes.NewReader(bytes.Repeat([]byte{1}, 8192))); err != nil {
		t.Fatal(err)
	}
	if m.Bytes() != 0 {
		t.Fatal("a page larger than the limit was held")
	}
}

// TestABoundedStoreChangesNoToken runs a prompt twice through a store bounded
// below what the first pass wrote: the second pass restores less than an
// unbounded store would (so the bound bit), and generates the same tokens as
// a run with no store at all.
func TestABoundedStoreChangesNoToken(t *testing.T) {
	path := jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"))
	m, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	const maxSeq, gen, page = 256, 12, 8
	prompt := make([]int32, 0, 70)
	for i := 0; i < 70; i++ {
		prompt = append(prompt, int32(300+i*7))
	}
	run := func(st KVStore) ([]int32, int) {
		defer m.setKVPageForTest(page)()
		s := m.NewState(maxSeq)
		defer s.Close()
		var lg []float32
		var err error
		if st != nil {
			s.SetKVStore(st)
			mustKey(t, s, "tinyllama/bounded")
			lg, err = s.PrefillCached(prompt)
		} else {
			lg, err = s.Prefill(prompt)
		}
		if err != nil {
			t.Fatal(err)
		}
		out := make([]int32, 0, gen)
		for i := 0; i < gen; i++ {
			id := int32(argmax(lg))
			out = append(out, id)
			if lg, err = s.Forward(id); err != nil {
				t.Fatal(err)
			}
		}
		return out, s.KVRestored()
	}
	want, _ := run(nil)

	whole := NewMemStore()
	defer whole.Close()
	run(whole)
	_, all := run(whole)
	if all == 0 {
		t.Fatal("an unbounded store restored nothing: the gate below would compare two cold runs")
	}

	limit := whole.Bytes() / 3
	bounded := NewBoundedMemStore(limit)
	defer bounded.Close()
	run(bounded)
	got, some := run(bounded)
	if some >= all {
		t.Fatalf("a store bounded to a third restored %d positions, an unbounded one %d: the bound never bit", some, all)
	}
	if some == 0 {
		t.Fatal("the bounded store restored nothing: it evicted the prompt's head and kept pages no restore can reach")
	}
	if bounded.Stats().Evicted == 0 {
		t.Fatal("the bounded store evicted nothing")
	}
	if bounded.Bytes() > limit {
		t.Fatalf("the bounded store holds %d bytes past its %d limit", bounded.Bytes(), limit)
	}
	sameTokens(t, "bounded store", want, got)
	t.Logf("unbounded restored %d, bounded %d of %d positions; tokens identical", all, some, len(prompt))
}

// TestAnInterruptedPrefillStopsBetweenChunks: an interrupt that answers true
// after the first chunk stops the prefill there with ErrPrefillInterrupted,
// and the State, reset, answers as a fresh one does.
func TestAnInterruptedPrefillStopsBetweenChunks(t *testing.T) {
	path := jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"))
	m, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	s := m.NewState(2048)
	defer s.Close()
	prompt := make([]int32, 0, 3*MaxPrefillChunk)
	for i := 0; i < cap(prompt); i++ {
		prompt = append(prompt, int32(300+i%900))
	}
	want, err := s.Prefill(prompt)
	if err != nil {
		t.Fatal(err)
	}
	want = append([]float32(nil), want...)
	s.Reset()

	asked := 0
	s.SetPrefillInterrupt(func() bool { asked++; return asked > 1 })
	if _, err := s.Prefill(prompt); !errors.Is(err, ErrPrefillInterrupted) {
		t.Fatalf("an interrupted prefill returned %v", err)
	}
	if asked != 2 {
		t.Fatalf("the interrupt was asked %d times; want once per chunk until it said stop (2)", asked)
	}
	if p := s.Pos(); p == 0 || p >= len(prompt) {
		t.Fatalf("the interrupted prefill stopped at position %d of %d: not between chunks", p, len(prompt))
	}
	s.SetPrefillInterrupt(nil)
	s.Reset()
	got, err := s.Prefill(prompt)
	if err != nil {
		t.Fatal(err)
	}
	if argmax(got) != argmax(want) {
		t.Fatalf("after an interrupt and a reset the prefill picks %d, a clean one %d", argmax(got), argmax(want))
	}
}
