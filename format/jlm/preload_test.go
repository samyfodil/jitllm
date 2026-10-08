package jlm

import (
	"sync"
	"testing"
)

// TestAShrinkDuringAPreloadIsHonoured: four pages are in flight -- claimed,
// pinned, not yet read -- when the budget drops to two pages. SetBudget cannot
// take a pinned frame, so for that moment the pages exceed the budget. Once
// the reads land the preload must evict down to the budget and read nothing
// more: a preload never keeps a page the current budget does not allow.
func TestAShrinkDuringAPreloadIsHonoured(t *testing.T) {
	const page, n, depth = 4096, 8, 4
	f := newFake(n, page, 0, Align)
	f.SetBudget(n * page)
	var (
		mu      sync.Mutex
		read    []int
		arrived = make(chan struct{}, depth)
		release = make(chan struct{})
	)
	f.preloadRead = func(i int) error {
		f.mu.Lock()
		f.fakeIn(i, byte(i+1))
		f.hold(i)
		f.mu.Unlock()
		mu.Lock()
		read = append(read, i)
		mu.Unlock()
		arrived <- struct{}{}
		<-release
		f.mu.Lock()
		f.drop(i)
		f.mu.Unlock()
		return nil
	}
	done := make(chan error)
	go func() { done <- f.Preload(depth, nil) }()
	for range depth {
		<-arrived
	}
	f.SetBudget(2 * page)
	if r := f.ResidentBytes(); r <= 2*page {
		t.Fatalf("with %d pages pinned the shrink left %d bytes: the gate holds nothing in flight", depth, r)
	}
	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if r := f.ResidentBytes(); r > 2*page {
		t.Fatalf("after the preload's reads landed %d bytes are resident, over the %d-byte budget", r, 2*page)
	}
	if len(read) != depth {
		t.Fatalf("the preload read %d pages; after the shrink it must stop at the %d in flight", len(read), depth)
	}
}
