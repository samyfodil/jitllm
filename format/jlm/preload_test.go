package jlm

import (
	"sync"
	"testing"
	"time"
)

// TestAShrinkDuringAPreloadIsHonoured: four pages are in flight -- claimed,
// pinned, not yet read -- when the budget drops to two pages. SetBudget cannot
// take a pinned frame, so it waits for those reads, evicts down to the budget
// and only then returns: a caller reading residency after the shrink never
// sees a preloaded page past it, and the preload reads nothing more.
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
	shrunk := make(chan struct{})
	go func() {
		f.SetBudget(2 * page)
		close(shrunk)
	}()
	// The shrink waits for the reads in flight: the gate holds them.
	select {
	case <-shrunk:
		t.Fatal("SetBudget returned with four pinned pages resident past its budget")
	case <-time.After(100 * time.Millisecond):
	}
	if r := f.ResidentBytes(); r <= 2*page {
		t.Fatalf("with %d pages pinned %d bytes are resident: the gate holds nothing in flight", depth, r)
	}
	close(release)
	<-shrunk
	// What a caller reads the moment SetBudget returns.
	if r := f.ResidentBytes(); r > 2*page {
		t.Fatalf("SetBudget returned with %d bytes resident, over its %d-byte budget", r, 2*page)
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if len(read) != depth {
		t.Fatalf("the preload read %d pages; after the shrink it must stop at the %d in flight", len(read), depth)
	}
}
