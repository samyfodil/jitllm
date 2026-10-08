package sched

import (
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// Two users of one core set share its workers: a second view starts no
// goroutines, regions from both run concurrently without losing or doubling a
// chunk, each view keeps its own participant count, and the workers stop when
// the last view closes.
func TestSharedViewsTakeTurnsOnOneCrew(t *testing.T) {
	cpus := []int{0, 1, 2, 3}
	g0 := runtime.NumGoroutine()
	a := Shared(cpus)
	g1 := runtime.NumGoroutine()
	b := Shared(cpus)
	if g := runtime.NumGoroutine(); g != g1 || g1-g0 != len(cpus)-1 {
		t.Fatalf("goroutines %d -> %d -> %d: want %d workers once, none for the second view", g0, g1, g, len(cpus)-1)
	}
	if a.crew != b.crew {
		t.Fatal("two views of one core set got two crews")
	}

	a.SetParticipants(1)
	if a.N() != 1 || b.N() != len(cpus) {
		t.Fatalf("participants a=%d b=%d: one view's count leaked into the other", a.N(), b.N())
	}

	// A region on a capped view must not use workers above its cap.
	var above atomic.Int64
	a.Do(64, 1, func(w, lo, hi int) {
		if w >= 1 {
			above.Add(1)
		}
	})
	if above.Load() != 0 {
		t.Fatalf("a view capped at 1 participant ran %d chunk(s) on other workers", above.Load())
	}
	a.SetParticipants(len(cpus))

	const regions, total = 2000, 257
	var wg sync.WaitGroup
	for _, p := range []*Pool{a, b} {
		wg.Add(1)
		go func(p *Pool) {
			defer wg.Done()
			for r := 0; r < regions; r++ {
				var sum atomic.Int64
				p.Do(total, 3, func(_, lo, hi int) { sum.Add(int64(hi - lo)) })
				if sum.Load() != total {
					t.Errorf("a region covered %d of %d", sum.Load(), total)
					return
				}
			}
		}(p)
	}
	done := make(chan struct{})
	go func() { wg.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(60 * time.Second):
		// Without the crew's lock two views overwrite each other's job and the
		// acknowledgement count, and a region never ends.
		t.Fatal("two views' regions did not finish: they are not taking turns")
	}
	if pa, _ := a.Regions(); pa < regions {
		t.Fatalf("view a counted %d parallel regions, want at least %d: its counters are not its own", pa, regions)
	}

	a.Close()
	a.Close() // idempotent: must not release b's hold
	if b.crew.stopped.Load() {
		t.Fatal("closing one view stopped the crew another still uses")
	}
	b.Close()
	deadline := time.Now().Add(5 * time.Second)
	for runtime.NumGoroutine() > g0 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if g := runtime.NumGoroutine(); g > g0 {
		t.Fatalf("%d goroutine(s) left after the last view closed", g-g0)
	}
	if c := Shared(cpus); c.crew == b.crew {
		t.Fatal("a closed crew was handed out again")
	} else {
		c.Close()
	}
}

// Crews over overlapping cores share their lock; disjoint ones do not.
func TestOverlappingCrewsShareTheirTurn(t *testing.T) {
	a, b, c := Shared([]int{0, 1}), Shared([]int{1, 2, 3}), Shared([]int{4, 5})
	defer a.Close()
	defer b.Close()
	defer c.Close()
	if a.crew.mu != b.crew.mu {
		t.Fatal("crews over overlapping cores take turns separately")
	}
	if a.crew.mu == c.crew.mu {
		t.Fatal("crews over disjoint cores share a lock they do not need")
	}
}
