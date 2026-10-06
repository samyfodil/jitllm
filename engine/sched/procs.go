package sched

import (
	"runtime"
	"sync"
)

// GOMAXPROCS is process state, and several models in one process each raise it
// for their own prefill. A save-and-restore per model breaks as soon as two of
// those phases overlap: the first to end restores a width the second still
// needs, and the second then restores the first's raised value. procs holds
// the live requests instead and sets the maximum of them, and puts back the
// value it found once the last one ends.
var procs struct {
	mu   sync.Mutex
	base int         // GOMAXPROCS when the first live request arrived
	live map[int]int // requested width -> live requests at that width
}

// RaiseProcs asks for GOMAXPROCS of at least n until the returned release
// runs. Concurrent requests get their maximum; when the last is released the
// value the first one found is restored. release is safe to call twice.
func RaiseProcs(n int) (release func()) {
	procs.mu.Lock()
	defer procs.mu.Unlock()
	if len(procs.live) == 0 {
		procs.base = runtime.GOMAXPROCS(0)
		procs.live = map[int]int{}
	}
	procs.live[n]++
	setProcsLocked()
	var once sync.Once
	return func() {
		once.Do(func() {
			procs.mu.Lock()
			defer procs.mu.Unlock()
			if procs.live[n]--; procs.live[n] == 0 {
				delete(procs.live, n)
			}
			setProcsLocked()
		})
	}
}

// setProcsLocked sets GOMAXPROCS to the widest live request, or the base when
// none is wider or none is live.
func setProcsLocked() {
	want := procs.base
	for n := range procs.live {
		want = max(want, n)
	}
	if runtime.GOMAXPROCS(0) != want {
		runtime.GOMAXPROCS(want)
	}
}
