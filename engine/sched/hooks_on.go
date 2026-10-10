//go:build jitllmtest

package sched

import "sync/atomic"

// onWorkerWake, set by a test (-tags jitllmtest), runs in worker id after it
// has seen a region and before it tries to join it: a test that blocks there
// stands for a worker the OS or the Go scheduler keeps off a CPU. Atomic,
// because a worker let go reads it while the test that set it clears it.
var onWorkerWake atomic.Pointer[func(p *Pool, id int)]

func workerWake(p *Pool, id int) {
	if f := onWorkerWake.Load(); f != nil {
		(*f)(p, id)
	}
}
