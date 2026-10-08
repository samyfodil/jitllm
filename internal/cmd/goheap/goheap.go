// Package goheap is the process-wide collector setup both binaries share:
// cmd/jitllm and server/cmd/jitllmd. It reads the environment, which is why it
// sits under a cmd/ directory, not in a library (AGENTS.md Scope: the
// libraries read none), and it is one copy so the two binaries cannot drift
// apart.
package goheap

import (
	"os"
	"runtime/debug"
	"strconv"

	"github.com/samyfodil/jitllm/engine/sched"
)

// Cap tells Go's collector about the cgroup, which it cannot see, and returns
// the memory limit it set, or 0 when it set none. Without it GOGC lets the heap
// target twice the live page frames, far past memory.high, and the run spends
// its time in direct reclaim (AGENTS.md RULE 2f).
//
// The headroom covers what the limit charges and the heap does not: goroutine
// stacks, runtime allocations and device-backend mappings. It is set by a
// binary, not by a library, because debug.SetMemoryLimit is process-wide.
func Cap() uint64 {
	lim := sched.MemLimit()
	// JITLLM_GOMEMLIMIT=0 sets no limit, leaving the collector on GOGC alone:
	// the control arm for an off-heap measurement, with the frames on the heap.
	if lim == 0 || os.Getenv("JITLLM_GOMEMLIMIT") == "0" {
		gogc()
		return 0
	}
	// The same constant sched.MemBudget reserves against, so the budget and
	// the goal cannot drift apart.
	if lim <= sched.GCHeadroom {
		return 0
	}
	set := lim - sched.GCHeadroom
	debug.SetMemoryLimit(int64(set))
	// JITLLM_GC_RESERVE is the bytes the weight budget leaves below the
	// collector's goal: lower it to hold more weights, raise it if
	// GODEBUG=gctrace=1 shows cycles that free nothing. JITLLM_GOGC is the
	// ordinary GOGC, applied after the limit so it composes with it.
	if v := os.Getenv("JITLLM_GC_RESERVE"); v != "" {
		if n, err := strconv.ParseUint(v, 10, 64); err == nil {
			sched.SetGCReserve(n)
		}
	}
	gogc()
	return set
}

// gogc applies JITLLM_GOGC, the ordinary GOGC.
func gogc() {
	if v := os.Getenv("JITLLM_GOGC"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			sched.SetGCPercent(n)
		}
	}
}
