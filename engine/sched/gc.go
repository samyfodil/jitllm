package sched

import (
	"runtime/debug"
	"sync/atomic"
)

// GCHeadroom is what the Go collector is not allowed to plan into, out of the
// cgroup's hard limit. cmd/jitllm hands the runtime `MemLimit() - GCHeadroom`
// and MemBudget reserves against the same figure, so the two cannot drift.
const GCHeadroom = 1 << 30

// gcReserve is how much of the collector's usable heap the weight budget leaves
// free. Zero means the derived default.
var gcReserve atomic.Uint64

// SetGCReserve fixes how many bytes the weight budget leaves below the Go
// collector's goal. Zero restores the derived default.
//
// The collector's goal under a memory limit is `limit - non-heap`, and page
// frames are permanently live, so a weight budget close to the limit leaves a
// live set sitting on the goal: the collector runs back to back and frees
// nothing. The cliff is sharp (a few hundred MiB separate 6 cycles from 648),
// so this is a reserve, not a tuning curve. The measurements are in
// docs/engineering-history/scheduling-and-measurement.md (AGENTS.md RULE 2f).
func SetGCReserve(b uint64) { gcReserve.Store(b) }

// SetGCPercent sets the collector's target growth, the GOGC knob, and returns
// what it was. It is here rather than left to the environment for the reason
// every other knob in this engine is: a library reads no environment, and the
// one caller that knows the workload is the one that should decide.
//
// It is not the fix for the spiral SetGCReserve prevents: under a memory limit
// the goal is `min(live*(1+p/100), limit-nonheap)`, so once live is at the
// limit no value of p moves it.
func SetGCPercent(p int) int { return debug.SetGCPercent(p) }

// GCUsable is the heap the collector is actually given: the cgroup limit less
// GCHeadroom, or 0 when there is no cgroup and the collector is unconstrained.
func GCUsable() uint64 {
	lim := MemLimit()
	if lim <= GCHeadroom {
		return 0
	}
	return lim - GCHeadroom
}

// weightsOffHeap says the weight memory -- format/jlm's page frames and dense
// region -- lives outside the Go heap, as it does wherever the platform has
// anonymous mmap (jlm.SetOffHeap). There it is not in the collector's goal,
// and the budget needs no reserve against it.
var weightsOffHeap atomic.Bool

func init() { weightsOffHeap.Store(offHeapPlatform) }

// SetWeightsOffHeap says whether the weight memory is off the Go heap, and
// reports what it was: off, the weight budget keeps GCBudgetCap's reserve. It
// pairs with jlm.SetOffHeap, which decides where the memory actually goes.
func SetWeightsOffHeap(on bool) bool { return weightsOffHeap.Swap(on) }

// WeightsOffHeap reports whether the weight memory is off the Go heap.
func WeightsOffHeap() bool { return weightsOffHeap.Load() }

// GCBudgetCap is the largest weight budget that leaves the collector room, or 0
// when there is no limit to reserve against or the weights are off the heap,
// where the collector's goal never sees them. Every path that sizes the weight
// budget has to pass through it -- MemBudget does, and so does cmd/jitllm's
// O_DIRECT raise.
//
// The reserve is a quarter of the usable heap, capped: a quarter lands a
// 12 GiB cgroup in the quiet zone, and the cap prevents a small margin
// from being wrongly applied to a large server, where the non-heap overhead is the same.
func GCBudgetCap() uint64 { return gcBudgetCap() }

func gcBudgetCap() uint64 {
	if weightsOffHeap.Load() {
		return 0
	}
	u := GCUsable()
	if u == 0 {
		return 0
	}
	r := gcReserve.Load()
	if r == 0 {
		r = u / 4
		if r > 8<<30 {
			r = 8 << 30
		}
	}
	if r >= u {
		return 0
	}
	return u - r
}
