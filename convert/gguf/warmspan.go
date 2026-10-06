package gguf

// warmSpan is how many bytes of a len-n mapping it is worth faulting in ahead
// of use, given a budget. A budget of 0 is "unknown", which means warm
// everything.
//
// An unbounded toucher on a model larger than the cgroup is a reclaim storm:
// every page it wins is reclaimed before the kernels reach it, and the load
// hangs rather than slows. This bounds the toucher only; it is not readahead
// advice.
//
// The prefix is warmed because it holds the header and the early blocks, which
// prefill touches first. Per-range prefetch for a sparse model is a different
// mechanism (File.LayerRange plus Evict).
func warmSpan(n, budget uint64) uint64 {
	if budget == 0 || budget >= n {
		return n // unknown budget, or it all fits: warm everything
	}
	return budget
}
