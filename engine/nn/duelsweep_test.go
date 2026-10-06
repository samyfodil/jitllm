package nn

import (
	"slices"
	"testing"

	"github.com/samyfodil/jitllm/format/quant"
)

// TestDuelSweepTriesEveryRung drives a duel over the fused-prefetch ladder's
// shape -- none, word-ahead, two row groups -- on synthetic rates where the
// second rung ties the first and the third wins by 5%. A climb stops at the
// first loss, which is wrong for a ladder not ordered by cost. Violation
// (sweep off): settles at 1, not 3.
func TestDuelSweepTriesEveryRung(t *testing.T) {
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	run := func(sweep bool) int {
		c := &duelTuner{label: "t", key: "t", cands: []int{1, 65, 3}, best: 1, chal: 65, on: true,
			quiet: true, rounds: 4, margin: 1.005, firstMargin: tuneMargin2, sweep: sweep}
		rate := map[int]float64{1: 10.0, 65: 10.0, 3: 10.5}
		for i := 0; c.on && i < 1000; i++ {
			c.observe(rate[c.pick()])
		}
		if c.on {
			t.Fatal("the duel never settled")
		}
		return c.best
	}
	if got := run(true); got != 3 {
		t.Fatalf("sweeping duel settled at %d, want 3 (the rung that wins)", got)
	}
	if got := run(false); got != 1 {
		t.Fatalf("climbing duel settled at %d; this gate's control expects the climb to stop at 1", got)
	}
}

// TestFreshPrefetchDuelStartsWithAPrefetch: until the duel settles, a process
// decodes at its incumbent, and a process too short to settle it starts over
// next time -- so the incumbent is what short sessions run at. It must be a
// prefetch, not none. Violation (none first): the incumbent is 0.
func TestFreshPrefetchDuelStartsWithAPrefetch(t *testing.T) {
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	// With shapes picking (the x86 default) the duel does not run, but its
	// incumbent is still what the batched paths start at.
	if j := NewJIT(2048, 4096, []quant.Type{quant.Q4_K}); j != nil {
		if ft := j.newFPFTuner(); ft != nil && ft.d.best-ft.d.offset == 0 {
			t.Fatal("with shapes picking, the default starts at no prefetch")
		}
		j.Close()
	}
	// The whole-token duel runs only where shapes do not pick (JIT.distPick).
	j := NewJIT(2048, 4096, []quant.Type{quant.Q4_K}, WithMatVecPick(-1))
	if j == nil {
		t.Skip("no generated tier")
	}
	defer j.Close()
	ft := j.newFPFTuner()
	if ft == nil {
		t.Skip("this tier's fused kernel has no prefetch form")
	}
	if !ft.d.on {
		t.Fatal("a fresh cache produced a settled duel -- the cache was not fresh")
	}
	if got := ft.d.best - ft.d.offset; got == 0 {
		t.Fatalf("a fresh duel decodes at distance %d (no prefetch) until it settles", got)
	}
	if !slices.Contains(prefetchDistances, 0) {
		t.Fatal("no prefetch is no longer a rung, so a host where it wins cannot reach it")
	}
}
