package gguf

import "testing"

// TestWarmSpanRespectsTheBudget gates the bound on the parallel toucher.
//
// It is a pure-function gate: the cgroup behaviour cannot be asserted from a
// test process, so the decision is factored out and asserted here. A budget of
// 0 means unknown, so "warm everything", not "warm nothing": MemBudget returns
// 0 where there is no cgroup.
func TestWarmSpanRespectsTheBudget(t *testing.T) {
	const gib = uint64(1) << 30
	for _, c := range []struct {
		name      string
		n, budget uint64
		want      uint64
	}{
		{"unknown budget warms everything", 20 * gib, 0, 20 * gib},
		{"budget larger than the file warms everything", 3 * gib, 24 * gib, 3 * gib},
		{"budget equal to the file warms everything", 3 * gib, 3 * gib, 3 * gib},
		{"budget smaller than the file warms exactly the budget", 17*gib + 287, 3 * gib, 3 * gib},
		{"one byte over the budget is still bounded", 3*gib + 1, 3 * gib, 3 * gib},
		{"zero-length mapping is zero", 0, 3 * gib, 0},
		{"zero-length mapping with an unknown budget is zero", 0, 0, 0},
	} {
		t.Run(c.name, func(t *testing.T) {
			got := warmSpan(c.n, c.budget)
			if got != c.want {
				t.Errorf("warmSpan(%d, %d) = %d, want %d", c.n, c.budget, got, c.want)
			}
			// A span is a slice bound: b[:got] must be legal for every input.
			if got > c.n {
				t.Errorf("warmSpan(%d, %d) = %d overruns the mapping", c.n, c.budget, got)
			}
			// And it must never exceed a known budget.
			if c.budget > 0 && got > c.budget {
				t.Errorf("warmSpan(%d, %d) = %d faults past the budget", c.n, c.budget, got)
			}
		})
	}
}
