//go:build (amd64 || arm64) && jitllmtest

package nn

import (
	"math"
	"math/rand"
	"testing"

	"github.com/samyfodil/jitllm/jit/cpu"
)

// TestMoETopK32JITIsKeyedByTier checks a forced SSE run actually selects an
// SSE kernel: a cache keyed without the tier would hand it the AVX2 kernel an
// earlier call mapped, which answers correctly. So this warms the host's own
// tier first, then forces SSE and requires a new kernel to be mapped on the
// SSE tier and none on the AVX2 one.
func TestMoETopK32JITIsKeyedByTier(t *testing.T) {
	const n, k = 128, 8
	rng := rand.New(rand.NewSource(11))
	p := make([]float32, n)
	for i := range p {
		p[i] = rng.Float32()
	}
	for i := n / 2; i < n; i++ {
		p[i] = p[i-n/2] // every pair a tie
	}

	warm := NewMoERoute(k)
	if !MoETopK32JIT(p, true, warm) {
		t.Fatalf("no kernel on the host's own tier (%v)", cpu.HostTier())
	}

	old := cpu.ForceTierForTest(cpu.TierSSE)
	defer cpu.ForceTierForTest(old)
	if cpu.HostTier() != cpu.TierSSE {
		t.Skipf("the tier could not be forced to SSE on this build (host tier %v)", cpu.HostTier())
	}
	before := cpu.MappedByTier()

	got := NewMoERoute(k)
	if !MoETopK32JIT(p, true, got) {
		t.Fatal("the forced SSE tier has no MoE top-k kernel")
	}

	after := cpu.MappedByTier()
	if d := after[cpu.TierAVX2] - before[cpu.TierAVX2]; d != 0 {
		t.Errorf("%d AVX2 kernel(s) were mapped while the SSE tier was forced", d)
	}
	if after[cpu.TierSSE] == before[cpu.TierSSE] {
		t.Error("no SSE-tier kernel was mapped -- the cache is not keyed by tier and " +
			"this call reused the AVX2 kernel warmed above")
	}

	// And the two tiers agree bit for bit, which is what makes one scoreboard
	// out of two encodings.
	for i := 0; i < k; i++ {
		if got.Sel[i] != warm.Sel[i] || got.Ord[i] != warm.Ord[i] {
			t.Fatalf("SSE sel %v ord %v, AVX2 %v %v", got.Sel, got.Ord, warm.Sel, warm.Ord)
		}
		if math.Float32bits(got.Wt[i]) != math.Float32bits(warm.Wt[i]) ||
			math.Float32bits(got.OWt[i]) != math.Float32bits(warm.OWt[i]) {
			t.Fatalf("SSE wt[%d] = %v/%v, AVX2 %v/%v", i, got.Wt[i], got.OWt[i], warm.Wt[i], warm.OWt[i])
		}
	}
	if math.Float32bits(got.Sum[0]) != math.Float32bits(warm.Sum[0]) {
		t.Fatalf("SSE sum %v, AVX2 %v", got.Sum[0], warm.Sum[0])
	}
}
