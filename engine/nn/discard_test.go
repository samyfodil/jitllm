package nn

import (
	"testing"

	"github.com/samyfodil/jitllm/format/quant"
)

// TestDiscardedTokensAreNotDuelSamples runs decode tokens that each call
// DiscardToken (a mixture's token that waited on expert pages) and demands the
// fused-prefetch duel saw none of them, then runs plain tokens and demands it
// saw those. Violation (DiscardToken ignored): the duel advances on the first
// batch.
func TestDiscardedTokensAreNotDuelSamples(t *testing.T) {
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	// The whole-token duel runs only where shapes do not pick (JIT.distPick).
	j := NewJIT(2048, 4096, []quant.Type{quant.Q4_K}, WithMatVecPick(-1))
	if j == nil {
		t.Skip("no generated tier")
	}
	defer j.Close()
	j.AddShape(quant.Q4_K, 2048)
	j.TokenStart()
	j.TokenEnd()
	if j.fpf == nil {
		t.Skip("this tier's fused kernel has no prefetch form")
	}
	if !j.fpf.d.on {
		t.Fatal("a fresh cache produced a settled duel")
	}
	seen := func() int { return j.fpf.skip + j.fpf.d.turn }
	before := seen()
	for range 4 * tuneSkip {
		j.TokenStart()
		j.DiscardToken()
		j.TokenEnd()
	}
	if got := seen(); got != before {
		t.Fatalf("the duel counted %d discarded tokens", got-before)
	}
	for range 4 * tuneSkip {
		j.TokenStart()
		j.TokenEnd()
	}
	if seen() == before {
		t.Fatal("the duel counted no plain tokens either -- this gate proved nothing")
	}
}
