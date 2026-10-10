package model

import (
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestScoreRowsFollowTheReach: a State made at the model's whole context holds
// prompt and batch score rows for the keys its rows reach, not for the
// context. Sized at maxSeq, a 170-token prompt on a 131072-position model held
// 89 MB of scores per State, and a server keeps one per concurrent request. The
// violation is sizing them at attStride: the first assertion fails.
func TestScoreRowsFollowTheReach(t *testing.T) {
	path, ok := existingModel(testmodels.Path("Qwen3-MOE-4x0.6B-Q4_K_M.gguf"))
	if !ok {
		t.Skip("MODEL MISSING: Qwen3-MOE-4x0.6B-Q4_K_M (JITLLM_MODELS) -- this gate proved nothing")
	}
	m, err := Open(jlmOf(t, path))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if m.Cfg.NCtx < 8*scoreReachMin {
		t.Fatalf("context %d is too short to tell the reach from the context", m.Cfg.NCtx)
	}
	ids := make([]int32, 300)
	for i := range ids {
		ids[i] = int32((7 + i*13) % m.Cfg.NVocab)
	}
	s := m.NewState(m.Cfg.NCtx)
	defer s.Close()
	if _, err := s.Prefill(ids[:40]); err != nil {
		t.Fatal(err)
	}
	full := 40 * s.attStride
	if got := cap(s.battf); got >= full || got > 40*attStride(scoreReachMin+m.sinkSlot()) {
		t.Fatalf("40 prompt rows hold %d score floats; the context would be %d, the reach %d",
			got, full, 40*attStride(scoreReachMin+m.sinkSlot()))
	}
	// Past the first reach the rows re-size by doubling, and the answer is
	// the one a State sized for exactly this prompt gives.
	lg, err := s.Prefill(ids[40:])
	if err != nil {
		t.Fatal(err)
	}
	if s.scoreReach != 2*scoreReachMin {
		t.Fatalf("reach %d after 300 positions, want %d", s.scoreReach, 2*scoreReachMin)
	}
	r := m.NewState(len(ids) + 1)
	defer r.Close()
	if _, err := r.Prefill(ids[:40]); err != nil {
		t.Fatal(err)
	}
	want, err := r.Prefill(ids[40:])
	if err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if lg[i] != want[i] {
			t.Fatalf("logit %d: %v at the whole context, %v at the prompt's", i, lg[i], want[i])
		}
	}

	// A batch step: three rows, the score rows a (row, head) each.
	b := m.NewBatch(3, m.Cfg.NCtx)
	defer b.Close()
	for step := 0; step < 4; step++ {
		if _, err := b.ForwardBatch([]int32{1, 2, 3}); err != nil {
			t.Fatal(err)
		}
	}
	if got, full := len(b.bathf), 3*m.Cfg.NHead*b.attStride; got >= full {
		t.Fatalf("a batch step at position 4 holds %d score floats, the context's %d", got, full)
	}
}
