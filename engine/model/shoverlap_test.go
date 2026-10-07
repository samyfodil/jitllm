package model

import (
	"testing"

	"github.com/samyfodil/jitllm/internal/testmodels"
)

// TestSharedOverlapIsTheSameSum: Kimi-K3's shared experts run behind the
// routed read, and the logits are the ones the serial order gives, bit for
// bit, on a page budget that evicts experts so the reads are real. The
// counter says which arm ran.
func TestSharedOverlapIsTheSameSum(t *testing.T) {
	path, ok := existingModel(testmodels.Path("kimik3/Kimi-K3-0.40B.Q8_0.gguf"))
	if !ok {
		t.Skip("MODEL MISSING: kimik3/Kimi-K3-0.40B.Q8_0.gguf -- this gate proved nothing")
	}
	src := jlmOf(t, path)
	run := func(on bool) ([][]float32, int, int64) {
		probe, err := Open(src, noTune)
		if err != nil {
			t.Fatal(err)
		}
		budget := probe.container.H.PageSize * 3
		probe.Close()
		m, err := Open(src, noTune, WithKVF16(false), WithPageBudget(budget), WithSharedOverlap(on))
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		s := m.NewState(32)
		defer s.Close()
		var out [][]float32
		for _, id := range []int32{1, 2, 3, 4, 5, 6, 7, 8} {
			l, err := s.Forward(id)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, append([]float32(nil), l...))
		}
		return out, s.ShOverlaps, m.container.Reads()
	}
	want, n0, _ := run(false)
	got, n1, reads := run(true)
	if n0 != 0 || n1 == 0 {
		t.Fatalf("overlaps: %d with it off, %d with it on", n0, n1)
	}
	if reads == 0 {
		t.Fatal("the overlapped run read nothing: the routed read it overlaps never happened")
	}
	for p := range want {
		for i := range want[p] {
			if want[p][i] != got[p][i] {
				t.Fatalf("pos %d logit %d: overlapped %v, serial %v", p, i, got[p][i], want[p][i])
			}
		}
	}
	t.Logf("%d layers overlapped, %d reads, logits bit-identical", n1, reads)
}
