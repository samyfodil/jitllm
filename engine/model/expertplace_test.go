//go:build linux

package model

import (
	"testing"

	"github.com/samyfodil/jitllm/internal/testmodels"
	"github.com/samyfodil/jitllm/jit/gpu/tier"
)

// TestParsePlacementExperts: %host and %card name where a block's experts run,
// anything else is refused, and the host cannot take them beside itself.
func TestParsePlacementExperts(t *testing.T) {
	p, err := ParsePlacement("1-3=cuda:0%host,4=cuda:0~%card", false)
	if err != nil {
		t.Fatal(err)
	}
	if p.Blocks[2].Experts != "host" || p.Blocks[4].Experts != "card" || !p.Blocks[4].Stream || p.Blocks[2].On != "cuda:0" {
		t.Fatalf("parsed %+v", p.Blocks)
	}
	for _, bad := range []string{"1=cuda:0%gpu", "1=host%host"} {
		if _, err := ParsePlacement(bad, false); err == nil {
			t.Errorf("%q parsed", bad)
		}
	}
}

// runExperts runs ids through m, its blocks on tier g (the host alone when g
// is nil), and returns the logits.
func runExperts(t *testing.T, m *Model, g *tier.GPU, ids []int32) [][]float32 {
	t.Helper()
	s := m.NewState(32)
	defer s.Close()
	if g != nil {
		s.SetDeviceLayers(g, m.Cfg.NLayer)
	}
	var out [][]float32
	for _, id := range ids {
		l, err := s.Forward(id)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, append([]float32(nil), l...))
	}
	return out
}

// TestExpertPlacementIsHonoured: WithExperts places every mixture block's
// experts off the card even where the blocks fit -- "host" runs them hybrid,
// "card" streams their sheets -- and each run takes its path alone (counted,
// RULE 10) with the host's greedy tokens.
func TestExpertPlacementIsHonoured(t *testing.T) {
	path, ok := existingModel(testmodels.Path("kimik3/Kimi-K3-0.40B.Q8_0.gguf"))
	if !ok {
		t.Skip("MODEL MISSING: kimik3/Kimi-K3-0.40B.Q8_0.gguf -- this gate proved nothing")
	}
	ids := []int32{1008, 10484, 318, 15383, 387, 17374}
	host, err := Open(jlmOf(t, path), noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	want := runExperts(t, host, nil, ids)
	host.Close()
	for _, where := range []string{"host", "card"} {
		t.Run(where, func(t *testing.T) {
			m, err := Open(jlmOf(t, path), noTune, WithKVF16(false), WithExperts(where), WithStreamTrial(false))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			g, err := tier.OpenWith(tier.WithDevices("cuda:0"), tier.WithDeviceTune(tier.TuneOff))
			if err != nil || g == nil {
				noDevice(t, "cuda:0", err)
			}
			defer g.Close()
			got := runExperts(t, m, g, ids)
			st := g.Stats()
			hybrid := where == "host"
			if hybrid && (st.HybridRuns == 0 || st.StreamFills != 0) || !hybrid && (st.StreamFills == 0 || st.HybridRuns != 0) {
				t.Fatalf("experts %s: %d hybrid block-steps, %d fills", where, st.HybridRuns, st.StreamFills)
			}
			for p := range want {
				if argmaxOf(got[p]) != argmaxOf(want[p]) {
					t.Fatalf("pos %d: argmax %d, host %d", p, argmaxOf(got[p]), argmaxOf(want[p]))
				}
			}
			t.Logf("experts %s: %d hybrid block-steps, %d fills", where, st.HybridRuns, st.StreamFills)
		})
	}
}

func argmaxOf(v []float32) int {
	b := 0
	for i := range v {
		if v[i] > v[b] {
			b = i
		}
	}
	return b
}
