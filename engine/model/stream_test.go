//go:build linux

package model

import (
	"math"
	"testing"

	"github.com/samyfodil/jitllm/jit/gpu/tier"
)

// TestStreamedExpertBankMatchesTheResidentOne compares a mixture's routed bank
// assembled per token (StreamExperts) against the same bank held resident on
// the card. The kernel cannot tell a compact bank from the full one (see
// backend.TestIndexedMatVecMatchesACompactBank), so the bar is bit equality:
// sheets uploaded in the wrong order would be finite and fluent. The tuner is
// pinned, and the stream counters show which arm ran.
func TestStreamedExpertBankMatchesTheResidentOne(t *testing.T) {
	// wideMoE is Qwen3-Next's mixture shape (512 experts, 10 routed), so the
	// compact bank is genuinely smaller than the resident one.
	//
	// The page budget must evict: fill() reads only the routed sheets
	// (nn.LayerWeights.EnsureExperts), and on a fully resident container
	// nothing is read, so a wrong sheet offset would be invisible.
	probe := hybridModelOpt(t, hyOpt{moe: true, wideMoE: true})
	page := probe.container.H.PageSize
	probe.Close()
	m := hybridModelOpt(t, hyOpt{moe: true, wideMoE: true, budget: 2 * page})
	defer m.Close()
	if !m.container.CanEvict() {
		t.Fatalf("the budget is %d bytes over pages of %d and nothing is evicted: "+
			"the narrowed expert read would never run", m.container.Budget(), page)
	}
	if m.Cfg.NExpertUsed >= m.Cfg.NExpert || m.Cfg.NExpertUsed < 2 {
		t.Fatalf("the fixture routes %d of %d experts; streaming needs 2 <= used < total",
			m.Cfg.NExpertUsed, m.Cfg.NExpert)
	}

	ids := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	run := func(stream bool) ([][]float32, tier.Stats, int, uint64) {
		t.Helper()
		opts := []tier.Option{tier.WithDevices("gpu:0"), tier.WithDeviceTune(tier.TuneOff)}
		if stream {
			opts = append(opts, tier.WithConfig(func(c *tier.Config) { c.StreamExperts = true }))
		}
		g, err := tier.OpenWith(opts...)
		if err != nil || g == nil {
			noDevice(t, "device", err)
		}
		defer g.Close()
		s := m.NewState(16)
		defer s.Close()
		s.SetDeviceLayers(g, m.Cfg.NLayer)
		placed := s.GPULayers()
		if placed == 0 {
			t.Skipf("the device took no block of this fixture: %s", g.Err())
		}
		out := make([][]float32, len(ids))
		for i, id := range ids {
			l, err := s.Forward(id)
			if err != nil {
				t.Fatalf("stream=%v: %v", stream, err)
			}
			out[i] = append([]float32(nil), l...)
		}
		// Read while the tier is open: after Close the driver reclaims the
		// context and the ledger is gone.
		return out, g.Stats(), placed, g.Bytes()
	}

	want, rs, rp, rb := run(false)
	got, ss, sp, sb := run(true)
	t.Logf("resident: %d blocks placed, %d streamed, %d fills, %d device bytes",
		rp, rs.StreamBlocks, rs.StreamFills, rb)
	t.Logf("streamed: %d blocks placed, %d streamed, %d fills, %d device bytes (%.2fx less)",
		sp, ss.StreamBlocks, ss.StreamFills, sb, float64(rb)/float64(sb))

	// The device bytes are asserted: a tier that uploaded the full bank and
	// also assembled a compact one would give equal logits.
	if sb >= rb {
		t.Errorf("the streamed arm holds %d device bytes against the resident arm's %d: "+
			"the bank is still on the card", sb, rb)
	}

	// The selection check, both ways: the streamed arm must stream, and the
	// resident arm must not (the knob must not leak between tiers).
	if ss.StreamBlocks == 0 || ss.StreamFills == 0 {
		t.Fatalf("the streamed arm placed %d streamed blocks and filled %d times: "+
			"the configuration under test never ran", ss.StreamBlocks, ss.StreamFills)
	}
	if rs.StreamBlocks != 0 || rs.StreamFills != 0 {
		t.Errorf("the RESIDENT arm reports %d streamed blocks and %d fills",
			rs.StreamBlocks, rs.StreamFills)
	}
	if sp != rp {
		t.Errorf("the streamed arm placed %d blocks and the resident one %d: "+
			"the two arms are not the same placement", sp, rp)
	}
	// One fill per streamed block per token, and nothing else: a block that
	// filled twice has uploaded a bank it then overwrote, and one that filled
	// none ran against whatever the buffer held.
	if n := ss.StreamBlocks * len(ids); ss.StreamFills != n {
		t.Errorf("%d fills over %d tokens with %d streamed blocks, want %d",
			ss.StreamFills, len(ids), ss.StreamBlocks, n)
	}

	for p := range ids {
		for i := range want[p] {
			w, gv := want[p][i], got[p][i]
			if math.IsNaN(float64(gv)) || math.IsInf(float64(gv), 0) {
				t.Fatalf("pos %d logit %d is %g streamed against %g resident: "+
					"not finite, which no tolerance can see", p, i, gv, w)
			}
			if w != gv {
				t.Fatalf("pos %d logit %d: streamed %v, resident %v -- a compact "+
					"bank runs the same kernel over the same bytes, so this is "+
					"the sheets or their order, not rounding", p, i, gv, w)
			}
		}
	}
}
