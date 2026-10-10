package tier

import (
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
)

// TestDeclineNamesTheGraphFeature is RULE 8a's gate on an absence. A graph
// feature that lands on the host alone silently removes an architecture from
// the device, and each feature below produces fluent wrong text if skipped. So
// every feature is either implemented (and names its gate) or declined by
// name.
func TestDeclineNamesTheGraphFeature(t *testing.T) {
	base := nn.LayerPlan{
		NEmbd: 2048, NHead: 16, NKVHead: 2, HeadDim: 128, NRot: 128,
		NFFN: 5120, MaxSeq: 128, RMSEps: 1e-6, RopeBase: 10000, ActWin: 32,
	}
	if r := declineReason(&base); r != "" {
		t.Fatalf("an ordinary attention block was declined: %s -- this test "+
			"would then pass for every case below without proving anything", r)
	}
	// Both directions are the gate: a feature newly accepted without a
	// correctness gate is fluent wrong text, and one newly declined after it
	// was implemented is a silent regression to the host.
	for _, c := range []struct {
		name string
		set  func(*nn.LayerPlan)
		runs bool // the tier implements this; a decline would be the regression
		gate string
	}{
		// Linear attention runs through devTier.emitLinear.
		{"linear attention", func(p *nn.LayerPlan) {
			p.Recurrent = nn.RecurrentPlan{Conv: 4, Chans: 8192, KHeads: 16,
				VHeads: 32, KDim: 128, VDim: 128, StateLen: 524288, ConvState: 24576}
		}, true, "model.TestHybridAttentionBlockOnDevice"},
		// The declined side keeps a member so the table can catch a decline
		// that should happen: a recurrent plan with no key heads has no
		// geometry (everything divides by KHeads) and is refused by name.
		{"linear attention with no key heads", func(p *nn.LayerPlan) {
			p.Recurrent = nn.RecurrentPlan{Conv: 4, Chans: 8192, KHeads: 0,
				VHeads: 32, KDim: 128, VDim: 128, StateLen: 524288, ConvState: 24576}
		}, false, ""},
		// Partial rotary: RoPERows writes only the dimensions it rotates, so
		// the tier copies the tail first (bs.copyQ/bs.copyK).
		{"partial rotary", func(p *nn.LayerPlan) { p.NRot = p.HeadDim / 4 }, true,
			"model.TestHybridAttentionBlockOnDevice and kernels.TestPartialRotaryMatchesTheHost"},
		{"attention output gate", func(p *nn.LayerPlan) { p.AttnOutGate = true }, true,
			"backend.TestSplitHeadGateMatchesTheHostDeinterleave"},
		{"shared expert", func(p *nn.LayerPlan) { p.NFFNShExp = 512 }, true,
			"backend.TestSharedExpertKernelsMatchTheHost"},
		// gpt-oss's three, each a device kernel.
		{"attention sinks", func(p *nn.LayerPlan) { p.AttnSinks = true }, true,
			"backend.TestSoftmaxSinkMatchesTheHost"},
		{"biased mixture experts", func(p *nn.LayerPlan) { p.MoEBias = true }, true,
			"backend.TestIndexedBiasAddMatchesTheHost"},
		{"swiglu-oai", func(p *nn.LayerPlan) { p.Act = nn.ActSwiGLUOAI }, true,
			"backend.TestLayerNormAndAct"},
	} {
		t.Run(c.name, func(t *testing.T) {
			p := base
			c.set(&p)
			r := declineReason(&p)
			if c.runs {
				if r != "" {
					t.Fatalf("%s is implemented and was DECLINED (%s): the block "+
						"silently goes back to the host, which is a regression a "+
						"rate measurement cannot see", c.name, r)
				}
				// Naming the gate keeps "accepted" from meaning "unchecked";
				// model.TestHybridAttentionBlockOnDevice diffs a block carrying
				// these features against the host.
				t.Logf("implemented; gated by %s", c.gate)
				return
			}
			if r == "" {
				t.Fatalf("%s was ACCEPTED: the device would run a block whose "+
					"graph it does not implement, which is fluent wrong text "+
					"rather than an error", c.name)
			}
			t.Logf("declined: %s", r)
		})
	}
}
