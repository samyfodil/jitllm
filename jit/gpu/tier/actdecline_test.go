package tier

import (
	"strings"
	"testing"

	"github.com/samyfodil/jitllm/engine/nn"
)

// TestEveryActivationHasADeviceKernel: every activation the host bakes has a
// device kernel, and a kind in a shape it does not exist in (a gated one in a
// vision block's ungated MLP) is declined by name rather than lowered as
// something else.
func TestEveryActivationHasADeviceKernel(t *testing.T) {
	p := visionPlan()
	for _, k := range []nn.ActKind{nn.ActSiLU, nn.ActGELU, nn.ActQuickGELU} {
		p.Act = k
		if r := declineReason(p); r != "" {
			t.Fatalf("a %v vision block is declined for %q", k, r)
		}
	}
	p.Act = nn.ActSwiGLUOAI
	r := declineReason(p)
	if r == "" {
		t.Fatal("a vision block with swiglu-oai, a GATED kind, is accepted: its " +
			"ungated MLP has nothing for the kind to gate")
	}
	if !strings.Contains(r, "swiglu-oai") {
		t.Errorf("the decline says %q, which does not name the activation", r)
	}
}

// TestDeclineReasonReachesLastErr: prepLayer must record its decline reason, or
// a block sent to the host looks exactly like one the tier never saw.
func TestDeclineReasonReachesLastErr(t *testing.T) {
	g := newDevice(&devShared{})
	p := visionPlan()
	p.Act = nn.ActSwiGLUOAI // a gated kind in an ungated block: still declined
	// The recover is part of the assertion: with no device behind this
	// devTier, prepLayer can only return by declining before any allocation,
	// where declineReason is placed on purpose. A removed decline walks into
	// the scratch builder and panics.
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("prepLayer went past the decline and panicked (%v): a declined "+
				"block reached allocation instead of being refused before it", r)
		}
	}()
	if g.prepLayer(0, 0, p, &nn.LayerWeights{}, false) {
		t.Fatal("prepLayer accepted a vision block with a gated activation")
	}
	if g.LastErr == "" || !strings.Contains(g.LastErr, "swiglu-oai") {
		t.Fatalf("LastErr is %q; a decline nobody can read is an absence", g.LastErr)
	}
	t.Logf("LastErr: %s", g.LastErr)
}
