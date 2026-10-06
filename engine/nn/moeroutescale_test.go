package nn

import "testing"

// TestMoERouteRefusesAScaleItWasNotBuiltFor: the gate's Scale only decides
// whether the kernel multiplies; the factor lives in the route's constant
// block, fixed at construction. Calling a route with another scale would
// silently multiply by the constructor's, so it panics.
func TestMoERouteRefusesAScaleItWasNotBuiltFor(t *testing.T) {
	logits := []float32{0.3, -1, 2, 0.5, 1.5, -0.2, 0.9, 0.1}
	g := MoEGate{Norm: true}
	r := NewMoERouteFor(len(logits), 2, g)
	if !MoERouteJIT(logits, g, r) {
		t.Skip("no router kernel on this host")
	}
	defer func() {
		if recover() == nil {
			t.Error("a route built for scale 1 ran under a gate of scale 0.22")
		}
	}()
	g.Scale = 0.22
	MoERouteJIT(logits, g, r)
}
