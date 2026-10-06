package model

import "github.com/samyfodil/jitllm/engine/nn"

// testState is a State over the tower's vision segment for a gate that
// encodes on its own: on share's JIT when one is given (a text State's, as
// State.Vision does), on one of its own otherwise, which the State closes. A
// session reaches the segment through State.Vision; this is the same State
// without a text State around it.
func (t *Tower) testState(share ...*nn.JIT) *State {
	if len(share) > 0 {
		return t.newState(share[0], false)
	}
	return t.newState(t.model.newJIT(), true)
}
