package model

// Hotness is how often each expert is actually selected, measured while the
// model runs.
//
// It is separate from residency on purpose: residency keeps a last-touch token
// for eviction and exists only when the model does not fit; placement
// (tier.Place) needs a rate per unit whether or not it fits.
//
// It is measured rather than shipped because the hot set depends on the
// workload (prose and code prompts share little of it; see
// docs/engineering-history/model-correctness.md). The cost is one increment per
// selected expert per layer per token.
type Hotness struct {
	// count[layer*NExpert+e] is how many tokens selected that expert.
	count []uint32
	// tokens is how many tokens have been counted, so count/tokens is the
	// probability that gives Unit.Uses its meaning.
	tokens   int
	nExpert  int
	overflow bool
}

func (s *State) initHotness() {
	c := s.c
	if !c.MoE() {
		return
	}
	s.hot = &Hotness{
		count:   make([]uint32, c.NExpert*len(s.m.layers)),
		nExpert: c.NExpert,
	}
}

// count records one layer's selection. sel holds global expert ids.
func (h *Hotness) add(li int, sel []int32) {
	if h == nil {
		return
	}
	for _, e := range sel {
		i := li*h.nExpert + int(e)
		// Saturate rather than wrap, so a long-running server never sees a
		// hot expert's count roll over to near zero.
		if h.count[i] == ^uint32(0) {
			h.overflow = true
			continue
		}
		h.count[i]++
	}
}

// endToken is called once per token, after every layer has routed.
func (h *Hotness) endToken() {
	if h != nil {
		h.tokens++
	}
}

// Uses returns each expert's selection probability, indexed
// [layer*NExpert+expert], and how many tokens it was measured over.
//
// The token count is returned so the caller can decide how much evidence it
// wants: over one token every selected expert reads 1.0, a confident and
// nearly worthless profile.
func (s *State) ExpertUses() (uses []float64, tokens int) {
	h := s.hot
	if h == nil || h.tokens == 0 {
		return nil, 0
	}
	u := make([]float64, len(h.count))
	for i, c := range h.count {
		u[i] = float64(c) / float64(h.tokens)
	}
	return u, h.tokens
}
