//go:build amd64 || arm64

package model

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestSamplerNeverDrawsAMaskedToken: a token whose logit is -inf (a
// grammar's mask) has probability zero and is never drawn, at the
// vocabulary's scale. The growth loop once reallocated its candidate list
// without the candidates already taken, so a Sampler's first draw that needed
// a second round walked zeros and landed on a masked token or on id 0.
func TestSamplerNeverDrawsAMaskedToken(t *testing.T) {
	ninf := float32(math.Inf(-1))
	rng := rand.New(rand.NewPCG(1, 1))
	lg := make([]float32, 128256)
	for _, seed := range []int64{1, 2, 3} {
		s := &Sampler{Temp: 1, Seed: seed}
		for it := range 300 {
			for i := range lg {
				lg[i] = float32(rng.NormFloat64() * 3)
				if rng.IntN(20) == 0 {
					lg[i] = ninf
				}
			}
			if x := s.Sample(lg); lg[x] == ninf {
				t.Fatalf("seed %d draw %d: token %d, whose logit is -inf", seed, it, x)
			}
		}
	}
}
