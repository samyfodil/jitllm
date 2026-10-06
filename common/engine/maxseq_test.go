package engine

import "testing"

// The context window is what was asked for, bounded by what the model knows.
//
// The model's own context is a ceiling; asking for less is legitimate, since
// the KV cache is allocated up front.
func TestTheContextWindowIsBoundedByTheModel(t *testing.T) {
	for _, c := range []struct {
		name           string
		modelCtx, want int
		ask            int
	}{
		{"asking for less is honoured", 8192, 2048, 2048},
		{"asking for more is clamped to the model", 2048, 2048, 8192},
		{"asking for nothing takes the engine default", 8192, defaultMaxSeq, 0},
		{"a model smaller than the default wins anyway", 1024, 1024, 0},
		{"a model that declares none takes the default", 0, defaultMaxSeq, 0},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := maxSeqFor(c.modelCtx, c.ask); got != c.want {
				t.Errorf("maxSeqFor(model %d, ask %d) = %d, want %d",
					c.modelCtx, c.ask, got, c.want)
			}
		})
	}
}
