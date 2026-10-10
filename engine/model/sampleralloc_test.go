package model

import (
	"math/rand"
	"sort"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
)

// oldSampler is engine/model/sample.go's Sampler before the kernels replaced it. It
// is kept only as the measured Go arm of the allocation census and the rate
// A/B (samplerperf_test.go); it lives in a test file so shipping code
// cannot call it.
type oldSampler struct {
	Temp, TopP, MinP, RepeatPen float64
	TopK, RepeatLastN           int
	Seed                        int64
	rng                         *rand.Rand
	buf                         []Cand
	hist                        []int32
}

func (s *oldSampler) Observe(t int32) {
	if s.RepeatPen == 1 || s.RepeatPen == 0 {
		return
	}
	n := s.RepeatLastN
	if n <= 0 {
		n = 64
	}
	s.hist = append(s.hist, t)
	if len(s.hist) > n {
		s.hist = s.hist[len(s.hist)-n:]
	}
}

func (s *oldSampler) Sample(logits []float32) int32 {
	if s.Temp <= 0 {
		return Greedy(logits)
	}
	if s.rng == nil {
		s.rng = rand.New(rand.NewSource(s.Seed))
	}
	c := s.buf[:0]
	pen := map[int32]bool{}
	if s.RepeatPen > 1 {
		for _, t := range s.hist {
			pen[t] = true
		}
	}
	for i, v := range logits {
		if pen[int32(i)] {
			if v > 0 {
				v /= float32(s.RepeatPen)
			} else {
				v *= float32(s.RepeatPen)
			}
		}
		c = append(c, Cand{int32(i), v})
	}
	s.buf = c
	sort.Slice(c, func(i, j int) bool { return c[i].Logit > c[j].Logit })
	if s.TopK > 0 && s.TopK < len(c) {
		c = c[:s.TopK]
	}
	p := make([]float32, len(c))
	for i, x := range c {
		p[i] = x.Logit
	}
	nn.Scale32JIT(p, float32(1/s.Temp))
	nn.Softmax32JIT(p, len(p))
	if s.MinP > 0 {
		cut := float32(s.MinP) * p[0]
		for i, v := range p {
			if v < cut {
				c, p = c[:i], p[:i]
				break
			}
		}
	}
	if s.TopP > 0 && s.TopP < 1 {
		acc := float32(0)
		for i, v := range p {
			acc += v
			if acc >= float32(s.TopP) {
				c, p = c[:i+1], p[:i+1]
				break
			}
		}
	}
	sum := float32(0)
	for _, v := range p {
		sum += v
	}
	r := float32(s.rng.Float64()) * sum
	for i, v := range p {
		r -= v
		if r <= 0 {
			return c[i].ID
		}
	}
	return c[len(c)-1].ID
}

func allocProbeLogits(nv int) []float32 {
	l := make([]float32, nv)
	rng := rand.New(rand.NewSource(7))
	for i := range l {
		l[i] = float32(rng.NormFloat64())
	}
	return l
}

// TestSamplerAllocatesNothingWarm is the gate: after its first call a Sampler
// allocates nothing, whatever the configuration and vocabulary: per-token
// allocation feeds a collector that can already be near its goal.
func TestSamplerAllocatesNothingWarm(t *testing.T) {
	for _, nv := range []int{257, 32000, 128256} {
		logits := allocProbeLogits(nv)
		for _, cfg := range []struct {
			name            string
			topk, lastn     int
			topp, minp, pen float64
		}{
			{"topk40", 40, 0, 0, 0, 1},
			{"topk40+pen", 40, 64, 0.95, 0.02, 1.2},
			{"notopk+minp", 0, 0, 0, 0.05, 1},
			{"notopk+topp", 0, 0, 0.9, 0, 1},
			{"nothing set", 0, 0, 0, 0, 1},
		} {
			s := &Sampler{Temp: 1, TopK: cfg.topk, TopP: cfg.topp, MinP: cfg.minp,
				RepeatPen: cfg.pen, RepeatLastN: cfg.lastn, Seed: 1}
			for i := 0; i < 8; i++ {
				s.Observe(s.Sample(logits))
			}
			r := testing.Benchmark(func(b *testing.B) {
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					s.Sample(logits)
				}
			})
			if r.AllocsPerOp() != 0 {
				t.Errorf("vocab %d %s: %d allocations and %d B per warm Sample, want none",
					nv, cfg.name, r.AllocsPerOp(), r.AllocedBytesPerOp())
			}
		}
	}
}

func TestSamplerAllocProbe(t *testing.T) {
	for _, nv := range []int{32000, 128256} {
		logits := allocProbeLogits(nv)
		for _, cfg := range []struct {
			name            string
			topk, lastn     int
			topp, minp, pen float64
		}{
			{"topk40", 40, 0, 0, 0, 1},
			{"topk40+pen", 40, 64, 0.95, 0.02, 1.2},
			{"notopk+topp95", 0, 0, 0.95, 0, 1},
		} {
			o := &oldSampler{Temp: 1, TopK: cfg.topk, TopP: cfg.topp, MinP: cfg.minp,
				RepeatPen: cfg.pen, RepeatLastN: cfg.lastn, Seed: 1}
			n := &Sampler{Temp: 1, TopK: cfg.topk, TopP: cfg.topp, MinP: cfg.minp,
				RepeatPen: cfg.pen, RepeatLastN: cfg.lastn, Seed: 1}
			for i := 0; i < 8; i++ {
				o.Observe(o.Sample(logits))
				n.Observe(n.Sample(logits))
			}
			ro := testing.Benchmark(func(b *testing.B) {
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					o.Sample(logits)
				}
			})
			rn := testing.Benchmark(func(b *testing.B) {
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					n.Sample(logits)
				}
			})
			// Allocations only: the rate comes from the paired harness in
			// samplerperf_test.go.
			t.Logf("vocab %6d %-14s  Go %4d allocs %9d B/op   generated %4d allocs %9d B/op",
				nv, cfg.name,
				ro.AllocsPerOp(), ro.AllocedBytesPerOp(),
				rn.AllocsPerOp(), rn.AllocedBytesPerOp())
		}
	}
}
