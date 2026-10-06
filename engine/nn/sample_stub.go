//go:build !(amd64 || arm64)

package nn

// The sampler's kernels have no emitter on this platform, and there is no Go
// path, so every call panics naming the missing op.
//
// This file must track engine/nn/sample.go declaration for declaration (see
// jit_stub.go).

// SampleOrder walks a logits row in strict descending order. See engine/nn/sample.go.
type SampleOrder struct{}

func (o *SampleOrder) Begin(vals []float32) { panic(noSampler) }
func (o *SampleOrder) Reserve(n int)        {}

func (o *SampleOrder) Next() (float32, int32, bool) { panic(noSampler) }

// SampleDrawResult is what one call of the draw kernel reports.
type SampleDrawResult struct {
	Res      int
	M        int
	Cut      bool
	Resolved bool
	Sum      float32
}

// SampleDraw is the parameter and counter block one caller reuses.
type SampleDraw struct{}

func (d *SampleDraw) Run(p []float32, minp, topp, u, sumAll float32) SampleDrawResult {
	panic(noSampler)
}

func (d *SampleDraw) Mass(p []float32) float32 { panic(noSampler) }

// SamplePenalty32JIT applies llama.cpp's repeat penalty in place.
func SamplePenalty32JIT(vals []float32, off []int64, pen float32) { panic(noSampler) }

const noSampler = "jit: this platform has no sampler kernel -- sampling is generated " +
	"code and has no Go path"
