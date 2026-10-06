//go:build amd64 || arm64

package nn_test

import (
	"fmt"
	"math"
	"testing"
	"time"
	"unsafe"

	"github.com/samyfodil/jitllm/format/quant"

	"github.com/samyfodil/jitllm/engine/nn"
)

// TestF32RouterCeiling measures where the F32 matvec actually sits against
// memory, at the shape the 30B's router uses, so that "is a JIT kernel worth
// writing" is answered with a number instead of an argument.
//
// No assertion: perf floors stay out of correctness tests. This prints.
func TestF32RouterCeiling(t *testing.T) {
	if testing.Short() {
		t.Skip("timing")
	}
	const nrows, k = 128, 2048 // Qwen3-30B-A3B's router
	const layers = 48          // one token's worth
	// One matrix per layer, not one reused 48 times: a single 1 MiB router
	// never leaves cache, where the real model's 48 distinct routers stream
	// from DRAM every token.
	ws := make([][]byte, layers)
	for l := range ws {
		ws[l] = make([]byte, nrows*k*4)
		fw := unsafe.Slice((*float32)(unsafe.Pointer(&ws[l][0])), nrows*k)
		for i := range fw {
			fw[i] = float32((i+l)%97) * 0.01
		}
	}
	x := make([]float32, k)
	for i := range x {
		x[i] = float32(i%31) * 0.03
	}
	out := make([]float32, nrows)
	f := nn.NewJIT(k, nrows, []quant.Type{quant.Q4_0, quant.F32})
	if f == nil {
		t.Fatal("no fast tier")
	}
	defer f.Close()
	// Warm, then time a whole token's worth of router calls.
	for i := 0; i < 20; i++ {
		f.MatVec(out, quant.F32, ws[i%layers], x, nrows, k)
	}
	best := math.Inf(1)
	for rep := 0; rep < 9; rep++ {
		t0 := time.Now()
		for l := 0; l < layers; l++ {
			f.MatVec(out, quant.F32, ws[l], x, nrows, k)
		}
		if d := time.Since(t0).Seconds(); d < best {
			best = d
		}
	}
	bytes := float64(layers) * float64(nrows) * float64(k) * 4
	fmt.Printf("\n  F32 router, %d layers of %dx%d: %.2f ms/token, %.1f GB/s\n",
		layers, nrows, k, best*1000, bytes/best/1e9)
	fmt.Printf("  (this box: ~36 GB/s Go scalar wall, 55.7 GB/s C+AVX wall -- RULE 8b)\n")
	// This can read above the memory wall: the loop repeats a set only about
	// twice the L3, so part stays resident. Use it to compare implementations
	// on the same set, never as an absolute fraction of the wall.
	fmt.Printf("  (55.7 GB/s C+AVX wall would be %.2f ms; over it means this set is partly L3-resident)\n",
		bytes/55.7e9*1000)

	// Batching the router saves nothing (1 MiB per layer fits in cache); the
	// fair comparison, both arms on the pool, is recorded in engine/model/moe.go.
}
