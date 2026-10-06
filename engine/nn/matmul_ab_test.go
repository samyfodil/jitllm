//go:build amd64 && linux

package nn

import (
	"math/rand"
	"testing"

	"github.com/samyfodil/jitllm/format/quant"

	"github.com/samyfodil/jitllm/dev/bench"
)

// TestABMatMulVsMatVec measures what batching prefill is worth: one pass over
// the weights for a whole batch against the loop of per-token matvecs.
//
// Both sides do the same arithmetic for the same tokens, so this is the honest
// prefill ratio rather than a kernel microbenchmark. Paired and interleaved,
// because the machine drifts.
func TestABMatMulVsMatVec(t *testing.T) {
	if testing.Short() {
		t.Skip("allocates")
	}
	// tinyllama is 53% Q3_K and gemma is Q4_0+Q8_0, so one format cannot answer
	// for the batched decode path the way it can for prefill.
	for _, wt := range []quant.Type{quant.Q8_0, quant.Q4_0, quant.Q3_K} {
		t.Run(wt.String(), func(t *testing.T) { abMatMul(t, wt) })
	}
}

func abMatMul(t *testing.T, wt quant.Type) {
	f := NewJIT(16384, 8192, []quant.Type{wt}, WithTune(TuneOff))
	if f == nil {
		t.Skip("no generated tier")
	}
	defer f.Close()

	const rows, k = 2048, 2048
	bb := int(wt.BlockBytes())
	for _, ntok := range []int{8, 16, 32, 128} {
		rng := rand.New(rand.NewSource(5))
		w := make([]byte, rows*(k/int(wt.BlockElems()))*bb)
		for i := range w {
			w[i] = byte(rng.Intn(256))
		}
		for off := 0; off+2 <= len(w); off += bb {
			w[off], w[off+1] = 0x00, 0x34
		}
		x := make([]float32, ntok*k)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		batched := make([]float32, ntok*rows)
		one := make([]float32, rows)
		macs := uint64(rows) * uint64(k) * uint64(ntok)

		r := bench.AB(
			bench.Case{Name: "per-token", BytesPerIter: 0, Threads: f.pool.N(), Fn: func(iters int) {
				for i := 0; i < iters; i++ {
					for n := 0; n < ntok; n++ {
						f.NewInput()
						f.MatVec(one, wt, w, x[n*k:(n+1)*k], rows, k)
					}
				}
			}},
			bench.Case{Name: "batched", BytesPerIter: 0, Threads: f.pool.N(), Fn: func(iters int) {
				for i := 0; i < iters; i++ {
					f.MatMul(batched, wt, w, x, rows, k, ntok)
				}
			}},
			15, 1)
		t.Logf("ntok=%-4d %s", ntok, r)
		t.Logf("          per-token %.1f Gmac/s   batched %.1f Gmac/s",
			float64(macs)/r.AMedian.Seconds()/1e9, float64(macs)/r.BMedian.Seconds()/1e9)
	}
}
