//go:build amd64 || arm64

package nn_test

import (
	"encoding/binary"
	"github.com/jitllm/jitllm/internal/oracle"
	"math"
	"math/rand"
	"sort"
	"testing"

	"github.com/jitllm/jitllm/format/quant"

	"github.com/jitllm/jitllm/engine/nn"
)

// topK is the MoE router's rule: the k largest, largest first, ties to the
// lower index (ggml_argsort descending). A local float64 reference for this
// gate; the engine runs nn.MoETopK32JIT.
func topK(p []float64, k int) []int {
	idx := make([]int, len(p))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool { return p[idx[a]] > p[idx[b]] })
	return idx[:k]
}

// TestF32RouterSelectionAgrees is the gate on computing the mixture-of-experts
// router in float32 rather than float64.
//
// Bit exactness is not the contract: the softmax between the matvec and the
// selection is itself not bit-monotone, so exactness upstream buys nothing.
// What must hold is that f32 only changes the selection where the k-th and
// (k+1)-th probabilities are within the arithmetic's own error, never where
// the router had made up its mind.
func TestF32RouterSelectionAgrees(t *testing.T) {
	for _, c := range []struct {
		name     string
		nrows, k int
		used     int
		spread   float64 // logit scale: small = near-ties everywhere
	}{
		{"qwen3-30b", 128, 2048, 8, 1.0},
		{"qwen3-30b/flat", 128, 2048, 8, 0.02}, // adversarial: everything a near-tie
		{"4x0.6b", 4, 1024, 2, 1.0},
		{"mixtral", 8, 4096, 2, 1.0},
	} {
		t.Run(c.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(c.nrows*7919 + c.k)))
			f := nn.NewJIT(c.k, c.nrows, []quant.Type{quant.Q4_0, quant.F32})
			if f == nil {
				t.Fatal("NewJIT returned nil; this test cannot run and must not silently pass")
			}
			defer f.Close()

			const trials = 200
			setDiff, orderDiff := 0, 0
			worstDecided := 0.0 // largest boundary margin at which the set changed

			w := make([]byte, c.nrows*c.k*4)
			x := make([]float32, c.k)
			for tr := 0; tr < trials; tr++ {
				for i := 0; i < c.nrows*c.k; i++ {
					binary.LittleEndian.PutUint32(w[i*4:],
						math.Float32bits(float32(rng.NormFloat64()*c.spread/math.Sqrt(float64(c.k)))))
				}
				for i := range x {
					x[i] = float32(rng.NormFloat64())
				}

				ref32 := make([]float32, c.nrows)
				if err := oracle.MatVec(ref32, quant.F32, w, x, c.nrows, c.k); err != nil {
					t.Fatal(err)
				}
				got := make([]float32, c.nrows)
				if !f.MatVec(got, quant.F32, w, x, c.nrows, c.k) {
					t.Fatal("the F32 fast path declined")
				}
				oracle.Softmax32(ref32)
				oracle.Softmax32(got)

				// The oracle is f32 with a compensated accumulator; this
				// comparison widens both sides only to score them.
				ref := make([]float64, len(ref32))
				oracle.Widen(ref, ref32)
				g64 := make([]float64, len(got))
				oracle.Widen(g64, got)
				a, b := topK(ref, c.used), topK(g64, c.used)
				sameSet := map[int]bool{}
				for _, e := range a {
					sameSet[e] = true
				}
				n := 0
				for _, e := range b {
					if sameSet[e] {
						n++
					}
				}
				if n != c.used {
					setDiff++
					// A flip at a wide margin is a bug; at a narrow one it is
					// rounding breaking a near-tie.
					sorted := append([]float64(nil), ref...)
					sort.Sort(sort.Reverse(sort.Float64Slice(sorted)))
					if m := sorted[c.used-1] - sorted[c.used]; m > worstDecided {
						worstDecided = m
					}
					continue
				}
				for i := range a {
					if a[i] != b[i] {
						orderDiff++
						break
					}
				}
			}

			t.Logf("%d trials: %d set changes, %d order-only changes, widest margin at a set change %.3g",
				trials, setDiff, orderDiff, worstDecided)

			// The bar is the margin, not the count: float32 carries ~1e-7
			// relative and these probabilities are O(1/nExpert), so a genuine
			// tie-break sits well below 1e-4.
			if worstDecided > 1e-4 {
				t.Errorf("float32 changed the selection at a margin of %.3g -- the router had "+
					"decided and the arithmetic overturned it; that is not a tie-break", worstDecided)
			}
		})
	}
}
