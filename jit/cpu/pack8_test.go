//go:build amd64 && linux

package cpu

import (
	"github.com/samyfodil/jitllm/internal/oracle"
	"math"
	"math/rand"
	"testing"

	"github.com/samyfodil/jitllm/format/quant"

	"github.com/samyfodil/jitllm/dev/bench"
)

func packVsReference(t *testing.T, wt quant.Type, rows, k int) {
	nb := k / 32
	code, err := EmitInterleaved(wt, rows, nb)
	if err != nil {
		t.Fatal(err)
	}
	kern := mustMap(t, code)
	defer kern.Close()

	const nrows = 64
	rng := rand.New(rand.NewSource(int64(rows*77 + k)))
	packed, exact := buildWeights(rng, wt, nrows, k)
	x := make([]float32, k)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	q := make([]int8, k)
	pairs := make([]float32, 2*nb)
	if err := oracle.QuantizeQ8(q, pairs, x, BiasC(wt)); err != nil {
		t.Fatal(err)
	}
	konst := KernelConst(wt)
	out := make([]float32, nrows)
	args := Args{Out: &out[0], W: &packed[0], A: &q[0], AScale: &pairs[0],
		Rows: int64(nrows / rows), K: int64(nb),
		RowStr: int64(RowBytes(wt, k)), Scr: &konst[0]}
	kern.Call(&args)

	var sse, sy2 float64
	for r := 0; r < nrows; r++ {
		var want float64
		for b := 0; b < nb; b++ {
			dx := float64(pairs[2*b])
			for i := 0; i < 32; i++ {
				want += exact[r][b*32+i] * float64(q[b*32+i]) * dx
			}
		}
		d := float64(out[r]) - want
		sse += d * d
		sy2 += want * want
	}
	if nmse := sse / sy2; nmse > 1e-10 || math.IsNaN(nmse) {
		t.Errorf("%s x%d k=%d: NMSE %.3e", wt, rows, k, nmse)
	}
	t.Logf("%s x%d k=%d: %d bytes, NMSE ok", wt, rows, k, len(code))
}

func TestPack8MatchesReference(t *testing.T) {
	for _, wt := range []quant.Type{quant.Q4_0, quant.Q8_0} {
		for _, rows := range []int{2, 4, 8} {
			for _, k := range []int{2048, 16384} {
				packVsReference(t, wt, rows, k)
			}
		}
	}
}

// TestABPack8 measures interleave width on a DRAM-resident set, paired.
func TestABPack8(t *testing.T) {
	if testing.Short() {
		t.Skip("allocates 128 MB")
	}
	const k = 2048
	nb := k / 32
	rowBytes := RowBytes(quant.Q8_0, k)
	rows := 128 << 20 / rowBytes

	rng := rand.New(rand.NewSource(17))
	w := make([]byte, rows*rowBytes)
	for i := range w {
		w[i] = byte(rng.Intn(256))
	}
	x := make([]float32, k)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	q := make([]int8, k)
	pairs := make([]float32, 2*nb)
	oracle.QuantizeQ8(q, pairs, x, BiasC(quant.Q8_0))
	konst := KernelConst(quant.Q8_0)
	out := make([]float32, rows)

	mk := func(width int) (*Code, int) {
		var b []byte
		var err error
		if width == 1 {
			b, err = Emit(Spec{W: quant.Q8_0, Rows: 1, Accs: 1, Cols: 1})
		} else {
			b, err = EmitInterleaved(quant.Q8_0, width, nb)
		}
		if err != nil {
			t.Fatal(err)
		}
		c := mustMap(t, b)
		return c, len(b)
	}
	one, n1 := mk(1)
	defer one.Close()
	eight, n8 := mk(8)
	defer eight.Close()
	t.Logf("1-row %d bytes, 8-row %d bytes", n1, n8)

	chunk := RowsPerCall(nb) / 8 * 8
	run := func(c *Code, width int) func(int) {
		return func(iters int) {
			for i := 0; i < iters; i++ {
				for r := 0; r < rows; r += chunk {
					n := min(chunk, rows-r)
					n = n / width * width
					if n == 0 {
						continue
					}
					args := Args{Out: &out[r], W: &w[r*rowBytes], A: &q[0], AScale: &pairs[0],
						Rows: int64(n / width), K: int64(nb), RowStr: int64(rowBytes), Scr: &konst[0]}
					c.Call(&args)
				}
			}
		}
	}
	bytes := uint64(rows) * uint64(rowBytes)
	r := bench.AB(
		bench.Case{Name: "1-row", BytesPerIter: bytes, Threads: 1, Fn: run(one, 1)},
		bench.Case{Name: "8-row (pack8)", BytesPerIter: bytes, Threads: 1, Fn: run(eight, 8)},
		20, 1)
	t.Logf("1 core, %d MB DRAM-resident: %v", bytes>>20, r)
}
