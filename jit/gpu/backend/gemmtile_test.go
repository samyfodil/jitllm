package backend_test

import (
	"fmt"
	"math/rand"
	"os"
	"testing"
	"time"

	"github.com/samyfodil/jitllm/jit/gpu/backend"
	"github.com/samyfodil/jitllm/jit/gpu/kernels"
)

// gemmTileCases are the shapes TestGemmTileMatchesTheReference sweeps: every
// staging nesting (rows a multiple of the threads, threads a multiple of the
// rows), one to four sub-blocks a trip, tiles that are tall, wide and square,
// and a bias on each family.
var gemmTileCases = []struct {
	rows, k, ntok int
	tl            kernels.TileGemm
	bias          bool
}{
	{64, 512, 32, kernels.TileGemm{MT: 4, NT: 2, WM: 2, WN: 2, KB: 2}, false},
	{128, 512, 64, kernels.TileGemm{MT: 4, NT: 2, WM: 2, WN: 2, KB: 2}, true},
	{128, 1024, 64, kernels.TileGemm{MT: 4, NT: 4, WM: 2, WN: 2, KB: 1}, false},
	{256, 768, 32, kernels.TileGemm{MT: 4, NT: 4, WM: 4, WN: 1, KB: 1}, true},
	{64, 1024, 128, kernels.TileGemm{MT: 2, NT: 4, WM: 2, WN: 2, KB: 2}, true},
}

// TestGemmTileMatchesTheReference holds Metal's staged simdgroup_matrix GEMM
// to the float64 reference and to the dp4a kernel it replaces, on every format
// that has a binary16 dequant. Only MSL lowers the collective tile ops from
// workgroup memory; anywhere else a refusal is the expected answer, and on MSL
// a refusal is a broken kernel.
func TestGemmTileMatchesTheReference(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	ran := 0
	for _, d := range devs {
		defer d.Close()
		for _, q := range []kernels.Quant{kernels.Q4_K, kernels.Q6_K, kernels.Q8_0, kernels.Q5_K,
			kernels.Q4_0, kernels.Q5_0, kernels.Q3_K, kernels.Q5_1} {
			for _, sh := range gemmTileCases {
				sub, _, _, _ := kernels.Layout(q)
				tl := sh.tl
				tl.KB = max(tl.KB*32/sub, 1)
				s := kernels.MatVecShape{T: q, K: sh.k, Rows: sh.rows, NTok: sh.ntok, Bias: sh.bias}
				kk, err := kernels.GemmTile(s, tl)
				if err != nil {
					t.Fatalf("%v %+v: %v", q, sh, err)
				}
				kern, err := d.Compile(kk)
				if err != nil {
					if d.API() == "msl" {
						t.Fatalf("%s %v %+v: %v", d.API(), q, sh, err)
					}
					t.Logf("%s: %v", d.API(), err)
					break
				}
				ran++
				mma70Case(t, d, kern, s, fmt.Sprintf("gemmtile %+v", tl),
					kernels.GemmTileGroups(s, tl), tl.Threads(), true)
				kern.Close()
			}
		}
	}
	if ran == 0 {
		t.Skip("no backend lowers the collective tile GEMM here")
	}
}

// TestGemmTileGroupedMatchesPerExpert is the grouped (mixture) GemmTile held
// to the dense GemmTile on each expert's own sheet bit for bit, and to the
// float64 reference, with a block past `used` required to stay poisoned --
// TestGemmVoltaGroupedMatchesPerExpert's contract, on Metal's tiles.
func TestGemmTileGroupedMatchesPerExpert(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	ran := 0
	for _, d := range devs {
		defer d.Close()
		for _, q := range []kernels.Quant{kernels.Q4_K, kernels.MXFP4, kernels.Q6_K, kernels.Q8_0, kernels.Q5_0,
			kernels.Q5_K, kernels.Q4_0, kernels.Q5_1} {
			sub, _, _, _ := kernels.Layout(q)
			kb := max(32/sub, 1)
			for _, c := range []struct {
				rows, k, experts int
				tl               kernels.TileGemm
				sel              []uint32
				used             int
				bias             bool
			}{
				{128, 512, 4, kernels.TileGemm{MT: 4, NT: 2, WM: 2, WN: 2, KB: kb}, []uint32{2, 0, 0, 3, 1}, 5, false},
				{192, 256, 4, kernels.TileGemm{MT: 4, NT: 2, WM: 2, WN: 2, KB: kb}, []uint32{3, 3, 1, 0}, 3, true},
				{256, 512, 3, kernels.TileGemm{MT: 4, NT: 4, WM: 2, WN: 2, KB: kb}, []uint32{1, 2, 0}, 3, false},
				{96, 512, 4, kernels.TileGemm{MT: 2, NT: 2, WM: 2, WN: 2, KB: kb}, []uint32{0, 3, 3, 1}, 4, true},
			} {
				s := kernels.MatVecShape{T: q, K: c.k, Rows: c.rows, NTok: c.tl.Toks() * len(c.sel), Experts: c.experts,
					Bias: c.bias}
				kk, err := kernels.GemmTile(s, c.tl)
				if err != nil {
					t.Fatalf("%v %+v: %v", q, c.tl, err)
				}
				kern, err := d.Compile(kk)
				if err != nil {
					if d.API() == "msl" {
						t.Fatalf("%s %v %+v: %v", d.API(), q, c.tl, err)
					}
					t.Logf("%s: %v", d.API(), err)
					break
				}
				ran++
				groupedGemmCase(t, d, kern, s, tileGrouped(c.tl), c.sel, c.used)
				kern.Close()
			}
		}
	}
	if ran == 0 {
		t.Skip("no backend lowers the collective tile GEMM here")
	}
}

// TestGemmTileSpeed times GemmTile's blockings against the dp4a batched twin on
// Llama-3.2-1B's projection shapes (JITLLM_GEMMTILE_BENCH=1). A probe, not a
// gate.
func TestGemmTileSpeed(t *testing.T) {
	if os.Getenv("JITLLM_GEMMTILE_BENCH") == "" {
		t.Skip("set JITLLM_GEMMTILE_BENCH=1")
	}
	gpuLock(t)
	for _, d := range backend.Open() {
		defer d.Close()
		if d.API() != "msl" {
			continue
		}
		for _, sh := range [][3]int{{8192, 2048, 512}, {2048, 8192, 512}, {2048, 2048, 512}, {512, 2048, 512}} {
			rows, k, ntok := sh[0], sh[1], sh[2]
			rng := rand.New(rand.NewSource(1))
			g := newGPU(t, d)
			qs, dw, scw, _ := kernels.PackWeights(kernels.Q4_K, rawWeights(kernels.Q4_K, rows, k, rng), rows, k)
			bQS, bD, bSC := g.up(u32bytes(qs)), g.up(u32bytes(dw)), g.up(u32bytes(scw))
			out := g.up(make([]byte, rows*ntok*4))
			bB := g.up(make([]byte, ntok*k*2))
			bA := g.up(make([]byte, ntok*k))
			bAX := g.up(make([]byte, ntok*k/32*8))
			sync := func() { out.Read(make([]byte, 4)) }
			time1 := func(name string, f func() error) {
				if err := f(); err != nil {
					t.Logf("%s: %v", name, err)
					return
				}
				sync()
				const n = 10
				t0 := time.Now()
				for i := 0; i < n; i++ {
					f()
				}
				sync()
				el := time.Since(t0) / n
				t.Logf("%5dx%-5d t%d %-30s %8.1f us  %.2f TFLOP/s", rows, k, ntok, name, float64(el.Microseconds()),
					2*float64(rows)*float64(k)*float64(ntok)/el.Seconds()/1e12)
			}
			if kk, err := kernels.MatVec(kernels.MatVecShape{T: kernels.Q4_K, K: k, Rows: rows, NTok: ntok, Tok: 8, Rowt: 4}); err == nil {
				if kern, err := d.Compile(kk); err == nil {
					th := rows / 4 * ntok / 8
					time1("dp4a tok8 rowt4", func() error { return kern.Launch((th+127)/128, 128, bQS, bD, bSC, bA, bAX, out) })
				}
			}
			for _, tl := range []kernels.TileGemm{
				{MT: 4, NT: 2, WM: 2, WN: 2, KB: 2}, {MT: 4, NT: 4, WM: 2, WN: 2, KB: 1},
				{MT: 2, NT: 4, WM: 2, WN: 2, KB: 2}, {MT: 4, NT: 2, WM: 2, WN: 2, KB: 1},
				{MT: 4, NT: 4, WM: 4, WN: 1, KB: 1}, {MT: 2, NT: 2, WM: 2, WN: 2, KB: 2},
			} {
				s := kernels.MatVecShape{T: kernels.Q4_K, K: k, Rows: rows, NTok: ntok}
				kk, err := kernels.GemmTile(s, tl)
				if err != nil {
					t.Logf("%+v: %v", tl, err)
					continue
				}
				kern, err := d.Compile(kk)
				if err != nil {
					t.Fatalf("%+v: %v", tl, err)
				}
				time1(fmt.Sprintf("tile m%dn%d w%dx%d k%d", tl.MT, tl.NT, tl.WM, tl.WN, tl.KB), func() error {
					return kern.Launch(kernels.GemmTileGroups(s, tl), tl.Threads(), bQS, bD, bSC, bB, out)
				})
				kern.Close()
			}
			g.free()
		}
	}
}
