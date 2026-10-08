//go:build arm64

package cpu

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/samyfodil/jitllm/format/quant"
	"github.com/samyfodil/jitllm/internal/oracle"
	"github.com/samyfodil/jitllm/jit/gpu/kernels"
)

// The kernels without FEAT_DotProd (sdotemu.go). Two gates beside the package's
// own: every SDOT-emitting family emits none once the probe says the chip lacks
// it, and the widened decode kernels give the SDOT kernels' bits, against the
// oracle, at ragged k and row counts with a guard after the output. The whole
// suite runs widened too, with JITLLM_FORCE_NO_DOTPROD=1 or under
// `qemu-aarch64-static -cpu cortex-a72` (sdotemu_env_test.go).

// withDot sets, for the rest of t, whether the emitters widen SDOT: absent
// forces the probe off, and present forces SDOT's encoding, which only a chip
// with the feature may then run (chipHasDot).
func withDot(t *testing.T, absent bool) {
	t.Helper()
	oldN, oldE := forceNoDotProd, forceDotEncoding
	t.Cleanup(func() { forceNoDotProd, forceDotEncoding = oldN, oldE })
	forceNoDotProd, forceDotEncoding = absent, !absent
	if absent != a64EmulateDot() {
		t.Fatalf("forced FEAT_DotProd absent=%v and the emitters disagree", absent)
	}
}

// chipHasDot is the probe with no override: whether SDOT may be executed.
func chipHasDot() bool {
	old := forceNoDotProd
	defer func() { forceNoDotProd = old }()
	forceNoDotProd = false
	return hasDotProd()
}

// sdotWords counts SDOT and SDOTelem encodings (any register, any Q) in code.
func sdotWords(code []byte) int {
	n := 0
	for i := 0; i+4 <= len(code); i += 4 {
		w := binary.LittleEndian.Uint32(code[i:])
		if w&0xBFE0FC00 == 0x0E809400 || w&0xBFC0F400 == 0x0F80E000 {
			n++
		}
	}
	return n
}

// dotFamilies emits every kernel family that uses SDOT, at shapes the engine
// asks for, keyed by a name.
func dotFamilies(t *testing.T) map[string][]byte {
	t.Helper()
	out := map[string][]byte{}
	add := func(name string, code []byte, err error) {
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		out[name] = code
	}
	for _, w := range a64KernelTypes {
		for _, win := range []int16{0, 256} {
			code, err := EmitA64(Spec{W: w, Rows: 1, Accs: 1, Cols: 1, ActWin: win})
			add("rowmajor/"+w.String()+"/win"+itoa(int(win)), code, err)
		}
		mr, nr := 2, 1
		code, err := EmitGEMMWindow(w, mr, nr, 8, 32)
		if w == quant.Q5_0 {
			continue // no row-major GEMM for Q5_0
		}
		add("gemm/"+w.String(), code, err)
	}
	for _, w := range quant.PackedTypes {
		code, err := EmitA64PackedMatVecPF(w, PackedRows, 0)
		add("packed/"+w.String(), code, err)
		code, err = EmitA64PackedMatVecFused(w, 16, 0)
		add("fused/"+w.String(), code, err)
		code, err = EmitA64PackedMatVecFusedWin(w, 16, 0, 256)
		add("fusedwin/"+w.String(), code, err)
		q, _ := kernels.QuantOf(w)
		k := PackedOuterElems(w) * 2
		group := packedFusedGroupK(q)
		for tok := 1; tok <= MaxTiledTokensA64(w); tok++ {
			code, err = EmitA64PackedMatMulTiled(w, k, group*2, tok)
			add("tiled/"+w.String()+"/tok"+itoa(tok), code, err)
		}
		for _, win := range []int{32, 256} {
			code, err = EmitA64PackedMatMulStationary(w, k, 16, win)
			if err != nil {
				continue // a format with no stationary form is declined by name
			}
			add("stationary/"+w.String()+"/win"+itoa(win), code, err)
		}
	}
	return out
}

// TestNoDotProdKernelsEmitNoSDOT: with the feature present every family uses
// SDOT (so the scan can see it), and with it absent none does and the widening
// counter moved. A family that forgot DotScratch fails here.
func TestNoDotProdKernelsEmitNoSDOT(t *testing.T) {
	withDot(t, false)
	with := dotFamilies(t)
	for name, code := range with {
		if sdotWords(code) == 0 {
			t.Errorf("%s: no SDOT with the feature present -- the scan sees nothing", name)
		}
	}
	withDot(t, true)
	before := DotEmulated()
	without := dotFamilies(t)
	if len(without) != len(with) {
		t.Fatalf("%d families without the feature, %d with", len(without), len(with))
	}
	for name, code := range without {
		if n := sdotWords(code); n != 0 {
			t.Errorf("%s: %d SDOTs on a chip without FEAT_DotProd", name, n)
		}
	}
	if DotEmulated() == before {
		t.Fatal("no SDOT was widened -- the forced path emitted nothing it claims to")
	}
	t.Logf("%d kernel families, %d SDOTs widened", len(without), DotEmulated()-before)
}

// TestNoDotProdMatchesDotBitForBit runs the container decode kernels (the tiled
// matvec and the fused one) for every packed format with and without
// FEAT_DotProd: the widened sequence computes SDOT's int32 lanes exactly, so
// the outputs are the same bits, and both are held to the oracle. k sweeps one
// to three outer steps and the rows one to three tiles, with a guard after the
// output that neither kernel may write.
func TestNoDotProdMatchesDotBitForBit(t *testing.T) {
	// On a chip without the feature the SDOT arm cannot run; the widened arm is
	// still held to the oracle and the guard.
	both := chipHasDot()
	if !both {
		t.Log("no FEAT_DotProd on this chip: the widened arm alone, against the oracle")
	}
	ran := 0
	for _, f := range packedFormats {
		for _, fused := range []bool{false, true} {
			step := PackedOuterElems(f.g)
			rows := PackedRows
			if fused {
				rows = 16
			}
			for ks := 1; ks <= 3; ks++ {
				for tiles := 1; tiles <= 3; tiles++ {
					k, nrows := ks*step, tiles*rows
					name := f.name + "/k" + itoa(k) + "/rows" + itoa(nrows)
					if fused {
						name = "fused/" + name
					}
					t.Run(name, func(t *testing.T) {
						emit := func(absent bool) []byte {
							withDot(t, absent)
							var code []byte
							var err error
							if fused {
								code, err = EmitA64PackedMatVecFused(f.g, rows, 0)
							} else {
								code, err = EmitA64PackedMatVecPF(f.g, rows, 0)
							}
							if err != nil {
								t.Fatal(err)
							}
							return code
						}
						dot, wide := emit(false), emit(true)
						if sdotWords(wide) != 0 || sdotWords(dot) == 0 {
							t.Fatalf("the arms are not what they claim: %d SDOTs widened, %d plain",
								sdotWords(wide), sdotWords(dot))
						}
						gotW := runPackedGuarded(t, wide, f.g, f.k, rows, nrows, k)
						got := gotW
						if both {
							got = runPackedGuarded(t, dot, f.g, f.k, rows, nrows, k)
						}
						for r := range got.out {
							if math.Float32bits(got.out[r]) != math.Float32bits(gotW.out[r]) {
								t.Fatalf("row %d: SDOT %v, widened %v -- not the same bits", r, got.out[r], gotW.out[r])
							}
						}
						var num, den float64
						for r := range gotW.out {
							dv := float64(gotW.out[r]) - float64(gotW.ref[r])
							num += dv * dv
							den += float64(gotW.ref[r]) * float64(gotW.ref[r])
						}
						if den == 0 || math.IsNaN(num) || math.IsNaN(den) {
							t.Fatalf("degenerate oracle (num=%v den=%v) -- this gate proved nothing", num, den)
						}
						// The bound is the int8 activation's error, which a k of one
						// block (32) does not average down as k=512 does: Q5_1 at
						// k=32 reads 1.06e-3 on both arms. The bit comparison
						// above is the kernel gate; this one holds the pair to
						// the arithmetic.
						if nmse := num / den; nmse > 4e-3 {
							t.Fatalf("widened against the oracle: NMSE %.3e", nmse)
						}
						ran++
					})
				}
			}
		}
	}
	if ran == 0 {
		t.Fatal("no case ran")
	}
}

type packedRun struct{ out, ref []float32 }

// runPackedGuarded runs a container decode kernel over nrows rows of k with a
// NaN-poisoned output and a guard after it, and returns the output and the
// oracle's. The output accumulates (out[r] +=), so it starts at zero.
func runPackedGuarded(t *testing.T, code []byte, g quant.Type, kq kernels.Quant, rows, nrows, k int) packedRun {
	t.Helper()
	kern := mustMap(t, code)
	defer kern.Close()
	rng := rand.New(rand.NewSource(int64(nrows*131 + k)))
	nb := uint64(nrows*k) / g.BlockElems()
	src := make([]byte, nb*g.BlockBytes())
	for i := range src {
		src[i] = byte(rng.Intn(256))
	}
	if !quant.PlantScales(g, src, 0) {
		t.Fatalf("%s: no scale planter", g)
	}
	qs, d, sc, err := kernels.PackWeights(kq, src, nrows, k)
	if err != nil {
		t.Fatal(err)
	}
	x := make([]float32, k)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	q8 := make([]int8, k)
	pairs := make([]float32, 2*(k/Q8Block))
	if err := oracle.QuantizeQ8(q8, pairs, x, BiasC(g)); err != nil {
		t.Fatal(err)
	}
	half := make([]float32, k/16)
	oracle.QuantizeHalfSums(half, q8, BiasC(g))
	scr := make([]byte, PackedScratchBytes)
	if err := PackedScratch(scr); err != nil {
		t.Fatal(err)
	}
	const guard = 16
	buf := make([]float32, nrows+guard)
	for i := nrows; i < len(buf); i++ {
		buf[i] = float32(math.NaN())
	}
	qb, db, scb := u32b(qs), u32b(d), u32b(sc)
	args := Args{
		Out: &buf[0], W: &qb[0], A: &q8[0], AScale: &pairs[0],
		Rows: int64(nrows / rows), RowStr: int64(nrows * 4),
		K: int64(k / PackedOuterElems(g)), Scr: &scr[0], PD: &db[0], AHalf: &half[0],
		DStr: int64(DSuperBytes(g, nrows)),
	}
	if len(scb) > 0 {
		args.PSC = &scb[0]
	}
	kern.Call(&args)
	for i := nrows; i < len(buf); i++ {
		if !math.IsNaN(float64(buf[i])) {
			t.Fatalf("guard %d after the output was written: %v", i-nrows, buf[i])
		}
	}
	ref := make([]float32, nrows)
	if err := oracle.MatVecPacked(ref, kq, qb, db, scb, x, nrows, k); err != nil {
		t.Fatal(err)
	}
	return packedRun{out: buf[:nrows], ref: ref}
}
