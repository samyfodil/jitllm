//go:build amd64 || arm64

package nn

import (
	"sync"
	"testing"

	"github.com/samyfodil/jitllm/format/quant"
)

// isoTypes are the formats both probes below need.
var isoTypes = []quant.Type{quant.Q8_0, quant.Q4_K}

// ranGEMM reports whether f runs a GEMM for a prefill-shaped batch: the
// row-major one (MatMul) or the container's weight-stationary one
// (MatMulPacked, read back through PackedGEMMCalls). WithoutGEMM turns off both.
//
// Both are probed because the row-major GEMM has no pre-VNNI form, so on such
// a host only the packed one runs.
func ranGEMM(t *testing.T, f *JIT, pk *Packed) bool {
	const rows, k, ntok = 64, 256, 32
	x := make([]float32, ntok*k)
	out := make([]float32, ntok*rows)
	if f.MatMul(out, quant.Q8_0, make([]byte, rows*k/32*34), x, rows, k, ntok) {
		return true
	}
	before := f.PackedGEMMCalls()
	if !f.MatMulPacked(out, quant.Q4_K, pk, x, rows, k, ntok) {
		t.Fatal("MatMulPacked declined a Q4_K batch outright")
	}
	return f.PackedGEMMCalls() > before
}

// Two JITs in one process must each behave per their own options. The two
// knobs below are read at call time rather than at construction, which makes a
// leak between JITs observable without a stopwatch:
//
//	noGEMM    read by every MatMul
//	pinChunk  read by ensureTuners, which PrefillChunk triggers lazily
//
// Both arms are exercised so the gate cannot pass by having run one
// configuration twice, and the arms are required to disagree.
func TestTwoJITsKeepTheirOwnOptions(t *testing.T) {
	types := isoTypes
	pk := packOne(t, quant.Q4_K, 64, 256)

	// A declines the GEMM and pins a prefill chunk that is not a candidate
	// default, so an unpinned duel cannot land on it by accident.
	a := NewJIT(4096, 4096, types, WithoutGEMM(true), WithPrefillChunk(17), WithTune(TuneOff))
	if a == nil {
		t.Skip("no generated tier")
	}
	defer a.Close()
	b := NewJIT(4096, 4096, types, WithoutGEMM(false), WithTune(TuneOff))
	if b == nil {
		t.Skip("no generated tier")
	}
	defer b.Close()

	// The decline is the selection check: the configuration is read back
	// through the path that consumes it.
	if ranGEMM(t, a, pk) {
		t.Error("A ran the GEMM although it was built WithoutGEMM(true) -- it read B's option")
	}
	if !ranGEMM(t, b, pk) {
		t.Error("B declined the GEMM although it was built WithoutGEMM(false); " +
			"the gate's other arm proves nothing without this one")
	}

	if got := a.PrefillChunk(); got != 17 {
		t.Errorf("A: PrefillChunk() = %d, want the pinned 17 -- it read B's option", got)
	}
	if got := b.PrefillChunk(); got == 17 {
		t.Errorf("B: PrefillChunk() = %d, which is A's pin; the arms must disagree", got)
	}
}

// NewJIT is also called concurrently, and each JIT must still see its own
// options: there must be no shared "last writer" at all.
func TestConcurrentNewJITKeepsEachJITsOptions(t *testing.T) {
	types := isoTypes
	pk := packOne(t, quant.Q4_K, 64, 256)
	const n = 4
	js := make([]*JIT, n)
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			js[i] = NewJIT(4096, 4096, types, WithoutGEMM(i%2 == 0), WithTune(TuneOff))
		}(i)
	}
	wg.Wait()

	for i, f := range js {
		if f == nil {
			t.Skip("no generated tier")
		}
		want := i%2 != 0 // WithoutGEMM(false) means MatMul should run
		if got := ranGEMM(t, f, pk); got != want {
			t.Errorf("JIT %d: MatMul ran = %v, want %v", i, got, want)
		}
	}
	for _, f := range js {
		f.Close()
	}
}
