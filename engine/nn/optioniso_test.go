//go:build amd64 || arm64

package nn

import (
	"sync"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
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

// tierHasGEMM reports whether f's tier emits either GEMM ranGEMM probes. The
// SSE tier declines both by design (the GGUF row-major machinery,
// cpu.Emitters.RowMajorGGUF, and the weight-stationary GEMM,
// cpu.EmitPackedGEMMSSE), so there WithoutGEMM has nothing to turn off and a
// GEMM must not run whatever the option says.
func tierHasGEMM(f *JIT) bool {
	if f.em.RowMajorGGUF {
		return true
	}
	_, err := f.em.PackedGEMM(quant.Q4_K, 256, 64, 0)
	return err == nil
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
	switch ran := ranGEMM(t, b, pk); {
	case tierHasGEMM(b) && !ran:
		t.Error("B declined the GEMM although it was built WithoutGEMM(false); " +
			"the gate's other arm proves nothing without this one")
	case !tierHasGEMM(b) && ran:
		t.Errorf("B ran a GEMM on tier %v, which emits none", b.tier)
	case !tierHasGEMM(b):
		// Only the prefill chunk below tells the two JITs apart here.
		t.Logf("tier %v emits no GEMM: WithoutGEMM is not observable on this host, "+
			"and the prefill chunk pin carries the isolation check", b.tier)
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

	for _, f := range js {
		if f == nil {
			t.Skip("no generated tier")
		}
	}
	defer func() {
		for _, f := range js {
			f.Close()
		}
	}()
	if !tierHasGEMM(js[0]) {
		// The decline itself still holds: no JIT may run a GEMM here.
		for i, f := range js {
			if ranGEMM(t, f, pk) {
				t.Errorf("JIT %d ran a GEMM on tier %v, which emits none", i, f.tier)
			}
		}
		t.Skipf("tier %v emits no GEMM (the SSE tier declines the row-major and the "+
			"weight-stationary one by design), so WithoutGEMM -- the only option this "+
			"gate reads back -- is not observable on this host; an AVX2 or NEON host "+
			"runs the isolation check", js[0].tier)
	}
	for i, f := range js {
		want := i%2 != 0 // WithoutGEMM(false) means MatMul should run
		if got := ranGEMM(t, f, pk); got != want {
			t.Errorf("JIT %d: MatMul ran = %v, want %v", i, got, want)
		}
	}
}
