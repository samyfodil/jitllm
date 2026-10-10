package nn

import (
	"testing"

	"github.com/jitllm/jitllm/format/quant"
)

// The counter's own gate: model-level gates assert the row-major counter reads
// zero, which is worth only as much as the counter moving when it should. It
// counts a quantized read and does not count an F32 one (the container carries
// F32 routers verbatim on purpose).
func TestRowMajorQuantCounterDiscriminates(t *testing.T) {
	// One Q8_0 block: an f16 scale of 1.0 followed by 32 signed bytes.
	w := make([]byte, 34)
	w[1] = 0x3C // f16 1.0
	for i := 0; i < 32; i++ {
		w[2+i] = 1
	}
	x := make([]float32, 32)
	for i := range x {
		x[i] = 1
	}
	out := make([]float32, 1)
	f := NewJIT(32, 1, []quant.Type{quant.Q8_0, quant.F32})
	defer f.Close()

	// The count is at the top of JIT.MatVec, before any kernel is chosen, so it
	// moves whether or not this host runs the Q8_0 row-major kernel.
	ResetRowMajorQuant()
	f.MatVec(out, quant.Q8_0, w, x, 1, 32)
	if got := RowMajorQuantCalls(); got != 1 {
		t.Fatalf("a Q8_0 read through JIT.MatVec counted %d, want 1: the instrument behind "+
			"the no-gguf gate does not move", got)
	}

	// The F32 control: same call shape, a type with no blocks.
	f32 := make([]byte, 32*4)
	ResetRowMajorQuant()
	if !f.MatVec(out, quant.F32, f32, x, 1, 32) {
		t.Fatal("the F32 matvec declined")
	}
	if got := RowMajorQuantCalls(); got != 0 {
		t.Fatalf("an F32 read counted %d, want 0: the counter would flag every container's "+
			"verbatim router and stop meaning anything", got)
	}

	// And the reset itself, or a gate that resets before a phase measures the
	// whole process instead.
	f.MatVec(out, quant.Q8_0, w, x, 1, 32)
	ResetRowMajorQuant()
	if got := RowMajorQuantCalls(); got != 0 {
		t.Fatalf("ResetRowMajorQuant left %d", got)
	}
}
