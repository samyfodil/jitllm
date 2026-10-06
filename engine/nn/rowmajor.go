package nn

import (
	"sync/atomic"

	"github.com/samyfodil/jitllm/format/quant"
)

// rowMajorQuant counts quantized weights read through the GGUF-block
// (row-major) layout. The row-major kernels compute the right answer from
// those bytes, so no oracle notices a container weight left in the source
// format; only a counter does.
//
// F32, F16 and BF16 are not counted: they have no blocks, and the container
// carries some (the MoE router) verbatim on purpose.
var rowMajorQuant atomic.Int64

// countRowMajor records a call into the row-major family.
func countRowMajor(t quant.Type) {
	switch t {
	case quant.F32, quant.F16, quant.BF16:
		return
	}
	rowMajorQuant.Add(1)
}

// RowMajorQuantCalls is how many times a quantized weight has been read through
// the GGUF-block layout since the process started, or since ResetRowMajorQuant.
// It must be zero across any inference on a container.
func RowMajorQuantCalls() int64 { return rowMajorQuant.Load() }

// ResetRowMajorQuant zeroes the counter so a gate can attribute calls to one
// phase. Package-wide and therefore not for concurrent use by two gates at
// once, which is the same contract MatMulProfile already has.
func ResetRowMajorQuant() { rowMajorQuant.Store(0) }
