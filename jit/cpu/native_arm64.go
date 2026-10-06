//go:build arm64

package cpu

import "github.com/samyfodil/jitllm/format/quant"

// The native emitter for this GOARCH; see native_amd64.go.
//
// SupportedNative is "can this host run it": what EmitA64 can produce and the
// chip having the instructions. Every quantized arm64 kernel emits SDOT
// (FEAT_DotProd, absent on Cortex-A53/A57/A72/A73); the float kernels are
// baseline NEON and always run.
func SupportedNative(t quant.Type) bool {
	if !SupportedA64(t) {
		return false
	}
	if t == quant.F32 || t == quant.F16 || t == quant.BF16 {
		return true // baseline ARMv8-A: LDRq, FMLA4s, FADDP4s, UBFM, FCVTL, SHLL
	}
	return hasDotProd()
}
func EmitNative(s Spec) ([]byte, error) { return EmitA64(s) }

// rowMajorPF is EmitNative: the amd64 prefetch has no arm64 form.
func rowMajorPF(s Spec, _ int) ([]byte, error) { return EmitA64(s) }

// primaryTier is the tier the primary emitter table generates for on arm64:
// the A64 emitters, which answer to the same names as the AVX2 tier's.
const primaryTier = TierNEON

// primaryRowMajorSupported is SupportedNative itself here: arm64 has one tier.
func primaryRowMajorSupported(t quant.Type) bool { return SupportedNative(t) }

// BestPackNative is 1 on arm64: there is no interleaved kernel, so AddShape
// declines before it can ask for one, and the single-row kernel runs across
// the pool.
func BestPackNative(t quant.Type) int { return 1 }

// PackWidthNative is BestPackNative with a pin, which arm64 ignores for the
// same reason: there is no interleaved kernel to widen.
func PackWidthNative(pin int, t quant.Type) int { return 1 }

func NativeArch() string { return "arm64" }
