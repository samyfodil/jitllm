//go:build arm64

package cpu

import (
	"sync"
)

// FEAT_DotProd detection, so a chip without SDOT is told which tensor no
// kernel reads instead of dying on an illegal instruction. SDOT is ARMv8.2-A:
// Cortex-A53/A57/A72/A73 (a Raspberry Pi 3 or 4) lack it. Every quantized
// arm64 kernel uses SDOT; the float kernels are baseline NEON.
var (
	dotOnce sync.Once
	hasDot  bool
)

// hasDotProd reports FEAT_DotProd. forceNoDotProd forces it false, which is how
// the fallback is tested on a machine that has the feature (assigned only from
// a _test.go file here).
func hasDotProd() bool {
	dotOnce.Do(func() {
		if forceNoDotProd {
			hasDot = false
			return
		}
		hasDot = probeDotProd()
	})
	return hasDot
}
