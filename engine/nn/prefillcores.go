//go:build amd64 || arm64

package nn

import "runtime"

// prefillCoreSet resolves Config.PrefillCores: the caller's name when there is
// one, else the host's measured default.
//
// On Apple Silicon prefill wants the E-cores and decode does not, so they are
// two pools: prefill is compute the E-cores add to, while decode is the bus,
// which the P-cores nearly fill. Mixtures stay on the decode pool
// (model.prefillSrc): their expert GEMMs are too small to amortise a wider
// barrier. On Linux the split measured neutral, so "p" (no prefill pool) is the
// default there. See docs/engineering-history/cpu-kernels.md.
func prefillCoreSet(name string) string {
	if name != "" {
		return name
	}
	if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
		return "pe"
	}
	return "p"
}
