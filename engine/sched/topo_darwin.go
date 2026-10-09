//go:build darwin

package sched

import (
	"runtime"
	"syscall"
)

// Apple Silicon reports its clusters through perflevel sysctls: level 0 is the
// performance cluster, level 1 the efficiency one.
//
// The indices are placeholders, not CPU ids: macOS exposes no processor
// affinity, and QoS schedules busy default-class threads on performance cores.
// Dynamic chunk claiming lets a thread that lands on an E-core simply claim
// fewer chunks.
func PCores() []int { return seq(perfLevel(0, runtime.NumCPU())) }

func ECores() []int {
	n := perfLevel(1, 0)
	p := perfLevel(0, runtime.NumCPU())
	out := make([]int, n)
	for i := range out {
		out[i] = p + i
	}
	return out
}

// SMTSiblings has nothing to add: no Apple Silicon core is SMT.
func SMTSiblings() []int { return PCores() }

func perfLevel(level, dflt int) int {
	name := "hw.perflevel0.physicalcpu"
	if level == 1 {
		name = "hw.perflevel1.physicalcpu"
	}
	if v, err := syscall.SysctlUint32(name); err == nil && v > 0 {
		return int(v)
	}
	return dflt
}

// CoreSource says where the pool's CPUs came from, for a report.
func CoreSource() string { return "perflevel sysctls" }
