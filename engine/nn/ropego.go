//go:build !jitllmtest

package nn

import "fmt"

// ropeGoTable is what a release build does if the rotary dose is turned on:
// nothing, loudly.
//
// Every tier generates the rotary table, so the float64 Go loop is a
// measurement instrument, not a fallback: it exists only under the jitllmtest
// tag.
func ropeGoTable(r Rope, cs []float32, pos, npairs int) {
	panic(fmt.Sprintf("jit: the float64 rotary table was asked for %d pair(s) at position %d, "+
		"and a release build has none -- it is the jitllmtest arm of the kernel's "+
		"measurement, not a fallback", npairs, pos))
}

// ropeGoRun is ropeGoTable for one run of a multi-axis table, and a release
// build has it for the same reason: not at all.
func ropeGoRun(r Rope, cs []float32, pos int, u RopeRun) {
	panic(fmt.Sprintf("jit: the float64 rotary table was asked for a %d-pair run at position %d, "+
		"and a release build has none", u.Pairs, pos))
}
