//go:build jitllmtest

package nn

import (
	"math"
	"sync/atomic"
)

// ropeGoTable is the float64 arithmetic the generated rotary table replaced
// (the pre-kernel Rope.Table body), kept under the test tag as the other arm of
// the kernel's measurement: nn.WithRopeGo routes every table here so
// model.TestRopeTableAgreesWithF64 can alternate the two arms in one process.
// A release build gets ropego.go, which panics.
func ropeGoTable(r Rope, cs []float32, pos, npairs int) {
	ropeGoCalls.Add(1)
	mscale := r.Scale
	if mscale == 0 {
		mscale = 1
	}
	theta := float64(pos)
	step := math.Pow(r.Base, -2/float64(r.NRot))
	for p := 0; p < npairs; p++ {
		th := theta
		if p < len(r.Freqs) {
			th /= float64(r.Freqs[p])
		}
		cs[2*p] = float32(math.Cos(th) * mscale)
		cs[2*p+1] = float32(math.Sin(th) * mscale)
		theta *= step
	}
}

var ropeGoCalls atomic.Int64

// RopeGoCalls is how many tables the float64 arm has built, the twin of
// RopeTableCalls, so a dose gate can check the arm it selected actually ran.
func RopeGoCalls() int64 { return ropeGoCalls.Load() }

// ropeGoRun is ropeGoTable for one run of a multi-axis table: pair j at the
// run's frequency index Freq + j*Step of the same recurrence.
func ropeGoRun(r Rope, cs []float32, pos int, u RopeRun) {
	ropeGoCalls.Add(1)
	mscale := r.Scale
	if mscale == 0 {
		mscale = 1
	}
	for j, f := range runFreqs(r.runRope(u), u.Pairs) {
		th := float64(pos) * f
		cs[2*j] = float32(math.Cos(th) * mscale)
		cs[2*j+1] = float32(math.Sin(th) * mscale)
	}
}
