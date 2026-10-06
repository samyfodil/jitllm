package model

import (
	"fmt"
	"os"
	"sort"
	"time"
)

// The seam tuner: how many blocks belong on the device, decided by measurement.
//
// Unlike engine/nn/tune.go it cannot alternate per token: switching a placement costs
// a migration far larger than a token, so arms are runs of tokens with the
// migration between them, outside the timed window.
//
// It descends from the load-time placement ("as many as the card would take"),
// asking only whether fewer blocks would be faster, e.g. on a contended or slow
// device. It is off by default; see docs/engineering-history/placement.md.
type seamTuner struct {
	on      bool
	settled bool

	cands []int // block counts to try, best first
	best  int
	chal  int

	warmup  int // tokens to skip before starting
	perRun  int // tokens in one timed run
	rounds  int // ABBA quads before deciding
	tok     int
	turn    int
	quad    [4]float64
	ratios  []float64
	runN    int
	runFrom time.Time
	verbose bool
	why     string // what settled it
	trial   bool   // the stream trial (initStreamTrial), not the seam tuner
}

// SeamTuneMargin is how much faster a challenger must be to be adopted. It is
// larger than engine/nn/tune.go's because acting on the answer costs a migration, and
// a placement must not flap between two nearly-equal answers.
const SeamTuneMargin = 1.05

func (s *State) initSeamTuner() {
	if s.ld == nil || s.gpuLayers <= s.lo {
		return
	}
	full := s.gpuLayers
	cands := []int{full}
	for _, d := range []int{2, 4} {
		if n := full / d; n > 0 && n != cands[len(cands)-1] {
			cands = append(cands, n)
		}
	}
	cands = append(cands, 0)
	s.seam = &seamTuner{
		on:      true,
		cands:   cands,
		best:    full,
		chal:    cands[1],
		warmup:  s.m.opt.seamWarmup,
		perRun:  s.m.opt.seamRun,
		rounds:  s.m.opt.seamRounds,
		verbose: s.m.opt.seamVerbose,
	}
}

// initStreamTrial measures a placement that streamed blocks no card could
// hold (the incumbent) against the host (the one challenger): the seam
// tuner's ABBA runs with the migration outside the clock, and the host is
// adopted only past SeamTuneMargin, so two near-equal answers cannot flap.
// On Kimi-K3 over eight V100s the two decoded within a few percent of each
// other while the streamed one prefilled 1.5x faster (placement.md 16c), so
// which wins is a property of the box, the disk and the link, not a rule.
func (s *State) initStreamTrial() {
	if s.ld == nil || s.gpuLayers <= s.lo {
		return
	}
	full := s.gpuLayers
	s.seam = &seamTuner{
		on:      true,
		trial:   true,
		cands:   []int{full, 0},
		best:    full,
		chal:    0,
		warmup:  s.m.opt.seamWarmup,
		perRun:  s.m.opt.seamRun,
		rounds:  s.m.opt.seamRounds,
		verbose: s.m.opt.seamVerbose,
	}
}

// SetSeamTuning turns automatic relocation on or off.
//
// Deciding takes about warmup + rounds*4*perRun tokens and four migrations.
// Off is the default and leaves the load-time placement as it is.
func (s *State) SetSeamTuning(on bool) {
	if !on {
		// The stream trial is not the seam tuner and has its own switch
		// (WithStreamTrial): turning seam tuning off leaves it running.
		if s.seam != nil && !s.seam.trial {
			s.seam = nil
		}
		return
	}
	if s.seam == nil {
		s.initSeamTuner()
	}
}

// SeamTuned reports the placement the tuner settled on, and whether it settled.
func (s *State) SeamTuned() (blocks int, settled bool) {
	if s.seam == nil {
		return s.gpuLayers, false
	}
	return s.seam.best, s.seam.settled
}

// seamStep runs one token's worth of the tuner. Called once per token, after
// the token is complete.
func (s *State) seamStep() {
	t := s.seam
	if t == nil || !t.on || t.settled {
		return
	}
	t.tok++
	if t.tok <= t.warmup {
		return
	}
	// The clock starts after the migration: counting it would bias every arm
	// switched to, on the same side of every switch, which ABBA cannot cancel.
	if t.runN == 0 {
		if got := s.SetGPULayers(t.armWant()); got != t.armWant() {
			// The card would not give us this arm; drop it and settle.
			t.finish(s, "the device refused the candidate placement")
			return
		}
		t.runFrom = time.Now()
	}
	t.runN++
	if t.runN < t.perRun {
		return
	}
	rate := float64(t.runN) / time.Since(t.runFrom).Seconds()
	t.runN = 0
	t.observe(s, rate)
}

// armWant is the placement this quad position measures: ABBA, so the incumbent
// is not always the earlier and cheaper arm.
func (t *seamTuner) armWant() int {
	if q := t.turn % 4; q == 0 || q == 3 {
		return t.best
	}
	return t.chal
}

func (t *seamTuner) observe(s *State, rate float64) {
	q := t.turn % 4
	t.quad[q] = rate
	t.turn++
	if t.verbose {
		fmt.Fprintf(os.Stderr, "seam: %d blocks -> %.2f tok/s\n", t.armWant(), rate)
	}
	if q != 3 {
		return
	}
	if t.quad[0] > 0 && t.quad[1] > 0 && t.quad[2] > 0 && t.quad[3] > 0 {
		// Two ratios per quad, incumbent first in one and second in the other,
		// so drift that is linear across the four runs cancels.
		t.ratios = append(t.ratios, t.quad[1]/t.quad[0], t.quad[2]/t.quad[3])
	}
	if len(t.ratios) < t.rounds*2 {
		return
	}
	med, iqr := medianIQR(t.ratios)
	t.ratios = t.ratios[:0]
	// A dispersed measurement keeps what it has rather than guessing.
	if iqr/med > 0.10 {
		t.finish(s, fmt.Sprintf("IQR/median %.3f is too wide to decide", iqr/med))
		return
	}
	if med <= SeamTuneMargin {
		t.finish(s, fmt.Sprintf("%d blocks is not %.0f%% better (%.3f)", t.chal, (SeamTuneMargin-1)*100, med))
		return
	}
	t.best = t.chal
	for i, v := range t.cands {
		if v == t.best && i+1 < len(t.cands) {
			t.chal = t.cands[i+1]
			return
		}
	}
	t.finish(s, "no candidate left below the winner")
}

func (t *seamTuner) finish(s *State, why string) {
	t.settled = true
	t.why = why
	if t.verbose {
		fmt.Fprintf(os.Stderr, "seam: settled on %d blocks (%s)\n", t.best, why)
	}
	// nil only in the tuner's own gate, which drives observe without a model.
	if s != nil {
		s.SetGPULayers(t.best)
	}
}

func medianIQR(v []float64) (med, iqr float64) {
	c := append([]float64(nil), v...)
	sort.Float64s(c)
	at := func(f float64) float64 {
		i := int(f * float64(len(c)-1))
		return c[i]
	}
	return at(0.5), at(0.75) - at(0.25)
}
