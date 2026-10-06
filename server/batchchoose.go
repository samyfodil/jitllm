package server

import (
	"slices"
	"sync"
	"time"
)

// JointSteps says how the step loop runs a decode step's rows when it could
// run them as one joint step (model.StepRuns across sessions).
type JointSteps int

const (
	// JointMeasured times both ways for each row count, in situ, and runs the
	// faster (jointChoice). The default.
	JointMeasured JointSteps = iota
	// JointAlways runs every step that can be joint as one.
	JointAlways
	// JointNever runs every session's token alone, one after another: the
	// one-at-a-time path's arithmetic, interleaved per token.
	JointNever
)

// A joint step is not always the faster one. It reads every block's weights
// once for all its rows, and it pays for that in ragged kernels and a wider
// head; one session after another pays a full read each, plus a graph
// recapture at every session switch. Which wins depends on the model, the
// device and the row count -- a 2-row step on a 1B model can sit near the
// crossover -- so the loop measures rather than assumes, the way the
// engine's own tuners do (model's seamtune, the tier's tuneSplit): arms run in
// ABBA order as runs of steps, compared by median, and a challenger is adopted
// only past a margin so a near-tie cannot flap.
const (
	// jointRun is the steps in one timed run of an arm. Its first step is not
	// timed: coming from the other arm it can record a launch sequence that
	// the rest of the run replays (a joint step's graph for this row count),
	// a cost a settled choice pays once, not every step.
	jointRun = 4
	// jointRounds is the ABBA quads a probe takes before deciding: two give
	// six timed steps an arm, enough for a median when the arms differ by the
	// tens of percent that decide anything here.
	jointRounds = 2
	// jointMargin is how much faster running separately must be to be
	// chosen: model.SeamTuneMargin's 5%, for the same reason. Joint is the
	// incumbent because it is what the loop exists for.
	jointMargin = 1.05
	// jointReprobe is the steps a choice stands before it is measured again:
	// the device's load and the sessions' history lengths drift, and a probe
	// costs 32 steps, under 2% of the steps between two.
	jointReprobe = 2048
)

// jointChoice holds a choice per row count. Only the loop goroutine calls
// pick and observe; telemetry reads under mu.
type jointChoice struct {
	mode JointSteps

	mu  sync.Mutex
	per map[int]*jointArm
}

// jointArm is one row count's probe, or its standing choice.
type jointArm struct {
	settled bool
	joint   bool
	since   int
	// Probing: turn is the run's index in the ABBA sequence, step the steps
	// taken in it.
	turn, step int
	j, s       []time.Duration
	// The last probe's medians, for telemetry.
	jMed, sMed time.Duration
	probes     int
}

func newJointChoice(mode JointSteps) *jointChoice {
	return &jointChoice{mode: mode, per: map[int]*jointArm{}}
}

// abba is the arm of each run in a quad: joint, separate, separate, joint.
var abba = [4]bool{true, false, false, true}

// pick says whether a step of n rows runs joint.
func (c *jointChoice) pick(n int) bool {
	switch c.mode {
	case JointAlways:
		return true
	case JointNever:
		return false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	a := c.per[n]
	if a == nil {
		a = &jointArm{}
		c.per[n] = a
	}
	if a.settled {
		return a.joint
	}
	return abba[a.turn%len(abba)]
}

// observe records how long a step of n rows took on the arm pick chose, and
// moves the probe on; a probe that has its rounds decides.
func (c *jointChoice) observe(n int, joint bool, d time.Duration) {
	if c.mode != JointMeasured {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	a := c.per[n]
	if a == nil {
		return
	}
	if a.settled {
		if a.since++; a.since >= jointReprobe {
			*a = jointArm{jMed: a.jMed, sMed: a.sMed, probes: a.probes}
		}
		return
	}
	if a.step > 0 {
		if joint {
			a.j = append(a.j, d)
		} else {
			a.s = append(a.s, d)
		}
	}
	if a.step++; a.step < jointRun {
		return
	}
	a.step = 0
	if a.turn++; a.turn < jointRounds*len(abba) {
		return
	}
	a.jMed, a.sMed = median(a.j), median(a.s)
	a.joint = !(float64(a.sMed)*jointMargin < float64(a.jMed))
	a.settled, a.since, a.turn, a.j, a.s = true, 0, 0, nil, nil
	a.probes++
}

func median(v []time.Duration) time.Duration {
	if len(v) == 0 {
		return 0
	}
	v = slices.Clone(v)
	slices.Sort(v)
	return v[len(v)/2]
}

// choiceStat is one row count's standing as telemetry reports it.
type choiceStat struct {
	rows           int
	settled        bool
	joint          bool
	jointMedian    time.Duration
	separateMedian time.Duration
	probes         int
}

func (c *jointChoice) snapshot() []choiceStat {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]choiceStat, 0, len(c.per))
	for n, a := range c.per {
		out = append(out, choiceStat{n, a.settled, a.joint, a.jMed, a.sMed, a.probes})
	}
	slices.SortFunc(out, func(x, y choiceStat) int { return x.rows - y.rows })
	return out
}
