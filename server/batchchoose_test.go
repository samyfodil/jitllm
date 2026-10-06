package server

import (
	"testing"
	"time"
)

// TestJointChoiceFollowsTheMeasurement drives the choice with step times it
// is told rather than ones it measures, so what it decides is a function of
// the numbers alone: running separately is chosen when it is faster past the
// margin and only then, a standing choice is measured again after
// jointReprobe steps and follows the new numbers, and a forced mode ignores
// them. Against a choice that settles joint whatever it measured, the
// separate-faster case fails.
func TestJointChoiceFollowsTheMeasurement(t *testing.T) {
	const ms = time.Millisecond
	probe := jointRounds * len(abba) * jointRun
	// drive runs steps of n rows; cost is a step's time given its arm.
	drive := func(c *jointChoice, n, steps int, cost func(joint bool) time.Duration) (joint, sep int) {
		for range steps {
			j := c.pick(n)
			if j {
				joint++
			} else {
				sep++
			}
			c.observe(n, j, cost(j))
		}
		return joint, sep
	}
	flat := func(jt, st time.Duration) func(bool) time.Duration {
		return func(j bool) time.Duration {
			if j {
				return jt
			}
			return st
		}
	}

	c := newJointChoice(JointMeasured)
	if j, s := drive(c, 2, probe, flat(10*ms, 5*ms)); j != probe/2 || s != probe/2 {
		t.Fatalf("a probe ran %d joint and %d separate steps, want %d of each", j, s, probe/2)
	}
	if c.pick(2) {
		t.Fatal("separate steps at half the time did not win")
	}
	drive(c, 3, probe, flat(5*ms, 10*ms))
	if !c.pick(3) {
		t.Fatal("joint steps at half the time did not win")
	}
	drive(c, 4, probe, flat(10*ms, 9800*time.Microsecond))
	if !c.pick(4) {
		t.Fatal("separate steps 2% faster displaced joint: the margin is not applied")
	}

	// Standing choices are measured again, and follow the new numbers.
	if _, s := drive(c, 2, jointReprobe, flat(10*ms, 5*ms)); s != jointReprobe {
		t.Fatalf("%d of %d steps ran separately under a standing separate choice", s, jointReprobe)
	}
	drive(c, 2, probe, flat(5*ms, 10*ms))
	if !c.pick(2) {
		t.Fatal("the re-probe did not follow numbers that had turned to joint")
	}
	if st := c.snapshot(); len(st) != 3 || st[0].rows != 2 || st[0].probes != 2 || !st[0].settled {
		t.Fatalf("snapshot %+v", st)
	}

	for _, mode := range []JointSteps{JointAlways, JointNever} {
		f := newJointChoice(mode)
		j, s := drive(f, 2, probe, flat(10*ms, 5*ms))
		if (mode == JointAlways) != (s == 0 && j == probe) || (mode == JointNever) != (j == 0 && s == probe) {
			t.Fatalf("mode %d ran %d joint and %d separate steps", mode, j, s)
		}
	}
}
