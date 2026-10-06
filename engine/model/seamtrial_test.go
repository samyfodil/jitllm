package model

import "testing"

// TestStreamTrialAdoptsOnlyPastTheMargin drives the stream trial's decision
// with rates: a host arm clearly faster than the streamed incumbent is
// adopted, one inside SeamTuneMargin is not (the two must not flap), a slower
// one is not, and a dispersed measurement keeps the incumbent.
func TestStreamTrialAdoptsOnlyPastTheMargin(t *testing.T) {
	run := func(incumbent, host []float64) *seamTuner {
		tu := &seamTuner{on: true, cands: []int{93, 0}, best: 93, chal: 0, rounds: 3}
		for r := 0; !tu.settled; r++ {
			// ABBA: incumbent, host, host, incumbent.
			for q := 0; q < 4 && !tu.settled; q++ {
				i := r % len(incumbent)
				if q == 1 || q == 2 {
					tu.observe(nil, host[i])
				} else {
					tu.observe(nil, incumbent[i])
				}
			}
		}
		return tu
	}
	steady := []float64{0.28, 0.281, 0.279}
	if tu := run(steady, []float64{0.33, 0.331, 0.329}); tu.best != 0 {
		t.Fatalf("a host 18%% faster was not adopted (%s)", tu.why)
	}
	if tu := run(steady, []float64{0.29, 0.291, 0.289}); tu.best != 93 {
		t.Fatalf("a host 3.5%% faster, inside the margin, was adopted (%s)", tu.why)
	}
	if tu := run(steady, []float64{0.20, 0.201, 0.199}); tu.best != 93 {
		t.Fatalf("a slower host was adopted (%s)", tu.why)
	}
	if tu := run(steady, []float64{0.20, 0.45, 0.30}); tu.best != 93 {
		t.Fatalf("a dispersed measurement switched the placement (%s)", tu.why)
	}
}
