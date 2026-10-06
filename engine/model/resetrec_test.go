//go:build linux

package model

import "testing"

// TestResetClearsTheRecurrentSummary checks Reset makes a hybrid's sequence
// fresh, forgetting the recurrent summary (a linear block's whole history) as
// well as positions and pages. The bar is bit equality against a never-used
// State, since the arithmetic is identical.
func TestResetClearsTheRecurrentSummary(t *testing.T) {
	m := hybridModelOpt(t, hyOpt{moe: true, layers: 8})
	defer m.Close()

	fresh := m.NewState(64)
	defer fresh.Close()
	if !fresh.recurrent() {
		t.Skip("this fixture has no recurrent blocks, so the gate proves nothing")
	}

	ids := []int32{1, 2, 3, 4, 5}
	want, err := fresh.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	wantCopy := append([]float32(nil), want...)

	// A second state, dirtied with a DIFFERENT sequence, then reset.
	used := m.NewState(64)
	defer used.Close()
	if _, err := used.Prefill([]int32{7, 8, 9, 10, 11, 12}); err != nil {
		t.Fatal(err)
	}
	used.Reset()
	got, err := used.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}

	if len(got) != len(wantCopy) {
		t.Fatalf("logit widths differ: %d against %d", len(got), len(wantCopy))
	}
	diff, worst := 0, float32(0)
	for i := range got {
		if got[i] != wantCopy[i] {
			diff++
			if d := got[i] - wantCopy[i]; d > worst || -d > worst {
				worst = d
				if worst < 0 {
					worst = -worst
				}
			}
		}
	}
	if diff != 0 {
		t.Errorf("%d of %d logits differ after Reset (worst |d| %g): the sequence "+
			"still carries the previous conversation's recurrent summary",
			diff, len(got), worst)
	}
}
