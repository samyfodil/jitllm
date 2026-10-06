package tier

import (
	"testing"
	"time"
)

// TestGroupTunerClimbsToTheKnee feeds the fill tuner a cost curve with its
// minimum at 4 groups: it must climb 1 -> 2 -> 4, refuse 8, and hold 4; a
// stated Config.StreamGroups is never tuned.
func TestGroupTunerClimbsToTheKnee(t *testing.T) {
	cost := map[int]time.Duration{1: 180, 2: 120, 4: 100, 8: 130, 16: 200}
	g := &devTier{Config: &Config{}}
	for i := 0; i < 400; i++ {
		n := g.streamGroups(3)
		g.fillDone(cost[n] * time.Millisecond)
	}
	if g.gtune.chal != 0 || g.gtune.best != 4 || g.StreamGroupsTuned != 4 {
		t.Fatalf("settled %v on %d groups (reported %d), want 4", g.gtune.chal == 0, g.gtune.best, g.StreamGroupsTuned)
	}
	f := &devTier{Config: &Config{StreamGroups: 2}}
	for i := 0; i < 50; i++ {
		f.fillDone(cost[f.streamGroups(3)] * time.Millisecond)
	}
	if f.gtune != nil || f.streamGroups(3) != 2 {
		t.Fatal("a stated group count was tuned")
	}
}
