package tier

import (
	"sort"
	"time"
)

// groupTuner picks how many groups a streamed fill splits its selection into
// (Config.StreamGroups), by measuring fills. More groups put more reads in
// flight behind the transfers and restart the page-locked pipeline more often;
// on Kimi-K3 over the V100s 4 beat 1 and 16 lost to both (placement.md 16c),
// and where the knee sits is the disk's, the link's and the sheet size's, so it
// is measured, not tabulated.
//
// It follows engine/nn/tune.go's pack tuner: paired duels of best against the
// next width up, interleaved in ABBA quads so drift cancels, climbing while the
// challenger wins by more than the margin and settling otherwise. A run is one
// token's fills on this device, so both arms see the same blocks.
type groupTuner struct {
	best, chal int // chal 0 once settled
	cur        int
	per        int // fills in one run: the device's streamed blocks
	n          int // fills so far in this run
	t          time.Duration
	turn       int
	quad       [4]time.Duration
	rat        []float64
}

const (
	groupTuneRound  = 6    // paired ratios per duel
	groupTuneMargin = 0.98 // a challenger must be >2% faster
	groupTuneMaxIQR = 0.10
	groupTuneMax    = 16
)

// streamGroups is the group count for this fill: Config.StreamGroups when
// set, else the tuner's current arm. per is the device's streamed blocks.
func (g *devTier) streamGroups(per int) int {
	if g.StreamGroups != 0 {
		return g.StreamGroups
	}
	t := g.gtune
	if t == nil {
		b := max(g.StreamGroupsStart, 1)
		t = &groupTuner{best: b, chal: b * 2, cur: b}
		g.gtune = t
	}
	t.per = max(per, 1)
	return t.cur
}

// fillDone charges one fill's wall to the tuner and moves it on at the end of
// a run.
func (g *devTier) fillDone(d time.Duration) {
	t := g.gtune
	if t == nil || g.StreamGroups != 0 || t.chal == 0 {
		return
	}
	t.t += d
	if t.n++; t.n < t.per {
		return
	}
	q := t.turn % 4
	t.quad[q], t.n, t.t = t.t, 0, 0
	t.turn++
	if q == 3 {
		// Incumbent at 0 and 3, challenger at 1 and 2: time ratios, so below
		// one is the challenger faster.
		t.rat = append(t.rat, float64(t.quad[1])/float64(t.quad[0]), float64(t.quad[2])/float64(t.quad[3]))
		if len(t.rat) >= groupTuneRound {
			med, iqr := medianIQRf(t.rat)
			t.rat = t.rat[:0]
			switch {
			case iqr/med > groupTuneMaxIQR || med >= groupTuneMargin:
				t.chal = 0
			case t.chal*2 > groupTuneMax:
				t.best, t.chal = t.chal, 0
			default:
				t.best, t.chal = t.chal, t.chal*2
			}
		}
	}
	// ABBA: the next run's arm.
	if t.chal == 0 {
		t.cur = t.best
	} else if p := t.turn % 4; p == 1 || p == 2 {
		t.cur = t.chal
	} else {
		t.cur = t.best
	}
	g.StreamGroupsTuned = t.best
}

func medianIQRf(v []float64) (med, iqr float64) {
	c := append([]float64(nil), v...)
	sort.Float64s(c)
	at := func(f float64) float64 { return c[int(f*float64(len(c)-1))] }
	return at(0.5), at(0.75) - at(0.25)
}
