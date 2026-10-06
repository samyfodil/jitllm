package tier

import (
	"strings"
	"testing"
)

// TestDeclineSizeRefusesOnlyWhatNoDeviceHolds is the size check placement asks
// before reading a block: a block above every device's whole budget is refused
// by name, one any device could hold is not, and a device that streams the
// block (the whole tier, or this block by placement) never refuses here,
// because its bank is not kept resident.
func TestDeclineSizeRefusesOnlyWhatNoDeviceHolds(t *testing.T) {
	const gib = 1 << 30
	mk := func(lims ...uint64) *GPU {
		g := &GPU{Config: &Config{}}
		for _, l := range lims {
			g.devs = append(g.devs, &devTier{Config: g.Config, limit: l})
		}
		return g
	}
	g := mk(14*gib, 14*gib)
	if why := g.DeclineSize(3, 16*gib, 15*gib); !strings.Contains(why, "above every device's whole budget") {
		t.Fatalf("a 16 GiB block on two 14 GiB cards was not refused: %q", why)
	}
	if why := g.DeclineSize(3, 13*gib, 12*gib); why != "" {
		t.Fatalf("a 13 GiB block on a 14 GiB card was refused: %q", why)
	}
	if why := mk(4*gib, 14*gib).DeclineSize(3, 13*gib, 12*gib); why != "" {
		t.Fatalf("the second card holds it, and it was refused: %q", why)
	}
	g.StreamExperts = true
	if why := g.DeclineSize(3, 16*gib, 15*gib); why != "" {
		t.Fatalf("a streamed tier was refused a block it does not keep resident: %q", why)
	}
	g.StreamExperts = false
	g.Stream(3, true)
	if why := g.DeclineSize(3, 16*gib, 15*gib); why != "" {
		t.Fatalf("a block the placement streams was refused: %q", why)
	}
	if why := g.DeclineSize(4, 16*gib, 15*gib); why == "" {
		t.Fatal("marking block 3 streamed let block 4 through")
	}
	if why := mk().DeclineSize(0, 16*gib, 0); why != "" {
		t.Fatalf("a tier with no devices refused: %q", why)
	}
}
