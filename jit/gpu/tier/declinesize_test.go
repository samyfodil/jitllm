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

// TestAutoCacheSlotsOnKimiK3 holds the auto-streamed cache size to the one
// measured on Kimi-K3 over eight V100s: 18 sheets a block kept all 93 blocks
// on the cards (placement.md 16c). The inputs are that container's: a 0.645
// GiB base, 17.5 MB sheets, 16 routed, a 14.46 GiB budget a card.
func TestAutoCacheSlotsOnKimiK3(t *testing.T) {
	const limit = 15527837696
	if n := autoCacheSlots(limit, 16415141888-15722348544, 15722348544/896, 93, 8, 16); n != 18 {
		t.Fatalf("Kimi-K3 on eight V100s gets %d cache sheets a block, want the measured 18", n)
	}
	// One card cannot hold its share of the bases with any cache beside them.
	if n := autoCacheSlots(limit, 16415141888-15722348544, 15722348544/896, 93, 1, 16); n != 0 {
		t.Fatalf("one card gets a %d-sheet cache it has no room for", n)
	}
	if n := autoCacheSlots(limit, 1, 1, 0, 8, 16); n != 0 {
		t.Fatalf("no blocks gets a %d-sheet cache", n)
	}
}

// TestAutoStreamMarksOnlyWhatABaseFits: a block whose base fits a card is
// marked streamed on every device and then passes the size check; one whose
// base fits none is not; NoAutoStream turns it off.
func TestAutoStreamMarksOnlyWhatABaseFits(t *testing.T) {
	const gib = 1 << 30
	g := &GPU{Config: &Config{}}
	for range 2 {
		g.devs = append(g.devs, &devTier{Config: g.Config, limit: 14 * gib})
	}
	if !g.AutoStream(5, 16*gib, 15*gib, 896, 16, 93) {
		t.Fatal("a 1 GiB base on 14 GiB cards was not streamed")
	}
	for i, d := range g.devs {
		if _, ok := d.autoStream[5]; !ok {
			t.Fatalf("device %d did not mark block 5", i)
		}
	}
	if why := g.DeclineSize(5, 16*gib, 15*gib); why != "" {
		t.Fatalf("an auto-streamed block is still refused by size: %q", why)
	}
	if g.AutoStream(6, 40*gib, 15*gib, 896, 16, 93) {
		t.Fatal("a 25 GiB base was streamed onto 14 GiB cards")
	}
	if g.AutoStream(7, 16*gib, 0, 896, 16, 93) {
		t.Fatal("a block with no bank was streamed")
	}
	g.NoAutoStream = true
	if g.AutoStream(8, 16*gib, 15*gib, 896, 16, 93) {
		t.Fatal("NoAutoStream streamed a block")
	}
}
