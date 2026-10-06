package model

import "testing"

// setKVPageForTest pins this model's KV page size for the States it builds
// afterwards and returns a function restoring it. It lives in a test file and
// writes the model's own option, so a release binary has no way to reach it
// and two models in one process never share it.
func (m *Model) setKVPageForTest(p int) func() {
	old := m.opt.kvPage
	m.opt.kvPage = p
	return func() { m.opt.kvPage = old }
}

// TestKVPageIsPerModel: two models in one process, one with its KV page pinned,
// build States in both orders. Each State pages at its own model's size.
func TestKVPageIsPerModel(t *testing.T) {
	pages := func(s *State) int {
		for li := range s.kv.layers {
			if p := s.kv.layers[li].p; p > 0 {
				return p
			}
		}
		t.Fatal("the fixture has no paged layer")
		return 0
	}
	for _, pinnedFirst := range []bool{true, false} {
		a := hybridModelOpt(t, hyOpt{})
		b := hybridModelOpt(t, hyOpt{})
		probe := b.NewState(64)
		want := pages(probe)
		probe.Close()
		if want == kvKeyTile {
			t.Fatalf("the default page is already %d positions; the pin would not show", kvKeyTile)
		}
		defer a.setKVPageForTest(kvKeyTile)()
		var sa, sb *State
		if pinnedFirst {
			sa, sb = a.NewState(64), b.NewState(64)
		} else {
			sb, sa = b.NewState(64), a.NewState(64)
		}
		if p := pages(sa); p != kvKeyTile {
			t.Errorf("pinned first=%v: the pinned model pages at %d, want %d", pinnedFirst, p, kvKeyTile)
		}
		if p := pages(sb); p != want {
			t.Errorf("pinned first=%v: the other model pages at %d, want its default %d", pinnedFirst, p, want)
		}
		sa.Close()
		sb.Close()
	}
}
