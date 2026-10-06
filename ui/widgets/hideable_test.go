package widgets

import (
	"testing"

	"github.com/gogpu/ui/event"
	"github.com/gogpu/ui/geometry"
	"github.com/gogpu/ui/state"
	"github.com/gogpu/ui/uitest"
	"github.com/gogpu/ui/widget"
)

type stub struct {
	widget.WidgetBase
	drew    int
	mounted int
}

func newStub() *stub {
	s := &stub{}
	s.SetVisible(true)
	return s
}

func (s *stub) Layout(widget.Context, geometry.Constraints) geometry.Size {
	return geometry.Sz(100, 40)
}
func (s *stub) Draw(_ widget.Context, c widget.Canvas) {
	s.drew++
	c.DrawText("here", s.Bounds(), 12, widget.ColorBlack, false, widget.TextAlignLeft)
}
func (s *stub) Event(widget.Context, event.Event) bool { return false }
func (s *stub) Children() []widget.Widget              { return nil }
func (s *stub) Mount(widget.Context)                   { s.mounted++ }
func (s *stub) Unmount()                               {}

// A hidden child must take no space at all: zero in Layout, not only in Draw.
func TestAHiddenChildTakesNoSpace(t *testing.T) {
	show := state.NewSignal(true)
	child := newStub()
	h := Hide(show.AsReadonly(), child)

	if got := h.Layout(nil, geometry.Constraints{MaxWidth: 300, MaxHeight: 300}); got.Height != 40 {
		t.Fatalf("shown height %v, want 40", got.Height)
	}
	show.Set(false)
	if got := h.Layout(nil, geometry.Constraints{MaxWidth: 300, MaxHeight: 300}); got.Height != 0 {
		t.Errorf("hidden height %v, want 0: the row still reserves its space", got.Height)
	}

	c := &uitest.MockCanvas{}
	h.Draw(nil, c)
	if len(c.Texts) != 0 {
		t.Errorf("a hidden child drew %d text call(s)", len(c.Texts))
	}

	// And it comes back and actually draws, so a wrapper that always hides
	// cannot pass.
	show.Set(true)
	if got := h.Layout(nil, geometry.Constraints{MaxWidth: 300, MaxHeight: 300}); got.Height != 40 {
		t.Errorf("after showing again height %v, want 40", got.Height)
	}
	if h.Bounds().Height() != 40 {
		t.Errorf("the wrapper's own bounds are %v after layout: the child is "+
			"handed a zero rect at Draw and paints into nothing", h.Bounds())
	}
	shown := &uitest.MockCanvas{}
	h.Draw(nil, shown)
	if len(shown.Texts) == 0 {
		t.Error("a SHOWN child drew nothing at all")
	}
}

// A hidden child contributes nothing to the compositor's traversal either,
// which walks Children() with no visibility check.
func TestAHiddenChildLeavesTheLayerTree(t *testing.T) {
	show := state.NewSignal(false)
	h := Hide(show.AsReadonly(), newStub())
	if got := h.Children(); len(got) != 0 {
		t.Errorf("a hidden child is still in the tree: %d child(ren)", len(got))
	}
	show.Set(true)
	if got := h.Children(); len(got) != 1 {
		t.Errorf("a shown child is missing from the tree: %d", len(got))
	}
}

// Hiding must not unmount: the child's signal bindings have to survive so it
// can come back showing current data rather than whatever it had when hidden.
func TestHidingDoesNotUnmount(t *testing.T) {
	show := state.NewSignal(false)
	child := newStub()
	h := Hide(show.AsReadonly(), child)

	ctx := uitest.NewMockContext()
	ctx.SchedulerVal = state.NewScheduler(func([]widget.Widget) {})
	widget.MountTree(h, ctx)

	if child.mounted != 1 {
		t.Errorf("a hidden child was mounted %d times, want 1: its bindings are dead", child.mounted)
	}
}
