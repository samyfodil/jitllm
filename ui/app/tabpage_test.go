package app

import (
	"testing"

	"github.com/gogpu/ui/event"
	"github.com/gogpu/ui/geometry"
	"github.com/gogpu/ui/state"
	"github.com/gogpu/ui/uitest"
	"github.com/gogpu/ui/widget"
)

// boundary is a leaf that is a RepaintBoundary, which is what a listview row
// is (core/listview/decorator.go:32) and therefore what the compositor blits.
type boundary struct {
	widget.WidgetBase
	mounted int
}

func newBoundary() *boundary {
	b := &boundary{}
	b.SetRepaintBoundary(true)
	b.SetVisible(true)
	return b
}

func (b *boundary) Layout(widget.Context, geometry.Constraints) geometry.Size {
	return geometry.Sz(10, 10)
}
func (b *boundary) Draw(widget.Context, widget.Canvas)     {}
func (b *boundary) Event(widget.Context, event.Event) bool { return false }
func (b *boundary) Children() []widget.Widget              { return nil }

// Mount and Unmount, because widget.Lifecycle requires both and WidgetBase
// supplies neither; with only Mount, MountTree silently skips the fixture.
func (b *boundary) Mount(widget.Context) { b.mounted++ }
func (b *boundary) Unmount()             {}

// countBoundaries walks the tree the way the compositor does -- app/layer_tree.go
// recurses Children() and emits a layer for every RepaintBoundary, with no
// check of tab selection and no check of IsVisible.
func countBoundaries(w widget.Widget) int {
	n := 0
	if b, ok := w.(interface{ IsRepaintBoundary() bool }); ok && b.IsRepaintBoundary() {
		n++
	}
	for _, c := range w.Children() {
		n += countBoundaries(c)
	}
	return n
}

// An unselected tab must contribute nothing to the layer tree, and its child
// must still be mounted. It gates the compositor walk, not Draw: tabview
// already drew only the selected tab while hidden textures were still blitted.
func TestUnselectedTabContributesNoLayer(t *testing.T) {
	sel := state.NewSignal(0)
	first, second := newBoundary(), newBoundary()
	p0 := newTabPage(0, sel, first)
	p1 := newTabPage(1, sel, second)

	if got := countBoundaries(p0); got != 1 {
		t.Errorf("the SELECTED tab contributes %d boundaries, want 1 -- it must still composite", got)
	}
	if got := countBoundaries(p1); got != 0 {
		t.Errorf("an UNSELECTED tab contributes %d boundaries, want 0: its textures "+
			"are blitted over whatever tab is selected, at the coordinates it had "+
			"when it was last visible", got)
	}

	// Selecting it brings it back, or switching away would be one-way.
	sel.Set(1)
	if got := countBoundaries(p1); got != 1 {
		t.Errorf("after selecting it the tab contributes %d boundaries, want 1", got)
	}
	if got := countBoundaries(p0); got != 0 {
		t.Errorf("after deselecting it the first tab contributes %d, want 0", got)
	}
}

// The cull must not cost the mount: every screen's signal bindings stay live
// across tab switches.
func mountCtx() *uitest.MockContext {
	ctx := uitest.NewMockContext()
	ctx.SchedulerVal = state.NewScheduler(func([]widget.Widget) {})
	return ctx
}

func TestAnUnselectedTabIsStillMounted(t *testing.T) {
	sel := state.NewSignal(0)
	hidden := newBoundary()
	p := newTabPage(1, sel, hidden) // never selected

	widget.MountTree(p, mountCtx())
	if hidden.mounted != 1 {
		t.Fatalf("a hidden tab's content was mounted %d times, want 1: its signal "+
			"bindings are dead, so the screen stops tracking the store", hidden.mounted)
	}

	// Selecting it must not mount it a second time now the outer walk can see
	// the child.
	sel.Set(1)
	widget.MountTree(p, mountCtx())
	if hidden.mounted != 1 {
		t.Errorf("content mounted %d times after selecting the tab, want 1", hidden.mounted)
	}
}
