package app

import (
	"github.com/gogpu/ui/event"
	"github.com/gogpu/ui/geometry"
	"github.com/gogpu/ui/state"
	"github.com/gogpu/ui/widget"
)

// tabPage wraps one tab's content and hides its subtree from the layer tree
// while that tab is not selected.
//
// tabview.Draw draws only the selected tab, but the compositor
// (app.UpdateLayerTree) walks Children() with no selection or visibility
// check and blits every RepaintBoundary texture it finds, so an unselected
// tab's cached rows were painted over the selected one. core/collapsible
// solves this with an empty clip in Draw, which an undrawn tab never reaches,
// so this culls at the traversal instead. Children() is the whole mechanism.
type tabPage struct {
	widget.WidgetBase
	idx   int
	sel   state.Signal[int]
	child widget.Widget
}

func newTabPage(idx int, sel state.Signal[int], child widget.Widget) *tabPage {
	p := &tabPage{idx: idx, sel: sel, child: child}
	if s, ok := child.(interface{ SetParent(widget.Widget) }); ok {
		s.SetParent(p)
	}
	p.SetVisible(true)
	p.SetEnabled(true)
	return p
}

// Children reports the child only while this tab is selected, so an unselected
// screen contributes no layer and no texture.
func (p *tabPage) Children() []widget.Widget {
	if p.sel.Get() != p.idx {
		return nil
	}
	return []widget.Widget{p.child}
}

// Mount mounts the child even though Children() may be hiding it, so every
// screen's signal bindings stay live (Shell.Build depends on it). MountTree
// skips an already-mounted widget, so no double mount follows.
func (p *tabPage) Mount(ctx widget.Context) { widget.MountTree(p.child, ctx) }

// Unmount is the twin, for the same reason: the traversal that would have
// unmounted the child cannot see it.
func (p *tabPage) Unmount() { widget.UnmountTree(p.child) }

func (p *tabPage) Layout(ctx widget.Context, c geometry.Constraints) geometry.Size {
	return widget.LayoutChild(p.child, ctx, c)
}

func (p *tabPage) Draw(ctx widget.Context, canvas widget.Canvas) {
	if s, ok := p.child.(interface{ SetBounds(geometry.Rect) }); ok {
		s.SetBounds(p.Bounds())
	}
	widget.StampScreenOrigin(p.child, canvas)
	p.child.Draw(ctx, canvas)
}

func (p *tabPage) Event(ctx widget.Context, e event.Event) bool {
	return p.child.Event(ctx, e)
}
