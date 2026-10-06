package widgets

import (
	"github.com/gogpu/ui/event"
	"github.com/gogpu/ui/geometry"
	"github.com/gogpu/ui/state"
	"github.com/gogpu/ui/widget"
)

// Hideable is a child that takes no space when its signal says it is absent.
// Lists here are built once and bound to signals, so unused slots (four device
// rows on a one-device machine) must render as nothing; an empty Text still
// reserves a line. It is zero in Layout, not only in Draw.
type Hideable struct {
	widget.WidgetBase
	show  state.ReadonlySignal[bool]
	child widget.Widget
}

// Hide wraps child so it disappears entirely when show is false.
func Hide(show state.ReadonlySignal[bool], child widget.Widget) *Hideable {
	h := &Hideable{show: show, child: child}
	if s, ok := child.(interface{ SetParent(widget.Widget) }); ok {
		s.SetParent(h)
	}
	h.SetVisible(true)
	h.SetEnabled(true)
	return h
}

func (h *Hideable) visible() bool { return h.show == nil || h.show.Get() }

// Children reports the child only while it is shown, so a hidden subtree
// contributes no layer to the compositor either -- the same cull the tab pages
// need, for the same reason.
func (h *Hideable) Children() []widget.Widget {
	if !h.visible() {
		return nil
	}
	return []widget.Widget{h.child}
}

// Mount and Unmount reach the child regardless, so its signal bindings stay
// live while it is hidden and it can come back.
func (h *Hideable) Mount(ctx widget.Context) {
	// show decides the measured size, so it invalidates the layout, not only
	// the paint.
	if h.show != nil && ctx != nil {
		if sched := ctx.Scheduler(); sched != nil {
			h.AddBinding(state.BindToSchedulerLayout(h.show, h, sched))
		}
	}
	widget.MountTree(h.child, ctx)
}
func (h *Hideable) Unmount() { widget.UnmountTree(h.child) }

func (h *Hideable) Layout(ctx widget.Context, c geometry.Constraints) geometry.Size {
	if !h.visible() {
		h.SetBounds(geometry.FromPointSize(h.Position(), geometry.Size{}))
		return geometry.Size{}
	}
	// Record our own bounds: a parent sets only the position, and without
	// this Draw would hand the child a zero rect.
	sz := widget.LayoutChild(h.child, ctx, c)
	h.SetBounds(geometry.FromPointSize(h.Position(), sz))
	return sz
}

func (h *Hideable) Draw(ctx widget.Context, canvas widget.Canvas) {
	if !h.visible() {
		return
	}
	if s, ok := h.child.(interface{ SetBounds(geometry.Rect) }); ok {
		s.SetBounds(h.Bounds())
	}
	widget.StampScreenOrigin(h.child, canvas)
	h.child.Draw(ctx, canvas)
}

func (h *Hideable) Event(ctx widget.Context, e event.Event) bool {
	if !h.visible() {
		return false
	}
	return h.child.Event(ctx, e)
}
