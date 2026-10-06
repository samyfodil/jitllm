package app

import (
	"github.com/gogpu/ui/event"
	"github.com/gogpu/ui/geometry"
	"github.com/gogpu/ui/widget"
)

// The sidebar is a labelled column on a wide window and a rail of icons on a
// narrow one, where 216px would leave the pages too little room.
const (
	sidebarWide   = 216
	sidebarNarrow = 64
	// railBreak is the window width below which the sidebar is a rail. It is
	// the width the pages were designed at plus the wide sidebar.
	railBreak = 1200
)

// rail sizes the sidebar from the width it is offered -- the whole row, since
// it comes first in it -- and gives its child exactly that.
type rail struct {
	widget.WidgetBase
	child widget.Widget
}

func newRail(child widget.Widget) *rail {
	r := &rail{child: child}
	if s, ok := child.(interface{ SetParent(widget.Widget) }); ok {
		s.SetParent(r)
	}
	r.SetVisible(true)
	r.SetEnabled(true)
	return r
}

func (r *rail) Layout(ctx widget.Context, c geometry.Constraints) geometry.Size {
	w := float32(sidebarWide)
	if c.MaxWidth < railBreak {
		w = sidebarNarrow
	}
	sz := c.Constrain(geometry.Sz(w, c.MaxHeight))
	widget.LayoutChild(r.child, ctx, geometry.Tight(sz))
	r.SetBounds(geometry.FromPointSize(r.Position(), sz))
	return sz
}

func (r *rail) Draw(ctx widget.Context, cv widget.Canvas) {
	if s, ok := r.child.(interface{ SetBounds(geometry.Rect) }); ok {
		s.SetBounds(r.Bounds())
	}
	widget.StampScreenOrigin(r.child, cv)
	r.child.Draw(ctx, cv)
}

func (r *rail) Event(ctx widget.Context, e event.Event) bool { return r.child.Event(ctx, e) }
func (r *rail) Children() []widget.Widget                    { return []widget.Widget{r.child} }
