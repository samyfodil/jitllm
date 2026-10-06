package widgets

import (
	"github.com/gogpu/ui/event"
	"github.com/gogpu/ui/geometry"
	"github.com/gogpu/ui/widget"
)

// pressState is the toolkit's own button protocol (core/button/event.go): the
// window hit-tests and sends Enter and Leave to the widget under the pointer,
// which trusts them rather than re-testing a position, and a click is a press
// and then a release inside the widget.
type pressState struct {
	hover, down bool
}

type pressTarget interface {
	Bounds() geometry.Rect
	SetNeedsRedraw(bool)
}

// handle runs the protocol for w and reports whether it took the event.
// enabled false keeps the hover (a disabled button still lights its tooltip
// region) but drops the click and the pointer cursor.
func (p *pressState) handle(w pressTarget, ctx widget.Context, me *event.MouseEvent, enabled bool, click func()) bool {
	redraw := func() {
		w.SetNeedsRedraw(true)
		if ctx != nil {
			ctx.InvalidateRect(w.Bounds())
		}
	}
	switch me.MouseType {
	case event.MouseEnter:
		p.hover = true
		if ctx != nil && enabled {
			ctx.SetCursor(widget.CursorPointer)
		}
		redraw()
		return true
	case event.MouseLeave:
		p.hover, p.down = false, false
		if ctx != nil {
			ctx.SetCursor(widget.CursorDefault)
		}
		redraw()
		return true
	case event.MousePress:
		if me.Button != event.ButtonLeft || !enabled {
			return false
		}
		p.down = true
		return true
	case event.MouseRelease:
		if me.Button != event.ButtonLeft {
			return false
		}
		was := p.down
		p.down = false
		if was && enabled && click != nil && w.Bounds().Contains(me.Position) {
			click()
		}
		return was
	}
	return false
}
