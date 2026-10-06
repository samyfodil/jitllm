package widgets

import (
	"strings"

	"github.com/gogpu/ui/event"
	"github.com/gogpu/ui/geometry"
	"github.com/gogpu/ui/widget"
)

// caret is a position in the wrapped text: which line, and how many runes into
// it.
type caret struct{ line, rune int }

func (a caret) before(b caret) bool {
	return a.line < b.line || (a.line == b.line && a.rune < b.rune)
}

// selection is an anchor and a cursor over the lines a Paragraph last drew.
type selection struct {
	anchor, cursor caret
	active         bool
	// dragging is true between a press and its release.
	dragging bool
	// pending is a point the mouse landed on that has not been resolved to a
	// caret yet. The hit test happens in Draw because only the canvas can
	// measure text and Paragraph wraps there; it costs one frame.
	pending    *geometry.Point
	pendingFor bool // true = the point sets the anchor, false = it extends
}

// ordered returns the selection with the anchor first.
func (s *selection) ordered() (caret, caret) {
	if s.cursor.before(s.anchor) {
		return s.cursor, s.anchor
	}
	return s.anchor, s.cursor
}

// empty reports whether nothing is selected.
func (s *selection) empty() bool {
	return !s.active || s.anchor == s.cursor
}

// Selectable turns dragging and copying on for this paragraph. The toolkit's
// only selection implementation (core/textfield) is unexported and
// single-line, so this works against the lines the paragraph already computes.
func (p *Paragraph) Selectable() *Paragraph {
	p.selectable = true
	return p
}

// IsFocusable lets the paragraph take keyboard focus, which is what makes
// ctrl+C reach it.
func (p *Paragraph) IsFocusable() bool { return p.selectable }

// Event handles press, drag, release and copy.
func (p *Paragraph) Event(ctx widget.Context, e event.Event) bool {
	if !p.selectable {
		return false
	}
	switch ev := e.(type) {
	case *event.MouseEvent:
		return p.mouse(ctx, ev)
	case *event.KeyEvent:
		return p.key(ev)
	}
	return false
}

func (p *Paragraph) mouse(ctx widget.Context, e *event.MouseEvent) bool {
	switch e.MouseType {
	case event.MousePress:
		if !p.Bounds().Contains(e.Position) {
			// A press elsewhere clears the selection, which is what every
			// other text surface does.
			if !p.sel.empty() {
				p.sel = selection{}
				p.SetNeedsRedraw(true)
			}
			return false
		}
		at := e.Position
		p.sel = selection{active: true, dragging: true, pending: &at, pendingFor: true}
		if ctx != nil {
			if f, ok := ctx.(interface{ RequestFocus(widget.Widget) }); ok {
				f.RequestFocus(p)
			}
			// Capture the pointer, as core/slider does: moves and the release
			// then arrive here without hit-testing, so a drag can leave the
			// line it started on.
			if pc, ok := ctx.(widget.PointerCapturer); ok {
				pc.CapturePointer(p)
			}
		}
		p.SetNeedsRedraw(true)
		return true

	case event.MouseMove:
		if !p.sel.dragging {
			return false
		}
		// No bounds check: a drag belongs to the widget that started it.
		at := e.Position
		p.sel.pending, p.sel.pendingFor = &at, false
		p.SetNeedsRedraw(true)
		return true

	case event.MouseRelease:
		if !p.sel.dragging {
			return false
		}
		p.sel.dragging = false
		if ctx != nil {
			if pc, ok := ctx.(widget.PointerCapturer); ok {
				pc.ReleasePointer(p)
			}
		}
		return true
	}
	return false
}

func (p *Paragraph) key(e *event.KeyEvent) bool {
	if e.KeyType != event.KeyPress || !e.Modifiers().IsCtrl() {
		return false
	}
	if e.Rune != 'c' && e.Rune != 'C' && e.Key != event.KeyC {
		return false
	}
	if s := p.Selected(); s != "" {
		widget.ClipboardWrite(s)
		return true
	}
	return false
}

// Selected is the selected text, in logical order. The selection indexes the
// visual (possibly bidi-reordered) line that was clicked, and this maps it
// back so Arabic is not copied reversed.
func (p *Paragraph) Selected() string {
	if p.sel.empty() || len(p.drawn) == 0 {
		return ""
	}
	from, to := p.sel.ordered()
	var b strings.Builder
	for li := from.line; li <= to.line && li < len(p.drawn); li++ {
		r := []rune(p.drawn[li])
		lo, hi := 0, len(r)
		if li == from.line {
			lo = clampi(from.rune, 0, len(r))
		}
		if li == to.line {
			hi = clampi(to.rune, 0, len(r))
		}
		if lo > hi {
			lo, hi = hi, lo
		}
		if li > from.line {
			b.WriteString("\n")
		}
		b.WriteString(string(r[lo:hi]))
	}
	// A line that was reordered for display cannot be cut accurately in
	// visual space, so the whole of it is returned in logical order rather
	// than a reversed fragment.
	out := b.String()
	if hasRTL(out) {
		return p.logicalFor(from.line, to.line)
	}
	return out
}

// logicalFor returns whole lines, unreordered, for a bidirectional range.
func (p *Paragraph) logicalFor(from, to int) string {
	var b strings.Builder
	for li := from; li <= to && li < len(p.logical); li++ {
		if li > from {
			b.WriteString("\n")
		}
		b.WriteString(p.logical[li])
	}
	return b.String()
}

func clampi(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// resolvePending turns a recorded mouse point into a caret, now that the
// canvas can measure and the lines are known. It does work only when something
// is pending.
func (p *Paragraph) resolvePending(canvas widget.Canvas, r geometry.Rect, lh float32) {
	if p.sel.pending == nil || len(p.drawn) == 0 {
		return
	}
	at := *p.sel.pending
	p.sel.pending = nil

	li := int((at.Y - r.Min.Y) / lh)
	li = clampi(li, 0, len(p.drawn)-1)
	c := caret{line: li, rune: p.runeAt(canvas, p.drawn[li], at.X-r.Min.X)}

	if p.sel.pendingFor {
		p.sel.anchor, p.sel.cursor = c, c
		return
	}
	p.sel.cursor = c
}

// runeAt is how many runes of line fit before x, by binary search over
// measured prefixes: proportional mixed-script text has no column width.
func (p *Paragraph) runeAt(canvas widget.Canvas, line string, x float32) int {
	if x <= 0 {
		return 0
	}
	r := []rune(line)
	if p.measure(canvas, line) <= x {
		return len(r)
	}
	lo, hi := 0, len(r)
	for lo < hi {
		mid := (lo + hi + 1) / 2
		if p.measure(canvas, string(r[:mid])) <= x {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	return lo
}

// measure is the width of s as this paragraph draws it.
func (p *Paragraph) measure(canvas widget.Canvas, s string) float32 {
	if s == "" {
		return 0
	}
	if sd, ok := canvas.(widget.StyledTextDrawer); ok && p.family != "" {
		return sd.MeasureStyledText(s, p.style())
	}
	return canvas.MeasureText(s, p.fontSize, p.bold)
}

// paintSelection draws the highlight behind the selected part of one line,
// before the text goes on top of it.
func (p *Paragraph) paintSelection(canvas widget.Canvas, li int, line string, box geometry.Rect) {
	if p.sel.empty() {
		return
	}
	from, to := p.sel.ordered()
	if li < from.line || li > to.line {
		return
	}
	r := []rune(line)
	lo, hi := 0, len(r)
	if li == from.line {
		lo = clampi(from.rune, 0, len(r))
	}
	if li == to.line {
		hi = clampi(to.rune, 0, len(r))
	}
	if lo >= hi {
		return
	}
	x0 := p.measure(canvas, string(r[:lo]))
	x1 := p.measure(canvas, string(r[:hi]))
	canvas.DrawRect(
		geometry.FromPointSize(
			geometry.Pt(box.Min.X+x0, box.Min.Y),
			geometry.Sz(x1-x0, box.Height()),
		),
		p.selColor,
	)
}

// SelectionColor sets the highlight.
func (p *Paragraph) SelectionColor(c widget.Color) *Paragraph {
	p.selColor = c
	return p
}
