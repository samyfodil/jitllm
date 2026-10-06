package widgets

import (
	"strings"
	"testing"

	"github.com/gogpu/ui/event"
	"github.com/gogpu/ui/geometry"
	"github.com/gogpu/ui/primitives"
	"github.com/gogpu/ui/state"
	"github.com/gogpu/ui/uitest"
	"github.com/gogpu/ui/widget"
)

// lay out and draw once, so the paragraph knows its lines.
func draw(p *Paragraph, w, h float32) *uitest.MockCanvas {
	p.Layout(nil, geometry.Constraints{MaxWidth: w, MaxHeight: h})
	p.SetBounds(geometry.FromPointSize(geometry.Pt(0, 0), geometry.Sz(w, h)))
	c := &uitest.MockCanvas{}
	p.Draw(nil, c)
	return c
}

func press(p *Paragraph, x, y float32) {
	p.Event(nil, &event.MouseEvent{MouseType: event.MousePress, Position: geometry.Pt(x, y)})
}
func drag(p *Paragraph, x, y float32) {
	p.Event(nil, &event.MouseEvent{MouseType: event.MouseMove, Position: geometry.Pt(x, y)})
}

// A drag must select the text between the two points.
func TestADragSelectsTheTextBetweenTwoPoints(t *testing.T) {
	p := NewParagraph("hello world").FontSize(14).Selectable()
	draw(p, 400, 100)

	if got := p.Selected(); got != "" {
		t.Fatalf("something was selected before any click: %q", got)
	}

	press(p, 0, 5)
	draw(p, 400, 100) // resolves the press
	drag(p, 4000, 5)  // far right: to the end of the line
	draw(p, 400, 100) // resolves the drag

	if got := p.Selected(); got != "hello world" {
		t.Errorf("selected %q, want the whole line", got)
	}
}

// The highlight must be painted, and behind the text.
func TestTheSelectionIsPainted(t *testing.T) {
	p := NewParagraph("hello world").FontSize(14).Selectable().
		SelectionColor(widget.Color{R: 0, G: 0, B: 1, A: 1})
	draw(p, 400, 100)
	press(p, 0, 5)
	draw(p, 400, 100)
	drag(p, 4000, 5)

	c := draw(p, 400, 100)
	if len(c.Rects) == 0 {
		t.Fatal("no highlight was drawn for a live selection")
	}
	if len(c.Texts)+len(c.StyledTexts) == 0 {
		t.Fatal("the text was not drawn at all")
	}
}

// A press outside clears it.
func TestAPressOutsideClearsTheSelection(t *testing.T) {
	p := NewParagraph("hello world").FontSize(14).Selectable()
	draw(p, 400, 100)
	press(p, 0, 5)
	draw(p, 400, 100)
	drag(p, 4000, 5)
	draw(p, 400, 100)
	if p.Selected() == "" {
		t.Fatal("nothing selected to clear")
	}

	p.Event(nil, &event.MouseEvent{MouseType: event.MousePress, Position: geometry.Pt(0, 9000)})
	if got := p.Selected(); got != "" {
		t.Errorf("a press outside left %q selected", got)
	}
}

// A paragraph that is not selectable ignores the mouse entirely.
func TestAPlainParagraphIgnoresTheMouse(t *testing.T) {
	p := NewParagraph("hello world").FontSize(14)
	draw(p, 400, 100)
	if p.Event(nil, &event.MouseEvent{MouseType: event.MousePress, Position: geometry.Pt(5, 5)}) {
		t.Error("a plain paragraph consumed a mouse press")
	}
	if p.Selected() != "" {
		t.Error("a plain paragraph selected something")
	}
}

// Selection is by rune, so a multi-byte line is never cut in half.
func TestSelectionCutsByRune(t *testing.T) {
	const s = "日本語のテキストです"
	p := NewParagraph(s).FontSize(14).Selectable()
	draw(p, 4000, 100)
	press(p, 0, 5)
	draw(p, 4000, 100)
	drag(p, 4000, 5)
	draw(p, 4000, 100)

	got := p.Selected()
	if got != s {
		t.Errorf("selected %q, want %q", got, s)
	}
	if strings.ContainsRune(got, '�') {
		t.Errorf("the selection split a rune: %q", got)
	}
}

// A press must reach a paragraph through a real tree, in the right
// coordinates. The other tests call Paragraph.Event directly with coordinates
// already correct; this builds the transcript's shape (a column of boxes, each
// wrapping a paragraph) and enters at the top.
func TestAPressReachesTheRightParagraphThroughTheTree(t *testing.T) {
	first := NewParagraph("first bubble text").FontSize(14).Selectable()
	second := NewParagraph("second bubble text").FontSize(14).Selectable()

	n := state.NewSignal(2)
	col := NewColumn(n.AsReadonly(), 0, func(i int) widget.Widget {
		if i == 0 {
			return primitives.Box(first).PaddingXY(0, 0)
		}
		return primitives.Box(second).PaddingXY(0, 0)
	})

	ctx := uitest.NewMockContext()
	ctx.SchedulerVal = state.NewScheduler(func([]widget.Widget) {})
	widget.MountTree(col, ctx)
	col.Layout(ctx, geometry.Constraints{MaxWidth: 400, MaxHeight: 1000})
	c := &uitest.MockCanvas{}
	col.Draw(ctx, c)

	// The second bubble's vertical middle, in column coordinates.
	y := second.Bounds().Min.Y
	if b, ok := col.Children()[1].(interface{ Bounds() geometry.Rect }); ok {
		y += b.Bounds().Min.Y
	}
	y += 5

	col.Event(ctx, &event.MouseEvent{MouseType: event.MousePress, Position: geometry.Pt(2, y)})
	col.Draw(ctx, c) // resolve the press
	col.Event(ctx, &event.MouseEvent{MouseType: event.MouseMove, Position: geometry.Pt(4000, y)})
	col.Draw(ctx, c)

	if got := second.Selected(); got == "" {
		t.Errorf("a press inside the second bubble selected nothing there")
	}
	if got := first.Selected(); got != "" {
		t.Errorf("the FIRST bubble took a press that landed in the second: %q", got)
	}
}
