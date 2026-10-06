package widgets

import (
	"testing"

	"github.com/gogpu/ui/geometry"
	"github.com/gogpu/ui/uitest"
	"github.com/gogpu/ui/widget"
)

func constraintsWide() geometry.Constraints {
	return geometry.Constraints{MaxWidth: 4000, MaxHeight: 400}
}

// mockDrawCanvas records only what the styled path drew, in order.
type mockDrawCanvas struct {
	uitest.MockCanvas
	styled []string
}

func (c *mockDrawCanvas) DrawStyledText(s string, b geometry.Rect, st widget.TextStyle) {
	c.styled = append(c.styled, s)
	c.MockCanvas.DrawStyledText(s, b, st)
}

const (
	// مرحبا -- "marhaba", read right to left: meem, reh, hah, beh, alef.
	hello = "مرحبا"
	// Its runes in the order they must be drawn left to right.
	helloVisual = "ابحرم"
	// שלום -- Hebrew, to show nothing here is about Arabic.
	shalom       = "שלום"
	shalomVisual = "םולש"
)

// A right-to-left line must be drawn in the order it is read. The Hebrew case
// checks that this is the Unicode algorithm, not a rule about one script.
func TestARightToLeftLineIsDrawnInReadingOrder(t *testing.T) {
	for _, c := range []struct{ name, in, want string }{
		{"arabic", hello, helloVisual},
		{"hebrew", shalom, shalomVisual},
		{"latin untouched", "hello world", "hello world"},
		{"empty", "", ""},
		{"digits are not RTL", "12345", "12345"},
		{"CJK is not RTL", "你好", "你好"},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := visualOrder(c.in); got != c.want {
				t.Errorf("visualOrder(%q)\n got %q\nwant %q", c.in, got, c.want)
			}
		})
	}
}

// A mixed line keeps each script in its own direction, with the runs in
// paragraph order (left to right here, as the line opens with Latin).
func TestAMixedLineKeepsEachScriptsDirection(t *testing.T) {
	got := visualOrder("hi " + hello + " bye")
	want := "hi " + helloVisual + " bye"
	if got != want {
		t.Errorf("visualOrder\n got %q\nwant %q", got, want)
	}
}

// A right-to-left paragraph orders its runs right to left too, so an embedded
// English phrase lands left of the Arabic that follows it.
func TestARightToLeftParagraphOrdersItsRunsRightToLeft(t *testing.T) {
	got := visualOrder(hello + " ok " + shalom)
	// Read right to left: the Arabic run is rightmost, the Hebrew leftmost.
	want := shalomVisual + " ok " + helloVisual
	if got != want {
		t.Errorf("visualOrder\n got %q\nwant %q", got, want)
	}
}

// The fast path must not change any answer -- it may only skip work.
func TestTheFastPathOnlySkipsWork(t *testing.T) {
	if hasRTL("plain ascii") {
		t.Error("ascii took the slow path")
	}
	if !hasRTL(hello) {
		t.Error("Arabic took the fast path and would never be reordered")
	}
	if !hasRTL(shalom) {
		t.Error("Hebrew took the fast path")
	}
}

// And Draw must actually apply it; the tests above call visualOrder directly.
func TestDrawSendsTheLineInVisualOrder(t *testing.T) {
	p := NewParagraph(hello).FontSize(14).FontFamily("F")
	p.Layout(nil, constraintsWide())

	c := &mockDrawCanvas{}
	p.Draw(nil, c)

	if len(c.styled) != 1 {
		t.Fatalf("drew %d call(s), want 1", len(c.styled))
	}
	if c.styled[0] == hello {
		t.Fatal("the canvas got the LOGICAL order: the line is drawn as it is " +
			"stored, so a right-to-left reader sees it backwards")
	}
	if c.styled[0] != helloVisual {
		t.Errorf("the canvas got %q, want %q", c.styled[0], helloVisual)
	}
}
