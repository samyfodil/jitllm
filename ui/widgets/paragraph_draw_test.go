package widgets

import (
	"strings"
	"testing"

	"github.com/gogpu/ui/geometry"
	"github.com/gogpu/ui/primitives"
	"github.com/gogpu/ui/state"
	"github.com/gogpu/ui/uitest"
	"github.com/gogpu/ui/widget"
)

// A signal change must reach the canvas without an intervening Layout, since a
// cached row only runs Draw. It drives Layout once and then draws twice.
func TestParagraphDrawsTheCurrentSignalWithoutRelayout(t *testing.T) {
	sig := state.NewSignal("hello")
	p := NewParagraph("").ContentSignal(sig.AsReadonly()).FontSize(14)

	// One layout, as a cached row gets. The text is non-empty so the row has
	// a real height.
	p.Layout(nil, geometry.Constraints{MaxWidth: 400, MaxHeight: 400})

	canvas := &uitest.MockCanvas{}
	sig.Set("world")
	p.Draw(nil, canvas)
	got := drawn(canvas)
	if !strings.Contains(got, "world") {
		t.Errorf("after the signal changed the canvas got %q: Draw is painting "+
			"lines Layout computed, so a streamed reply shows whatever it was "+
			"when the row was built", got)
	}
	if strings.Contains(got, "hello") {
		t.Errorf("the canvas still shows the old value: %q", got)
	}
}

// Wrapping must follow the bounds the widget was given, not a width it
// remembered from a different constraint.
func TestParagraphWrapsToItsBounds(t *testing.T) {
	sig := state.NewSignal(strings.Repeat("word ", 40))
	p := NewParagraph("").ContentSignal(sig.AsReadonly()).FontSize(14)
	p.Layout(nil, geometry.Constraints{MaxWidth: 200, MaxHeight: 800})

	canvas := &uitest.MockCanvas{}
	p.Draw(nil, canvas)
	if len(canvas.Texts) < 2 {
		t.Fatalf("a 200-word paragraph drew %d line(s): it is not wrapping",
			len(canvas.Texts))
	}
	// Every painted line must sit inside the widget's own width.
	w := p.Bounds().Width()
	for i, c := range canvas.Texts {
		if c.Bounds.Width() > w+0.5 {
			t.Errorf("line %d is %v wide against a %v bound", i, c.Bounds.Width(), w)
		}
	}
}

// drawn collects what reached the canvas by either path: the shipped
// configuration sets a family and draws through DrawStyledText.
func drawn(c *uitest.MockCanvas) string {
	var b strings.Builder
	for _, t := range c.Texts {
		b.WriteString(t.Text)
		b.WriteString("\n")
	}
	for _, t := range c.StyledTexts {
		b.WriteString(t.Text)
		b.WriteString("\n")
	}
	return b.String()
}

// A family must reach the canvas AS a family, and the text must survive the
// trip. Both halves matter: the styled path is the one the transcript takes,
// and it is where a reply can vanish without any gate noticing. This asserts
// the wiring; glyph coverage is gated in app.
func TestParagraphDrawsThroughTheStyledPathWhenAFamilyIsSet(t *testing.T) {
	const family = "DejaVu Sans"
	p := NewParagraph("hello world").FontSize(14).FontFamily(family)
	p.Layout(nil, geometry.Constraints{MaxWidth: 400, MaxHeight: 400})

	canvas := &uitest.MockCanvas{}
	p.Draw(nil, canvas)

	if len(canvas.StyledTexts) == 0 {
		t.Fatalf("a paragraph with a family drew %d plain and 0 styled calls: "+
			"the family never reached the canvas", len(canvas.Texts))
	}
	for _, c := range canvas.StyledTexts {
		if c.Style.FontFamily != family {
			t.Errorf("styled call carried family %q, want %q", c.Style.FontFamily, family)
		}
	}
	if got := drawn(canvas); !strings.Contains(got, "hello") {
		t.Errorf("the canvas got %q, want the text", got)
	}

	// And with no family it must take the plain path, or the fallback that
	// makes a missing font harmless is not being exercised at all.
	q := NewParagraph("hello world").FontSize(14)
	q.Layout(nil, geometry.Constraints{MaxWidth: 400, MaxHeight: 400})
	plain := &uitest.MockCanvas{}
	q.Draw(nil, plain)
	if len(plain.StyledTexts) != 0 {
		t.Errorf("a paragraph with no family made %d styled calls", len(plain.StyledTexts))
	}
	if len(plain.Texts) == 0 {
		t.Error("a paragraph with no family drew nothing at all")
	}
}

// The lines painted must be the lines the row was measured for: a canvas
// re-wrap in Draw leaves a blank band or drops lines past the bound.
//
// uitest.MockCanvas measures at runes*fontSize*0.5 against the widget's own
// 0.6 estimate, so the two budgets differ by 20% at any width and the counts
// cannot coincide by luck.
func TestParagraphPaintsExactlyTheLinesItWasMeasuredFor(t *testing.T) {
	p := NewParagraph(strings.Repeat("word ", 300)).
		FontSize(14).LineHeight(1.35).FontFamily("DejaVu Sans")

	sz := p.Layout(nil, geometry.Constraints{MaxWidth: 400, MaxHeight: 40000})
	measured := int(sz.Height / (14 * 1.35))

	c := &uitest.MockCanvas{}
	p.Draw(nil, c)
	painted := len(c.Texts) + len(c.StyledTexts)

	if painted != measured {
		t.Errorf("the row was measured for %d line(s) and painted %d: "+
			"%d line(s) of blank band, or text dropped at the bound",
			measured, painted, measured-painted)
	}
}

// A line whose runes need different fonts must be drawn as one run per font,
// left to right, with no gaps and no overlap. The chain is a fake, so this
// gates the split only; coverage is app.FamilyFor's and is gated there.
func TestAMixedLineIsDrawnOneRunPerFamily(t *testing.T) {
	// "upper" draws capitals, "lower" draws everything else. Nothing here
	// knows what a script is -- neither does the widget.
	chain := func(r rune, prefer string) string {
		want := "lower"
		if r >= 'A' && r <= 'Z' {
			want = "upper"
		}
		return want
	}
	p := NewParagraph("abcDEFghi").FontSize(14).Font(Face{For: chain})
	p.Layout(nil, geometry.Constraints{MaxWidth: 4000, MaxHeight: 400})

	c := &uitest.MockCanvas{}
	p.Draw(nil, c)

	if len(c.StyledTexts) != 3 {
		t.Fatalf("a mixed line drew %d run(s), want 3: %+v", len(c.StyledTexts), c.StyledTexts)
	}
	wantFam := []string{"lower", "upper", "lower"}
	wantTxt := []string{"abc", "DEF", "ghi"}
	prevX := float32(-1)
	for i, call := range c.StyledTexts {
		if call.Style.FontFamily != wantFam[i] {
			t.Errorf("run %d used family %q, want %q", i, call.Style.FontFamily, wantFam[i])
		}
		if call.Text != wantTxt[i] {
			t.Errorf("run %d drew %q, want %q", i, call.Text, wantTxt[i])
		}
		if call.Bounds.Min.X <= prevX {
			t.Errorf("run %d starts at x=%.1f, not after the previous run's x=%.1f: "+
				"the runs are stacked on top of each other", i, call.Bounds.Min.X, prevX)
		}
		prevX = call.Bounds.Min.X
	}
}

// A line every family in the chain covers must be one call, byte for byte what
// a single family would have drawn; the per-run path cannot centre.
func TestAUniformLineIsStillOneCall(t *testing.T) {
	chain := func(rune, string) string { return "one" }
	p := NewParagraph("all the same font").FontSize(14).Font(Face{For: chain})
	p.Layout(nil, geometry.Constraints{MaxWidth: 4000, MaxHeight: 400})

	c := &uitest.MockCanvas{}
	p.Draw(nil, c)

	if len(c.StyledTexts) != 1 {
		t.Fatalf("a uniform line drew %d call(s), want 1", len(c.StyledTexts))
	}
	if c.StyledTexts[0].Text != "all the same font" {
		t.Errorf("drew %q", c.StyledTexts[0].Text)
	}
}

// A content change must invalidate the layout, not only the paint, or a
// growing reply keeps the height it was first measured at.
func TestAContentChangeInvalidatesTheLayout(t *testing.T) {
	sig := state.NewSignal("short")
	p := NewParagraph("").ContentSignal(sig.AsReadonly()).FontSize(14)

	ctx := uitest.NewMockContext()
	ctx.SchedulerVal = state.NewScheduler(func([]widget.Widget) {})
	parent := primitives.VBox(p)
	widget.MountTree(parent, ctx)

	parent.Layout(ctx, geometry.Constraints{MaxWidth: 400, MaxHeight: 4000})
	if !p.IsLayoutCacheValid() {
		t.Fatal("the paragraph was never measured through widget.LayoutChild, so " +
			"MarkNeedsLayout is a permanent no-op and this gate would pass for " +
			"the wrong reason")
	}

	sig.Set(strings.Repeat("a much longer reply arriving one token at a time. ", 20))
	if p.IsLayoutCacheValid() {
		t.Error("the layout cache survived a content change: the bubble keeps the " +
			"height it had when the row was built, however much text arrives")
	}
}

// MaxLines must cap the lines drawn and mark the cut, which
// primitives.Text.Ellipsis() does not do.
func TestMaxLinesCapsWhatIsDrawn(t *testing.T) {
	const content = "a reply long enough to need several lines at this width, " +
		"with more words after the cap than before it, so the cut is visible"

	p := NewParagraph(content).FontSize(14).MaxLines(1)
	sz := p.Layout(nil, geometry.Constraints{MaxWidth: 200, MaxHeight: 400})

	c := &uitest.MockCanvas{}
	p.Draw(nil, c)
	got := drawn(c)

	if n := strings.Count(strings.TrimSpace(got), "\n") + 1; n != 1 {
		t.Errorf("drew %d line(s) with MaxLines(1):\n%s", n, got)
	}
	if !strings.Contains(got, "…") {
		t.Errorf("the cut is not marked: %q", got)
	}
	if strings.Contains(got, "visible") {
		t.Errorf("text past the cap was drawn: %q", got)
	}
	// The height must match what is drawn, or the row reserves space for
	// lines that were never painted.
	if want := float32(14 * 1.35); sz.Height > want+1 {
		t.Errorf("measured %v high for one line of %v", sz.Height, want)
	}
}

// Uncapped is unchanged.
func TestNoCapDrawsEverything(t *testing.T) {
	const content = "one two three four five six seven eight nine ten eleven twelve"
	p := NewParagraph(content).FontSize(14)
	p.Layout(nil, geometry.Constraints{MaxWidth: 200, MaxHeight: 400})
	c := &uitest.MockCanvas{}
	p.Draw(nil, c)
	if got := drawn(c); !strings.Contains(got, "twelve") {
		t.Errorf("an uncapped paragraph dropped text: %q", got)
	}
}

// A one-line label is cut where the real text runs out of room, not where the
// mean advance says it should: a file name of capitals and digits is wider than
// running English.
func TestALabelIsCutWhereTheTextReallyEnds(t *testing.T) {
	wide := func(s string) float32 { return float32(len([]rune(s))) * 10 }
	if got := fitLine("ABCDEFGHIJ", 100, wide); got != "ABCDEFGHIJ" {
		t.Errorf("a line that fits was cut to %q", got)
	}
	got := fitLine("ABCDEFGHIJKL", 100, wide)
	if wide(got) > 100 || !strings.HasSuffix(got, "…") {
		t.Errorf("a line too wide became %q, %.0f px of 100", got, wide(got))
	}
}
