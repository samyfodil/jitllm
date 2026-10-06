package screen

import (
	"strings"
	"testing"

	"github.com/gogpu/ui/geometry"
	"github.com/gogpu/ui/uitest"
	"github.com/gogpu/ui/widget"

	"github.com/samyfodil/jitllm/ui/app"
)

// A bubble's head must show what the turn says now, not what it said when the
// row was built (an empty placeholder, filled by SetTurn at the end). It
// renders twice from one widget, the only shape that catches a head that is
// not reactive.
func TestTheHeadFollowsTheTurnItWasBuiltFor(t *testing.T) {
	sh := app.NewShell(nil, nil)
	st := sh.Store

	idx := st.AppendTurn(app.Turn{Role: app.RoleAssistant})
	bubble := sessionBubble(sh, idx, st.Turn(idx), false)
	ctx := mockCtx()
	widget.MountTree(bubble, ctx)

	draw := func() string {
		bubble.Layout(ctx, geometry.Constraints{MaxWidth: 800, MaxHeight: 4000})
		c := &uitest.MockCanvas{}
		bubble.Draw(ctx, c)
		return drawnText(c)
	}

	if got := draw(); !strings.Contains(got, "assistant") {
		t.Fatalf("a placeholder turn should be named %q, drew:\n%s", "assistant", got)
	}

	// What the engine does when the generation finishes.
	st.SetTurn(idx, app.Turn{
		Role: app.RoleAssistant, Text: "Paris.", Tokens: 69, TokPerSec: 32.43,
		Model: "Qwen3-1.7B-Q4_K_M.jlm",
	})
	st.Touch()

	got := draw()
	for _, want := range []string{"Qwen3-1.7B-Q4_K_M.jlm", "69 tok"} {
		if !strings.Contains(got, want) {
			t.Errorf("the head never picked up %q after SetTurn+Touch; drew:\n%s", want, got)
		}
	}
	if strings.Contains(got, "assistant") {
		t.Errorf("the head kept the placeholder name; drew:\n%s", got)
	}
}

// The two halves of a bubble's head must not be drawn on top of each other:
// widgets.Paragraph is greedy, so the footer must be Intrinsic. It compares
// drawn rects, because the concatenated text looks correct either way.
func TestTheHeadDoesNotPaintItsTwoHalvesOnTopOfEachOther(t *testing.T) {
	sh := app.NewShell(nil, nil)
	st := sh.Store
	idx := st.AppendTurn(app.Turn{
		Role: app.RoleAssistant, Text: "Paris.", Tokens: 217, TokPerSec: 39.61,
		Model: "Llama-3.2-1B-Instruct-Q4_K_M.jlm",
	})

	bubble := sessionBubble(sh, idx, st.Turn(idx), false)
	ctx := mockCtx()
	widget.MountTree(bubble, ctx)
	bubble.Layout(ctx, geometry.Constraints{MaxWidth: 940, MaxHeight: 4000})
	c := &uitest.MockCanvas{}
	bubble.Draw(ctx, c)

	var head []run
	for _, x := range c.Texts {
		head = append(head, run{x.Text, x.Bounds})
	}
	for _, x := range c.StyledTexts {
		head = append(head, run{x.Text, x.Bounds})
	}

	var name, footer *run
	for i := range head {
		switch {
		case strings.Contains(head[i].text, "Llama-3.2-1B"):
			name = &head[i]
		case strings.Contains(head[i].text, "217 tok"):
			footer = &head[i]
		}
	}
	if name == nil || footer == nil {
		t.Fatalf("the head did not draw both halves; drew:\n%s", drawnText(c))
	}

	// A zero-width rect is the real failure: with a greedy footer the name gets
	// no width and DrawText still paints the glyphs (it uses the rect for
	// alignment only), so comparing origins alone passes.
	for _, r := range []*run{name, footer} {
		if r.r.Width() <= 0 {
			t.Fatalf("%q was drawn into a %vx%v rect -- it paints over whatever is beside it",
				r.text, r.r.Width(), r.r.Height())
		}
	}
	if name.r.Max.X > footer.r.Min.X {
		t.Errorf("the name spans x=%v..%v and the footer starts at x=%v -- they overlap",
			name.r.Min.X, name.r.Max.X, footer.r.Min.X)
	}
	// The footer must not be cut: an intrinsic width a pixel short wraps the
	// last word, which under MaxLines(1) is an ellipsis.
	if !strings.Contains(footer.text, "tok/s") {
		t.Errorf("the footer was truncated to %q", footer.text)
	}
}

// run is one text run the canvas recorded: what was drawn and where.
type run struct {
	text string
	r    geometry.Rect
}
