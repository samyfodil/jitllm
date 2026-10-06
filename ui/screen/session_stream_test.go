package screen

import (
	"strings"
	"testing"

	"github.com/gogpu/ui/geometry"
	"github.com/gogpu/ui/uitest"
	"github.com/gogpu/ui/widget"

	"github.com/samyfodil/jitllm/ui/app"
)

// drawSession lays out and draws the whole chat column, and returns everything
// that reached the canvas by either text path.
func drawSession(t *testing.T, sh *app.Shell, w, h float32) (string, *uitest.MockCanvas) {
	t.Helper()
	root := SessionWith(SessionOptions{})(sh)
	ctx := mockCtx()
	widget.MountTree(root, ctx)
	root.Layout(ctx, geometry.Tight(geometry.Sz(w, h)))
	c := &uitest.MockCanvas{}
	root.Draw(ctx, c)

	var b strings.Builder
	for _, x := range c.Texts {
		b.WriteString(x.Text)
		b.WriteString("\n")
	}
	for _, x := range c.StyledTexts {
		b.WriteString(x.Text)
		b.WriteString("\n")
	}
	return b.String(), c
}

// The in-flight reply is rendered exactly once: a second copy through a
// plain primitives.Text used to paint an unwrapped line mid-window.
func TestTheStreamingReplyIsRenderedOnce(t *testing.T) {
	sh := app.NewShell(nil, nil)
	sh.Store.AppendTurn(app.Turn{Role: app.RoleUser, Text: "hello"})
	sh.Store.AppendTurn(app.Turn{Role: app.RoleAssistant, Text: ""})
	const marker = "zzmarkerzz"
	sh.Store.Stream.Set(marker)

	got, _ := drawSession(t, sh, 900, 700)
	if n := strings.Count(got, marker); n != 1 {
		t.Errorf("the in-flight reply reached the canvas %d time(s), want 1:\n%s", n, got)
	}
}

// Every horizontal band in the chat column must span the window (the stats
// strip's background stopped part way without a stretched cross axis). It
// checks bands rather than the widest rect, because the full-width page
// background makes a max() pass against the violation.
func TestEveryBandInTheColumnSpansTheWindow(t *testing.T) {
	sh := app.NewShell(nil, nil)
	sh.Store.PromptTokS.Set(6.09)

	const w = 900
	_, c := drawSession(t, sh, w, 700)

	bands := 0
	for _, r := range c.Rects {
		// A band: a short filled strip starting at the left edge. The tall
		// page and transcript rects are not what this is about.
		if r.Bounds.Min.X != 0 || r.Bounds.Height() > 100 {
			continue
		}
		bands++
		if r.Bounds.Width() < w-1 {
			t.Errorf("a %0.fpx band at y=%.0f is %.0f wide of %d: the column was "+
				"laid out at its content's width, so its background stops short "+
				"of the window", r.Bounds.Height(), r.Bounds.Min.Y, r.Bounds.Width(), w)
		}
	}
	if bands == 0 {
		t.Fatal("no bands were drawn at all, so this gate asserted nothing")
	}
}
