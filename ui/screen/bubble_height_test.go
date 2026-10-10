package screen

import (
	"strings"
	"testing"

	"github.com/gogpu/ui/geometry"
	"github.com/gogpu/ui/uitest"
	"github.com/gogpu/ui/widget"

	"github.com/jitllm/jitllm/ui/app"
)

// A turn bubble must be as tall as the text it paints, and no taller. The
// widget-level gate cannot see a mismatch because it lays out and draws at one
// width; in the bubble, Layout uses the constraint width and Draw the bounds
// width, with padding, gap and stretch between them.
func TestABubbleIsAsTallAsTheTextItPaints(t *testing.T) {
	sh := app.NewShell(nil, nil)
	const w = 1560

	// Realistic: several paragraphs, blank lines, mixed scripts.
	body := strings.Join([]string{
		"### **1. Arabic**",
		"",
		"- **Transliteration:** marhaba",
		"مرحبا، كيف حالك؟",
		"",
		"### **2. Chinese**",
		"",
		"- **Standard term:** 序言 (Xùyán)",
		"*序言：在字裡行間，尋找靈魂的倒影。*",
	}, "\n")

	bubble := sessionBubble(sh, 0, app.Turn{Role: app.RoleAssistant, Text: body, Tokens: 1216}, false)
	ctx := mockCtx()
	widget.MountTree(bubble, ctx)
	sz := bubble.Layout(ctx, geometry.Constraints{MaxWidth: w, MaxHeight: 10000})

	c := &uitest.MockCanvas{}
	bubble.Draw(ctx, c)

	// The lowest pixel any text was drawn at.
	var lowest float32
	for _, x := range c.Texts {
		if b := x.Bounds.Max.Y; b > lowest {
			lowest = b
		}
	}
	for _, x := range c.StyledTexts {
		if b := x.Bounds.Max.Y; b > lowest {
			lowest = b
		}
	}
	if lowest == 0 {
		t.Fatal("the bubble drew no text at all, so this gate asserted nothing")
	}

	// Padding is real and expected, and so is the reply's row of actions --
	// 28px icons under a 6px gap -- which draws no text; hundreds of pixels
	// are not.
	const slack = 40 + 34
	if gap := sz.Height - lowest; gap > slack {
		t.Errorf("the bubble is %.0f tall and its last line ends at %.0f: "+
			"%.0f empty pixels below the text", sz.Height, lowest, gap)
	}
	t.Logf("bubble %.0f tall, text ends at %.0f, %d line(s) drawn",
		sz.Height, lowest, len(c.Texts)+len(c.StyledTexts))
}

// A turn with reasoning gets a collapsed section; one without gets nothing.
// It also asserts the answer is still drawn, since a reply placed inside the
// collapsed section would pass every count-based check.
func TestReasoningIsCollapsedAndTheAnswerIsNot(t *testing.T) {
	sh := app.NewShell(nil, nil)

	with := sessionBubble(sh, 0, app.Turn{
		Role: app.RoleAssistant, Text: "the visible answer", Think: "the monologue",
	}, false)
	ctx := mockCtx()
	widget.MountTree(with, ctx)
	with.Layout(ctx, geometry.Constraints{MaxWidth: 900, MaxHeight: geometry.Infinity})
	c := &uitest.MockCanvas{}
	with.Draw(ctx, c)
	got := drawnText(c)

	if !strings.Contains(got, "the visible answer") {
		t.Errorf("the answer was not drawn: %q", got)
	}
	if !strings.Contains(got, "thinking") {
		t.Errorf("no disclosure for the reasoning at all: %q", got)
	}
	// Collapsed means drawn inside an empty clip, not absent: core/collapsible
	// still draws hidden content inside PushClip(Rect{}) so stale textures are
	// culled, and a MockCanvas records the draw either way.
	if !hasEmptyClip(c) {
		t.Errorf("no empty clip was pushed: the reasoning is drawn expanded, "+
			"not collapsed: %q", got)
	}

	// No reasoning: no disclosure.
	plain := sessionBubble(sh, 0, app.Turn{Role: app.RoleAssistant, Text: "just an answer"}, false)
	widget.MountTree(plain, ctx)
	plain.Layout(ctx, geometry.Constraints{MaxWidth: 900, MaxHeight: geometry.Infinity})
	c2 := &uitest.MockCanvas{}
	plain.Draw(ctx, c2)
	if got := drawnText(c2); strings.Contains(got, "thinking") {
		t.Errorf("a turn with no reasoning still shows a disclosure: %q", got)
	}
	if hasEmptyClip(c2) {
		t.Error("a turn with no reasoning still built a collapsed section")
	}
}

// hasEmptyClip reports whether anything was drawn inside a zero-size clip,
// which is how core/collapsible hides its content.
func hasEmptyClip(c *uitest.MockCanvas) bool {
	for _, r := range c.Clips {
		if r.Width() <= 0 || r.Height() <= 0 {
			return true
		}
	}
	return false
}

func drawnText(c *uitest.MockCanvas) string {
	var b strings.Builder
	for _, x := range c.Texts {
		b.WriteString(x.Text)
		b.WriteString("\n")
	}
	for _, x := range c.StyledTexts {
		b.WriteString(x.Text)
		b.WriteString("\n")
	}
	return b.String()
}
