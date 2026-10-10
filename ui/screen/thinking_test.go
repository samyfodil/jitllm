package screen

import (
	"strings"
	"testing"

	"github.com/gogpu/ui/core/collapsible"
	"github.com/gogpu/ui/event"
	"github.com/gogpu/ui/geometry"
	"github.com/gogpu/ui/uitest"
	"github.com/gogpu/ui/widget"

	"github.com/jitllm/jitllm/ui/app"
)

// tickTree is what app.Window does before every layout pass: walk the tree and
// advance any animation. Without it a collapsible's progress never moves in a
// headless test, so a disclosure "expanded" would still draw at zero height.
func tickTree(w widget.Widget, ctx widget.Context) {
	if t, ok := w.(widget.AnimationTicker); ok {
		t.TickAnimation(ctx)
	}
	for _, ch := range w.Children() {
		tickTree(ch, ctx)
	}
}

// originOf is where target sits in root's coordinate space.
//
// A child's Bounds() are in its parent's space, so a press must be in root
// coordinates: the sum of origins down the chain.
func originOf(root, target widget.Widget) (geometry.Point, bool) {
	if root == target {
		return geometry.Point{}, true
	}
	for _, ch := range root.Children() {
		if p, ok := originOf(ch, target); ok {
			var off geometry.Point
			if b, has := ch.(interface{ Bounds() geometry.Rect }); has {
				off = b.Bounds().Min
			}
			return geometry.Pt(off.X+p.X, off.Y+p.Y), true
		}
	}
	return geometry.Point{}, false
}

// findCollapsible is how a gate reaches the widget a person clicks.
func findCollapsible(w widget.Widget) *collapsible.Widget {
	if c, ok := w.(*collapsible.Widget); ok {
		return c
	}
	for _, ch := range w.Children() {
		if c := findCollapsible(ch); c != nil {
			return c
		}
	}
	return nil
}

// Clicking a turn's reasoning must open it. The gate presses where the title
// was drawn and runs frames, because the open state is an animation and a
// test that only asserts the computed state passes against a control nobody
// can operate.
func TestClickingTheReasoningOpensIt(t *testing.T) {
	sh := app.NewShell(nil, nil)
	st := sh.Store
	const monologue = "Okay, the user wants a capital. France's is Paris."

	idx := st.AppendTurn(app.Turn{
		Role: app.RoleAssistant, Text: "Paris.", Think: monologue,
		Model: "Qwen3-1.7B-Q4_K_M.jlm", Tokens: 69, TokPerSec: 32.43,
	})
	bubble := sessionBubble(sh, idx, st.Turn(idx), false)
	ctx := mockCtx()
	widget.MountTree(bubble, ctx)

	settle := func() geometry.Size {
		var sz geometry.Size
		// Frames, the way the window runs them: tick, then lay out.
		for i := 0; i < 40; i++ {
			tickTree(bubble, ctx)
			sz = bubble.Layout(ctx, geometry.Constraints{MaxWidth: 900, MaxHeight: 4000})
		}
		return sz
	}

	shut := settle()
	cv := &uitest.MockCanvas{}
	bubble.Draw(ctx, cv)
	if !strings.Contains(drawnText(cv), "thinking") {
		t.Fatalf("the disclosure was not drawn at all:\n%s", drawnText(cv))
	}
	if st.ThinkOpen(idx).Get() {
		t.Fatal("it starts open, so this gate cannot tell a press from the default")
	}

	c := findCollapsible(bubble)
	if c == nil {
		t.Fatal("no collapsible in the bubble")
	}
	org, ok := originOf(bubble, c)
	if !ok {
		t.Fatal("the disclosure is not reachable from the bubble through Children()")
	}
	// Inside the header band, past the chevron.
	at := geometry.Pt(org.X+c.Bounds().Min.X+30, org.Y+c.Bounds().Min.Y+4)

	// Button matters: handleMouseEvent ignores anything but ButtonLeft, and a
	// zero-valued Button is ButtonNone.
	bubble.Event(ctx, &event.MouseEvent{MouseType: event.MousePress, Button: event.ButtonLeft, Position: at})
	bubble.Event(ctx, &event.MouseEvent{MouseType: event.MouseRelease, Button: event.ButtonLeft, Position: at})

	if !st.ThinkOpen(idx).Get() {
		t.Fatalf("pressing the disclosure at %v did not reach the store: the "+
			"toggle still writes nowhere", at)
	}
	if open := settle(); open.Height <= shut.Height {
		t.Errorf("it reports open and the bubble did not grow: %v then %v -- "+
			"the reasoning is not on screen", shut.Height, open.Height)
	}
}

// The "Reasoning" chip reaches turns that already have a state of their own,
// since ShowThinking is only the default a row inherits when built.
func TestTheReasoningChipReachesEveryTurn(t *testing.T) {
	sh := app.NewShell(nil, nil)
	st := sh.Store

	a := st.AppendTurn(app.Turn{Role: app.RoleAssistant, Think: "one"})
	b := st.AppendTurn(app.Turn{Role: app.RoleAssistant, Think: "two"})
	if st.ThinkOpen(a).Get() || st.ThinkOpen(b).Get() {
		t.Fatal("turns start open with the chip off")
	}

	st.SetAllThinkOpen(true)
	if !st.ThinkOpen(a).Get() || !st.ThinkOpen(b).Get() {
		t.Error("the chip did not reach turns that already existed")
	}
	// And a turn built afterwards inherits it.
	if c := st.AppendTurn(app.Turn{Role: app.RoleAssistant, Think: "three"}); !st.ThinkOpen(c).Get() {
		t.Error("a new turn did not inherit the chip")
	}

	st.SetAllThinkOpen(false)
	if st.ThinkOpen(a).Get() || st.ThinkOpen(b).Get() {
		t.Error("the chip did not close turns that already existed")
	}
}

// A turn's disclosure state survives the next message: widgets.Column rebuilds
// every child when the turn count moves, so the state lives in the Store.
func TestADisclosureSurvivesTheNextTurn(t *testing.T) {
	sh := app.NewShell(nil, nil)
	st := sh.Store

	idx := st.AppendTurn(app.Turn{Role: app.RoleAssistant, Think: "reasoning"})
	st.ThinkOpen(idx).Set(true)

	st.AppendTurn(app.Turn{Role: app.RoleUser, Text: "and again"})
	st.AppendTurn(app.Turn{Role: app.RoleAssistant})

	// What the rebuilt row asks for.
	if !st.ThinkOpen(idx).Get() {
		t.Error("the disclosure closed itself when the next turn arrived")
	}
}

// A write from outside the widget must open it too; see thinkingSection for
// why the disclosure uses Expanded + OnToggle + a watcher.
func TestAWriteFromOutsideOpensTheDisclosure(t *testing.T) {
	sh := app.NewShell(nil, nil)
	st := sh.Store
	idx := st.AppendTurn(app.Turn{
		Role: app.RoleAssistant, Text: "Paris.",
		Think: "Okay, the user wants a capital. France's is Paris.",
	})
	bubble := sessionBubble(sh, idx, st.Turn(idx), false)
	ctx := mockCtx()
	widget.MountTree(bubble, ctx)

	settle := func() geometry.Size {
		var sz geometry.Size
		for i := 0; i < 40; i++ {
			tickTree(bubble, ctx)
			sz = bubble.Layout(ctx, geometry.Constraints{MaxWidth: 900, MaxHeight: 4000})
		}
		return sz
	}
	shut := settle()

	// What the chip and the engine do.
	st.SetAllThinkOpen(true)

	c := findCollapsible(bubble)
	if c == nil {
		t.Fatal("no collapsible in the bubble")
	}
	if !c.IsExpanded() {
		t.Fatal("the widget did not follow the store: SetExpanded was never reached")
	}
	open := settle()
	if open.Height <= shut.Height {
		t.Errorf("the widget reports expanded and the bubble did not grow: %v then %v -- "+
			"progress never moved, which is the state every external write left it in",
			shut.Height, open.Height)
	}
	if c.Progress() < 1 {
		t.Errorf("the animation did not finish: progress %v", c.Progress())
	}
}
