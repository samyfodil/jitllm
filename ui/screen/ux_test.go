package screen

import (
	"strings"
	"testing"

	"github.com/gogpu/ui/event"
	"github.com/gogpu/ui/geometry"
	"github.com/gogpu/ui/primitives"
	"github.com/gogpu/ui/uitest"
	"github.com/gogpu/ui/widget"

	"github.com/samyfodil/jitllm/ui/app"
)

// An empty transcript must say what to do next, differently before and after
// a model is loaded: it is the first screen every user meets.
func TestAnEmptyTranscriptSaysWhatToDo(t *testing.T) {
	sh := app.NewShell(nil, nil)
	st := sh.Store

	tr, _, _ := sessionTranscript(sh)
	ctx := mockCtx()
	widget.MountTree(tr, ctx)

	draw := func() string {
		tr.Layout(ctx, geometry.Constraints{MaxWidth: 900, MaxHeight: 800})
		c := &uitest.MockCanvas{}
		tr.Draw(ctx, c)
		return drawnText(c)
	}

	cold := draw()
	for _, want := range []string{"No model loaded", "Models"} {
		if !strings.Contains(cold, want) {
			t.Errorf("the cold start never says %q; it drew:\n%s", want, cold)
		}
	}

	st.Loaded.Set(true)
	st.ModelPath.Set("/m/Qwen3-1.7B-Q4_K_M.jlm")
	ready := draw()
	if !strings.Contains(ready, "Qwen3-1.7B-Q4_K_M") {
		t.Errorf("a loaded model is not named on the empty screen:\n%s", ready)
	}
	if strings.Contains(ready, "No model loaded") {
		t.Errorf("it still says there is no model:\n%s", ready)
	}

	// And it gets out of the way once there is a conversation.
	st.AppendTurn(app.Turn{Role: app.RoleUser, Text: "hello"})
	if got := draw(); strings.Contains(got, "is ready") {
		t.Errorf("the hint is still on screen with a turn in the transcript:\n%s", got)
	}
}

// The footer must not describe a model that is not loaded, and must say when
// the one that is pages from disk: a rate alone cannot tell a fully resident
// run from one a page short.
func TestTheFooterIsSilentWithNoModelAndNamesPaging(t *testing.T) {
	sh := app.NewShell(nil, nil)
	st := sh.Store
	footer := sessionFooter(sh)

	st.AppendTurn(app.Turn{Role: app.RoleUser, Text: "hello"})
	if got := footer.Get(); got != "" {
		t.Errorf("with no model the footer reads %q", got)
	}
	st.Loaded.Set(true)
	st.Pager.Set(app.PagerStat{Frames: 12, PageIns: 400, PageOuts: 388, TurnOuts: 36, BytesRead: 40 << 30})
	if got := footer.Get(); !strings.Contains(got, "paging") {
		t.Errorf("with a paging model loaded the footer reads %q", got)
	}
}

// Escape must stop a reply in flight, and must not eat the key otherwise
// (dialogs and menus need it when nothing is generating).
func TestEscapeStopsAReplyAndIsOtherwiseTransparent(t *testing.T) {
	var stops int
	sh := app.NewShell(nil, nil)
	Attach(sh, Deps{Engine: &fakeEngine{stop: func() { stops++ }}})
	col := sessionColumn(sh, SessionOptions{MaxTokens: 8})
	ctx := mockCtx()
	widget.MountTree(col, ctx)
	col.Layout(ctx, geometry.Constraints{MaxWidth: 900, MaxHeight: 800})

	esc := &event.KeyEvent{KeyType: event.KeyPress, Key: event.KeyEscape}

	if col.Event(ctx, esc) {
		t.Error("Escape was consumed while nothing was generating")
	}
	if stops != 0 {
		t.Errorf("Stop was called %d time(s) with nothing in flight", stops)
	}

	sh.Store.Busy.Set(true)
	if !col.Event(ctx, esc) {
		t.Fatal("Escape did not reach the session while a reply was in flight")
	}
	if stops != 1 {
		t.Errorf("Stop called %d time(s), want 1", stops)
	}
}

// The transcript must follow a reply as it is written, and stop following the
// moment the reader scrolls away from the bottom. Both directions are
// asserted, since a fix for one that breaks the other reads as a fix.
func TestTheTranscriptFollowsAReplyButNotAReader(t *testing.T) {
	sh := app.NewShell(nil, nil)
	st := sh.Store
	st.Loaded.Set(true)

	tr, follow, scroll := sessionTranscript(sh)
	root := primitives.VBox(primitives.Expanded(tr), follow).
		CrossAlign(primitives.CrossAxisStretch)
	ctx := mockCtx()
	widget.MountTree(root, ctx)

	// A conversation long enough to overflow a short viewport.
	for i := 0; i < 6; i++ {
		st.AppendTurn(app.Turn{Role: app.RoleUser, Text: "a question that takes a line or two to write out"})
		st.AppendTurn(app.Turn{Role: app.RoleAssistant, Text: strings.Repeat("an answer with enough words in it to wrap. ", 6)})
	}
	frame := func() {
		root.Layout(ctx, geometry.Constraints{MaxWidth: 600, MaxHeight: 300})
		c := &uitest.MockCanvas{}
		root.Draw(ctx, c)
		sh.DrainQueue(0)
	}
	frame()
	frame()

	// Streaming into the last turn.
	st.AppendTurn(app.Turn{Role: app.RoleAssistant})
	frame()
	frame()
	at := scroll.Get()

	st.Stream.Set(strings.Repeat("the reply keeps growing and growing. ", 20))
	frame()
	frame()
	if after := scroll.Get(); after <= at {
		t.Errorf("the reply grew and the view did not follow: %v then %v", at, after)
	}

	// Now the reader scrolls up. Further tokens must leave them there.
	scroll.Set(0)
	frame()
	st.Stream.Set(strings.Repeat("and more still, well past the bottom. ", 40))
	frame()
	frame()
	if after := scroll.Get(); after > followSlack {
		t.Errorf("the reader scrolled up and was dragged back to %v", after)
	}
}

// The newest assistant turn's reasoning must be reachable after the stream
// that carried it is gone: isLiveTurn stays true for that turn, so the
// disclosure must fall back to the turn's own copy as the body does.
func TestTheNewestTurnKeepsItsReasoning(t *testing.T) {
	sh := app.NewShell(nil, nil)
	st := sh.Store

	st.AppendTurn(app.Turn{Role: app.RoleUser, Text: "think about it"})
	idx := st.AppendTurn(app.Turn{
		Role: app.RoleAssistant, Text: "Paris.",
		Think: "Okay, the user wants a capital. France's is Paris.",
	})
	if !isLiveTurn(st, idx) {
		t.Fatal("the last assistant turn is not the live one: this gate is testing the wrong path")
	}
	// What sending the next message does.
	st.Stream.Set("")
	st.StreamThink.Set("")

	bubble := sessionBubble(sh, idx, st.Turn(idx), true)
	ctx := mockCtx()
	widget.MountTree(bubble, ctx)
	bubble.Layout(ctx, geometry.Constraints{MaxWidth: 900, MaxHeight: 4000})
	c := &uitest.MockCanvas{}
	bubble.Draw(ctx, c)

	got := drawnText(c)
	if !strings.Contains(got, "thinking") {
		t.Errorf("the reasoning disclosure is gone once the stream is empty; drew:\n%s", got)
	}
	if !strings.Contains(got, "Paris.") {
		t.Errorf("the reply is gone too; drew:\n%s", got)
	}
}

// A model chip must measure narrow enough to leave its neighbours alone: an
// HBox positions a sibling by its child's measured width and DrawText does
// not clip.
func TestAModelChipLabelFitsItsSlot(t *testing.T) {
	long := "Qwen3-Next-80B-A3B-Instruct-Q4_K_M.jlm"
	got := chipLabel(long)
	if n := len([]rune(got)); n > chipLabelRunes {
		t.Errorf("chipLabel(%q) is %d runes, over the %d budget: %q", long, n, chipLabelRunes, got)
	}
	if !strings.HasSuffix(got, "…") {
		t.Errorf("a shortened label does not say so: %q", got)
	}
	// The extension is noise in a switch; the summary row carries the full name.
	if strings.Contains(got, ".jlm") {
		t.Errorf("the container extension survived into a chip: %q", got)
	}
	// A short name is left exactly alone.
	if got := chipLabel("gemma-2b.jlm"); got != "gemma-2b" {
		t.Errorf("a short name was rewritten to %q", got)
	}
}

// Clear must ask first, and must not clear under a reply that is still
// running; a confirm answered mid-decode would drop the reply.
func TestClearAsksAndWillNotClearUnderARunningReply(t *testing.T) {
	sh := app.NewShell(nil, nil)
	st := sh.Store
	st.AppendTurn(app.Turn{Role: app.RoleUser, Text: "hello"})
	st.Draft.Set("unsent")

	sessionClear(sh)
	sh.DrainQueue(0) // a dialog is raised from the UI goroutine
	if st.Turns.Get() == 0 {
		t.Fatal("Clear emptied the transcript without asking")
	}
	d := st.Dialog.Get()
	if d.Kind != app.DialogConfirm || d.OnOK == nil {
		t.Fatalf("Clear raised %+v, want a confirm", d)
	}

	st.Busy.Set(true)
	d.OnOK()
	if st.Turns.Get() == 0 {
		t.Error("the confirm cleared the transcript under a running reply")
	}

	st.Busy.Set(false)
	d.OnOK()
	if st.Turns.Get() != 0 || st.Draft.Get() != "" {
		t.Errorf("confirming did not clear: %d turn(s), draft %q", st.Turns.Get(), st.Draft.Get())
	}
}

// A cleared transcript must start again from the top: scrollview draws its
// offset unclamped, so a surviving offset hides the next message's first
// line.
func TestClearTakesTheTranscriptBackToTheTop(t *testing.T) {
	sh := app.NewShell(nil, nil)
	st := sh.Store
	st.Loaded.Set(true)

	tr, follow, scroll := sessionTranscript(sh)
	root := primitives.VBox(primitives.Expanded(tr), follow).
		CrossAlign(primitives.CrossAxisStretch)
	ctx := mockCtx()
	widget.MountTree(root, ctx)
	frame := func() {
		root.Layout(ctx, geometry.Constraints{MaxWidth: 600, MaxHeight: 300})
		root.Draw(ctx, &uitest.MockCanvas{})
		sh.DrainQueue(0)
	}

	for i := 0; i < 6; i++ {
		st.AppendTurn(app.Turn{Role: app.RoleUser, Text: "a question that takes a line or two to write out"})
		st.AppendTurn(app.Turn{Role: app.RoleAssistant, Text: strings.Repeat("an answer with enough words in it to wrap. ", 6)})
	}
	frame()
	frame()
	if scroll.Get() <= 0 {
		t.Fatal("setup: the long conversation did not scroll, so this gate would pass without the fix")
	}

	st.ClearConversation()
	frame()
	st.AppendTurn(app.Turn{Role: app.RoleUser, Text: "what is my name?"})
	st.AppendTurn(app.Turn{Role: app.RoleAssistant, Text: "I don't know."})
	frame()
	frame()
	if got := scroll.Get(); got != 0 {
		t.Errorf("the new conversation is drawn scrolled by %v, cutting off its first line", got)
	}
}

// Clear must be offered only when it would remove something.
func TestClearIsOfferedOnlyWhenThereIsSomethingToClear(t *testing.T) {
	st := app.NewStore()
	off := sessionClearDisabled(st)
	if !off.Get() {
		t.Error("Clear is enabled on an empty screen")
	}
	st.Draft.Set("half a thought")
	if off.Get() {
		t.Error("Clear is disabled with a draft to clear")
	}
	st.Draft.Set("")
	st.Attach.Set([]string{"/pic.png"})
	if off.Get() {
		t.Error("Clear is disabled with a picture attached")
	}
	st.Attach.Set(nil)
	st.AppendTurn(app.Turn{Role: app.RoleUser, Text: "hi"})
	if off.Get() {
		t.Error("Clear is disabled with a conversation")
	}
	st.Busy.Set(true)
	if !off.Get() {
		t.Error("Clear is enabled while a reply runs")
	}
}

// Pictures attached for a model that cannot see them must hold Send back and
// say why, where the pictures are.
func TestPicturesForABlindModelHoldSendAndSayWhy(t *testing.T) {
	sh := app.NewShell(nil, nil)
	st := sh.Store
	st.Loaded.Set(true)
	st.Draft.Set("what colour is it?")
	st.Attach.Set([]string{"/pic.png"})

	st.Vision.Set(true)
	if sessionSendDisabled(st).Get() {
		t.Fatal("setup: Send is disabled for a vision model with a picture")
	}
	st.Vision.Set(false)
	if !sessionSendDisabled(st).Get() {
		t.Error("Send is lit with a picture attached to a model that cannot see it")
	}

	strip := sessionAttachments(sh)
	ctx := mockCtx()
	widget.MountTree(strip, ctx)
	strip.Layout(ctx, geometry.Constraints{MaxWidth: 600, MaxHeight: 400})
	c := &uitest.MockCanvas{}
	strip.Draw(ctx, c)
	if got := drawnText(c); !strings.Contains(got, "can't see pictures") {
		t.Errorf("the strip does not say why Send is held; drew:\n%s", got)
	}
}

// A problem must be shown in full, with its fix one press away, and take no
// space when there is none.
func TestAProblemIsShownWithItsFix(t *testing.T) {
	sh := app.NewShell(nil, nil)
	st := sh.Store
	banner := sessionProblem(sh)
	ctx := mockCtx()
	widget.MountTree(banner, ctx)
	draw := func() string {
		banner.Layout(ctx, geometry.Constraints{MaxWidth: 600, MaxHeight: 400})
		c := &uitest.MockCanvas{}
		banner.Draw(ctx, c)
		return drawnText(c)
	}
	if got := draw(); strings.TrimSpace(got) != "" {
		t.Fatalf("with no problem the banner drew:\n%s", got)
	}

	fixed := false
	sh.Report(app.Problem{
		Title:  "This model was converted by an older jitllm",
		Detail: "old.jlm is container format 17 and this build reads 21. Reconvert it from the file it came from.",
		Action: "Reconvert",
		Do:     func() { fixed = true },
	})
	sh.DrainQueue(0)
	got := strings.Join(strings.Fields(draw()), " ") // the detail wraps
	for _, want := range []string{"older jitllm", "Reconvert it from the file it came from.", "Reconvert", "Dismiss"} {
		if !strings.Contains(got, want) {
			t.Errorf("the banner does not show %q; drew:\n%s", want, got)
		}
	}
	if st.Status.Get() != "This model was converted by an older jitllm" {
		t.Errorf("the status line reads %q", st.Status.Get())
	}

	sessionProblemFix(st)
	if !fixed || st.Problem.Get().Seq != 0 {
		t.Errorf("the fix ran %v and the banner is still up %v", fixed, st.Problem.Get().Seq != 0)
	}
}
