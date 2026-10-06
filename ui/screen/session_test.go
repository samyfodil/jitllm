package screen

import (
	"reflect"
	"strings"
	"testing"

	"github.com/gogpu/ui/event"
	"github.com/gogpu/ui/geometry"
	"github.com/gogpu/ui/state"
	"github.com/gogpu/ui/uitest"
	"github.com/gogpu/ui/widget"

	"github.com/samyfodil/jitllm/ui/app"
)

// ---------------------------------------------------------------------------
// ChatMessages
// ---------------------------------------------------------------------------

// The transcript is what the user sees and the message list is what the model
// sees; three kinds of row belong only to the first.
func TestChatMessagesDropsWhatTheModelNeverSaid(t *testing.T) {
	turns := []app.Turn{
		{Role: app.RoleUser, Text: "what is the capital of France?"},
		{Role: app.RoleAssistant, Text: "Paris."},
		{Role: app.RoleError, Text: "cuda: out of memory"},
		{Role: app.RoleSystem, Text: "the model was reloaded"},
		{Role: app.RoleUser, Text: "and of Germany?"},
		{Role: app.RoleAssistant, Text: ""}, // a generation stopped before a token
	}

	got := ChatMessages("be brief", turns, "and of Italy?")

	want := []ChatMessage{
		{Role: "system", Content: "be brief"},
		{Role: "user", Content: "what is the capital of France?"},
		{Role: "assistant", Content: "Paris."},
		// "and of Germany?" got no answer, so the model never had that exchange:
		// keeping it would put two user turns in a row.
		{Role: "user", Content: "and of Italy?"},
	}
	if len(got) != len(want) {
		t.Fatalf("got %d messages, want %d: %+v", len(got), len(want), got)
	}
	for i := range want {
		if !reflect.DeepEqual(got[i], want[i]) {
			t.Errorf("message %d = %+v, want %+v", i, got[i], want[i])
		}
	}

	// The violation this guards: a straight cast over the transcript would put
	// the engine's own failure in the model's mouth.
	for _, m := range got {
		if strings.Contains(m.Content, "out of memory") {
			t.Fatal("an engine error reached the model as an assistant turn")
		}
	}
}

// The system prompt is its own field. A system row in the transcript is a
// display artefact, and emitting both would send it twice.
func TestChatMessagesDoesNotEmitTheSystemPromptTwice(t *testing.T) {
	turns := []app.Turn{{Role: app.RoleSystem, Text: "be brief"}}
	got := ChatMessages("be brief", turns, "hello")

	n := 0
	for _, m := range got {
		if m.Role == "system" {
			n++
		}
	}
	if n != 1 {
		t.Fatalf("got %d system messages, want 1: %+v", n, got)
	}
}

// An empty system prompt emits no system turn at all, rather than an empty one
// the template would render as an instruction to say nothing.
func TestChatMessagesOmitsAnEmptySystemPrompt(t *testing.T) {
	got := ChatMessages("   ", nil, "hello")
	if len(got) != 1 || got[0].Role != "user" {
		t.Fatalf("got %+v, want exactly the user turn", got)
	}
}

// ---------------------------------------------------------------------------
// LiveText
// ---------------------------------------------------------------------------

func mockCtx() *uitest.MockContext {
	ctx := uitest.NewMockContext()
	ctx.SchedulerVal = state.NewScheduler(func([]widget.Widget) {})
	return ctx
}

// ---------------------------------------------------------------------------
// TailBox
// ---------------------------------------------------------------------------

// fixed is a child of a size the test chooses, so the anchoring can be checked
// without depending on the toolkit's character-width heuristic.
type fixed struct {
	widget.WidgetBase
	size geometry.Size
}

func (f *fixed) Layout(_ widget.Context, c geometry.Constraints) geometry.Size {
	s := c.Constrain(f.size)
	f.SetBounds(geometry.FromPointSize(geometry.Point{}, s))
	return s
}
func (f *fixed) Draw(_ widget.Context, _ widget.Canvas)     {}
func (f *fixed) Event(_ widget.Context, _ event.Event) bool { return false }
func (f *fixed) Children() []widget.Widget                  { return nil }

// A reply taller than the box shows its end.
func TestTailBoxShowsTheTailOfAnOverflowingChild(t *testing.T) {
	ctx := mockCtx()
	child := &fixed{size: geometry.Sz(300, 900)}
	child.SetVisible(true)
	box := NewTailBox(child, 260)

	size := box.Layout(ctx, geometry.Constraints{MaxWidth: 300, MaxHeight: 600})

	if size.Height != 260 {
		t.Fatalf("box height %v, want the 260 cap", size.Height)
	}
	if got := box.TailOffset(); got != 900-260 {
		t.Fatalf("tail offset %v, want %v -- the box is showing the TOP of the reply", got, 900-260)
	}
}

// A reply that fits is not scrolled at all, and the box is exactly as tall as
// it is -- so an empty stream costs no height and the strip disappears.
func TestTailBoxDoesNotScrollAChildThatFits(t *testing.T) {
	ctx := mockCtx()
	child := &fixed{size: geometry.Sz(300, 40)}
	child.SetVisible(true)
	box := NewTailBox(child, 260)

	size := box.Layout(ctx, geometry.Constraints{MaxWidth: 300, MaxHeight: 600})

	if size.Height != 40 {
		t.Fatalf("box height %v, want the child's own 40", size.Height)
	}
	if got := box.TailOffset(); got != 0 {
		t.Fatalf("tail offset %v, want 0", got)
	}
}

// ---------------------------------------------------------------------------
// the disabled rules
// ---------------------------------------------------------------------------

// Send is refused for three different reasons and each has to be live: the
// button greys itself out as the draft empties, without a per-keystroke
// rebuild.
func TestSessionSendDisabledTracksAllThreeReasons(t *testing.T) {
	st := app.NewStore()
	d := sessionSendDisabled(st)

	if !d.Get() {
		t.Fatal("send must be refused with no model loaded")
	}

	st.Loaded.Set(true)
	if !d.Get() {
		t.Fatal("send must be refused with an empty draft")
	}

	st.Draft.Set("   ")
	if !d.Get() {
		t.Fatal("whitespace is an empty draft")
	}

	st.Draft.Set("hello")
	if d.Get() {
		t.Fatal("send must be allowed with a model and a draft")
	}

	st.Busy.Set(true)
	if !d.Get() {
		t.Fatal("send must be refused while a generation is in flight")
	}
}

// Prefill produces no token and therefore no signal, so "reading the prompt"
// and "generating" must read differently or a multi-second prefill looks like
// an idle window.
func TestTheStatusNamesPrefillAndDecodeApart(t *testing.T) {
	sh := app.NewShell(nil, nil)
	st := sh.Store
	status := sessionStatus(sh)

	if got := status.Get().Label; got != "No model" {
		t.Fatalf("with nothing loaded the status reads %q", got)
	}
	st.Loaded.Set(true)
	idle := status.Get().Label

	st.Busy.Set(true)
	st.Streaming.Set(true)
	prefill := status.Get().Label
	if prefill == idle {
		t.Fatalf("a busy engine with no tokens yet reads like an idle one: %q", prefill)
	}

	st.DecodeTokS.Set(39.6)
	st.Stream.Set("Paris")
	if decode := status.Get().Label; decode == prefill || !strings.Contains(decode, "tok/s") {
		t.Fatalf("decode reads %q, prefill %q: they must differ, and decode carries its rate", decode, prefill)
	}
}

// After failed replies, what reaches the template still alternates user and
// assistant and ends on the user: the chat that broke Mistral -- a reply that
// came back empty, the question asked again, another empty reply -- and then a
// new message.
func TestChatMessagesAlternateAfterFailedReplies(t *testing.T) {
	turns := []app.Turn{
		{Role: app.RoleUser, Text: "what can i cook with rice and chicken"},
		{Role: app.RoleAssistant, Text: "Fried rice."},
		{Role: app.RoleUser, Text: "what is the best"},
		{Role: app.RoleAssistant, Text: ""},
		{Role: app.RoleUser, Text: "what is the best"},
		{Role: app.RoleAssistant, Text: "way to make it"},
		{Role: app.RoleUser, Text: "hi"},
		{Role: app.RoleError, Text: "Couldn't build the prompt"},
		{Role: app.RoleAssistant, Text: ""},
	}
	got := ChatMessages("be terse", turns, "hello")
	want := "user"
	for i, m := range got[1:] {
		if m.Role != want {
			t.Fatalf("message %d is %s where %s comes next: %+v", i+1, m.Role, want, got)
		}
		want = map[string]string{"user": "assistant", "assistant": "user"}[want]
	}
	if last := got[len(got)-1]; last.Role != "user" || last.Content != "hello" {
		t.Errorf("the conversation does not end on the new message: %+v", last)
	}
}
