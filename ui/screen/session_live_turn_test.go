package screen

import (
	"testing"

	"github.com/jitllm/jitllm/ui/app"
)

// Only the last assistant turn is live, because only that one is being
// filled. A non-live row renders a frozen string, so marking the wrong row
// shows one turn's stream under another's header, and marking none shows no
// reply at all.
func TestOnlyTheLastAssistantTurnIsLive(t *testing.T) {
	st := app.NewStore()
	st.AppendTurn(app.Turn{Role: app.RoleUser, Text: "hi"})
	st.AppendTurn(app.Turn{Role: app.RoleAssistant, Text: "done"})
	st.AppendTurn(app.Turn{Role: app.RoleUser, Text: "again"})
	reply := st.AppendTurn(app.Turn{Role: app.RoleAssistant})

	for i, want := range []bool{false, false, false, true} {
		if got := isLiveTurn(st, i); got != want {
			t.Errorf("isLiveTurn(%d) = %v, want %v", i, got, want)
		}
	}
	if !isLiveTurn(st, reply) {
		t.Error("the turn the engine is about to fill is not live")
	}
	// A user turn last -- nothing is being generated, so nothing is live.
	st.AppendTurn(app.Turn{Role: app.RoleUser, Text: "and again"})
	if isLiveTurn(st, reply) {
		t.Error("the finished reply is still live after a new user turn")
	}
	// Out of range must not panic or claim liveness.
	if isLiveTurn(st, 99) || isLiveTurn(st, -1) {
		t.Error("an index the transcript does not have reads as live")
	}
}
