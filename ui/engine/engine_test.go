package engine

import (
	"reflect"
	"testing"

	"github.com/samyfodil/jitllm/ui/app"
)

// Every engine value is one of the Store's own signals, so a write from the
// worker is what a screen reads. A field left out of StateOf is a nil the
// engine dereferences on its first load.
func TestEveryEngineValueIsTheStoresSignal(t *testing.T) {
	st := app.NewStore()
	s := reflect.ValueOf(StateOf(st))
	store := reflect.ValueOf(st).Elem()
	for i := range s.NumField() {
		name := s.Type().Field(i).Name
		got := s.Field(i)
		if got.IsNil() {
			t.Errorf("State.%s is nil", name)
			continue
		}
		want := store.FieldByName(name)
		if !want.IsValid() {
			t.Errorf("State.%s has no Store signal of that name", name)
			continue
		}
		if got.Elem().Interface() != want.Interface() {
			t.Errorf("State.%s is not Store.%s", name, name)
		}
	}
}

// The front's transcript writes reach the Store.
func TestTheFrontWritesTheTranscript(t *testing.T) {
	sh := app.NewShell(nil, nil)
	f := front{sh}
	i := sh.Store.AppendTurn(app.Turn{Role: app.RoleAssistant})
	f.SetTurn(i, app.Turn{Role: app.RoleAssistant, Text: "hello"})
	if got := sh.Store.Turn(i).Text; got != "hello" {
		t.Errorf("turn %d reads %q after SetTurn", i, got)
	}
	f.SetThinkOpen(i, true)
	if !sh.Store.ThinkOpen(i).Get() {
		t.Error("SetThinkOpen(true) left the turn's reasoning folded")
	}
}
