package server

import (
	"context"
	"testing"

	"github.com/samyfodil/jitllm/engine/model"
)

// Two models loaded into one engine divide its host budget: neither may
// believe it holds the machine. A pin comes off the top, priority gives the
// favoured model all but an eighth, and an unload hands its share back.
func TestLoadedModelsDivideTheHostBudget(t *testing.T) {
	e, a, _ := loadedEngine(t, smallModel, "a", LoadOptions{})
	b, err := e.LoadModel(LoadOptions{Path: a.path, ModelID: "b"})
	if err != nil {
		t.Fatal(err)
	}
	total := e.hostTotal()
	want := func(what string, wa, wb uint64) {
		t.Helper()
		if a.Budget() != wa || b.Budget() != wb {
			t.Fatalf("%s: a=%d b=%d, want %d and %d of %d", what, a.Budget(), b.Budget(), wa, wb, total)
		}
	}
	want("two models", total/2, total/2)

	if err := e.Pin("a", total/4); err != nil {
		t.Fatal(err)
	}
	want("a pinned", total/4, total-total/4)
	if !a.Pinned() || b.Pinned() {
		t.Fatal("the pin is on the wrong model")
	}
	if err := e.Pin("a", 0); err != nil {
		t.Fatal(err)
	}
	want("pin released", total/2, total/2)

	e.Favor("b")
	e.SetPriority(true)
	bg := total / backgroundShare
	want("b favoured", bg, total-bg)

	if _, err := e.UnloadModel("b", true); err != nil {
		t.Fatal(err)
	}
	if a.Budget() != total {
		t.Fatalf("after b unloaded a holds %d, want all %d", a.Budget(), total)
	}
}

// A session with a prompt store resumes a prompt it has seen: the second
// generate of the same prompt restores its prefix instead of prefilling it.
func TestASessionWithAStoreReusesItsPrompt(t *testing.T) {
	e, _, _ := loadedEngine(t, smallModel, "small", LoadOptions{})
	s, err := e.CreateSession(SessionOptions{ModelID: "small", KVStore: model.NewMemStore(), CacheKey: "test"})
	if err != nil {
		t.Fatal(err)
	}
	run := func() int {
		o := GenerateOptions{SessionID: s.ID(), Prompt: Prompt{Kind: PromptText, Text: story + story + story}, MaxTokens: 2}
		if err := e.Generate(context.Background(), o, func(Event) error { return nil }); err != nil {
			t.Fatal(err)
		}
		var n int
		s.Inspect(func(st *model.State) { n = st.KVRestored() })
		return n
	}
	if n := run(); n != 0 {
		t.Fatalf("the first generate restored %d positions from an empty store", n)
	}
	if n := run(); n == 0 {
		t.Fatal("the second generate of the same prompt restored nothing: the store was not used")
	}
}
