package server

import (
	"context"
	"path/filepath"
	"runtime"
	"slices"
	"testing"
	"time"
)

// TestPreemptionKeepsEveryAnswer: a KV budget that fits two sessions' histories,
// three sessions generating in turn, each continuing its own sequence three
// times. Every turn past the second parks the least recently stepped other
// session, and every session's tokens equal its solo run's, generated alone
// with no budget. Parks, resumes and pages out and in are counted, the pages
// in equal the pages out, and no goroutine outlives the engine.
func TestPreemptionKeepsEveryAnswer(t *testing.T) {
	base := runtime.NumGoroutine()
	const rounds, perTurn, maxSeq = 5, 70, 512
	prompts := map[string]string{
		"a": "Once upon a time",
		"b": "The little dog",
		"c": "One day a girl",
	}
	order := []string{"a", "b", "c"}
	ctx := context.Background()

	turn := func(e *Engine, id string, cont bool) []int32 {
		t.Helper()
		var out []int32
		p := Prompt{Kind: PromptText, Text: prompts[id]}
		if cont {
			// A continue feeds one more token, the same in both runs.
			p = Prompt{Kind: PromptIDs, IDs: []int32{13}}
		}
		err := e.Generate(ctx, GenerateOptions{SessionID: id, Prompt: p, MaxTokens: perTurn,
			Continue: cont, IgnoreEOS: true}, func(ev Event) error {
			if ev.Kind == EventToken && ev.Token.ID >= 0 {
				out = append(out, ev.Token.ID)
			}
			return nil
		})
		if err != nil {
			t.Fatalf("%s: %v", id, err)
		}
		return out
	}

	run := func(budgeted bool, only string) map[string][]int32 {
		path := modelPath(t, smallModel)
		e := New(Config{ModelDir: filepath.Dir(path), Probe: oneCardProbe, Version: "test"})
		defer e.Close()
		if _, err := e.LoadModel(LoadOptions{Path: path, ModelID: "small"}); err != nil {
			t.Fatal(err)
		}
		got := map[string][]int32{}
		ids := order
		if only != "" {
			ids = []string{only}
		}
		for _, id := range ids {
			if _, err := e.CreateSession(SessionOptions{ModelID: "small", SessionID: id, MaxSeq: maxSeq}); err != nil {
				t.Fatal(err)
			}
		}
		if budgeted {
			// Two sessions' full histories, measured on a probe State.
			st := e.sessions["a"].st
			per := st.PositionBytes()
			if per == 0 {
				t.Fatal("a position costs no bytes: the budget would be nothing")
			}
			if err := e.SetKVBudget("small", 2*uint64(rounds*(perTurn+8))*per); err != nil {
				t.Fatal(err)
			}
		}
		for r := range rounds {
			for _, id := range ids {
				got[id] = append(got[id], turn(e, id, r > 0)...)
			}
		}
		if budgeted {
			parks, resumes, out, in := e.PreemptCounts()
			t.Logf("parks %d, resumes %d, pages out %d, in %d", parks, resumes, out, in)
			if parks == 0 || resumes == 0 || out == 0 || in == 0 {
				t.Fatalf("parks %d, resumes %d, pages out %d, in %d: the budget never made a page go and come back",
					parks, resumes, out, in)
			}
			// The sessions still parked hold the pages not yet back.
			var held int64
			for _, id := range ids {
				s := e.sessions[id]
				if s.Parked() {
					held += s.st.ParkedOut - s.st.ParkedIn
				}
			}
			if in+held != out {
				t.Fatalf("%d pages out, %d in and %d still parked", out, in, held)
			}
		}
		return got
	}

	got := run(true, "")
	for _, id := range order {
		want := run(false, id)[id]
		if !slices.Equal(got[id], want) {
			t.Fatalf("session %s under preemption: %v\nsolo: %v", id, got[id], want)
		}
		if len(want) < rounds*perTurn/2 {
			t.Fatalf("session %s generated %d tokens: too few to have sealed a page", id, len(want))
		}
	}
	deadline := time.Now().Add(5 * time.Second)
	for runtime.NumGoroutine() > base && time.Now().Before(deadline) {
		time.Sleep(20 * time.Millisecond)
	}
	if n := runtime.NumGoroutine(); n > base {
		t.Fatalf("%d goroutines after four engines closed, %d before", n, base)
	}
}
