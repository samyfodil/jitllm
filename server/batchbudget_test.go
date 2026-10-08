package server

import (
	"strings"
	"sync"
	"testing"
	"time"

	v1 "github.com/samyfodil/jitllm/server/gen/jitllm/v1"
)

// TestStepBudgetFollowsTheMeasurement drives the prompt budget with step
// times it is told: it starts small; prompt tokens that cost nothing beside
// a 10 ms decode step double it a step up to the ceiling; at 0.5 ms a token
// it settles at the 20 tokens that cost one decode step; slower tokens take
// it to its floor; a step the budget did not limit does not raise it; and a
// fixed budget ignores the clock. Against a budget that never reads its
// measurements, every case after the first reads budgetStart.
func TestStepBudgetFollowsTheMeasurement(t *testing.T) {
	const ms = time.Millisecond
	b := newStepBudget(0, 512)
	if got := b.tokens(); got != budgetStart {
		t.Fatalf("unmeasured budget %d, want %d", got, budgetStart)
	}
	b.observe(4, 0, 10*ms)
	var seen []int
	for range 6 {
		b.observe(4, b.tokens(), 10*ms) // prompt tokens free
		seen = append(seen, b.tokens())
	}
	if seen[0] != 2*budgetStart || seen[len(seen)-1] != 512 {
		t.Fatalf("free prompt tokens grew the budget %v, want doubling to 512", seen)
	}
	for range 4 {
		p := b.tokens()
		b.observe(4, p, 10*ms+time.Duration(p)*ms/2) // 0.5 ms a token
	}
	if got := b.tokens(); got != 20 {
		t.Fatalf("budget %d, want 20 (10 ms / 0.5 ms)", got)
	}
	b.observe(4, 5, 10*ms) // under the budget: says nothing about more
	if got := b.tokens(); got != 20 {
		t.Fatalf("an unlimited step moved the budget to %d", got)
	}
	b.observe(4, 20, 10*ms+2000*ms)
	if got := b.tokens(); got != budgetFloor {
		t.Fatalf("budget %d, want the floor %d", got, budgetFloor)
	}
	if got := newStepBudget(16, 512).tokens(); got != 16 {
		t.Fatalf("a fixed budget reads %d", got)
	}
}

// TestHostBatchPromptBesideDecodeIsBounded: a long prompt admitted beside two
// decoding rows is fed at most StepPromptTokens tokens a step while they
// decode -- counted, the most prompt tokens any step carried beside a
// decoding row -- it still finishes, and every row comes out as it does
// alone. Then the measured budget (StepPromptTokens 0) on the same requests:
// bounded by the chunk, rows equal to alone, the budget it settled on logged.
// The violation lifts the bound (the budget fixed at the prompt chunk): a
// step then carries more than the bound beside decoding rows.
func TestHostBatchPromptBesideDecodeIsBounded(t *testing.T) {
	const bound = 16
	e, lm, c := hostBatchEngine(t, "Llama-3.2-1B-Instruct-Q4_K_M.jlm",
		Config{JointSteps: JointAlways, PromptChunk: 64, StepPromptTokens: bound, DefaultMaxSeq: 1024})
	long := strings.Repeat("The little dog ran to the park and played with a red ball. ", 12)
	reqs := []*v1.GenerateRequest{
		{ModelId: "dev", Prompt: text(hostPrompts[0]), MaxTokens: 40},
		{ModelId: "dev", Prompt: text(hostPrompts[1]), MaxTokens: 40},
		{ModelId: "dev", Prompt: text(long), MaxTokens: 8},
	}
	prompts := []string{hostPrompts[0], hostPrompts[1], long}
	alone := make([]completion, len(reqs))
	for i, r := range reqs {
		if alone[i] = complete(c, r); alone[i].err != nil {
			t.Fatal(alone[i].err)
		}
	}
	// The two short prompts are admitted first and decode; the long one
	// arrives once they do.
	run := func() ([]completion, int64) {
		lm.loop.stats.maxPromptBeside.Store(0)
		got := make([]completion, len(reqs))
		release := holdGates(t, e, lm)
		var wg sync.WaitGroup
		for i := range 2 {
			wg.Add(1)
			go func() { defer wg.Done(); got[i] = complete(c, reqs[i]) }()
		}
		waitFor(t, "the short requests to wait", func() bool { return waitingFor(lm.loop) == 2 })
		release()
		waitFor(t, "the short rows to decode", func() bool { return lm.loop.stats.steps.Load() > 0 && waitingFor(lm.loop) == 0 })
		got[2] = complete(c, reqs[2])
		wg.Wait()
		for i := range got {
			if got[i].err != nil {
				t.Fatal(got[i].err)
			}
			if j := parted(t, lm, prompts[i], got[i].ids, alone[i].ids); j >= 0 {
				t.Fatalf("row %d parts from alone at token %d\nbatched %v\nalone   %v", i, j, got[i].ids, alone[i].ids)
			}
		}
		return got, lm.loop.stats.maxPromptBeside.Load()
	}
	_, most := run()
	t.Logf("fixed bound %d: at most %d prompt tokens beside decoding rows", bound, most)
	if most == 0 || most > bound {
		t.Fatalf("%d prompt tokens in one step beside decoding rows, want 1..%d", most, bound)
	}

	lm.loop.budget = newStepBudget(0, 64)
	_, most = run()
	t.Logf("measured: at most %d prompt tokens beside decoding rows; budget now %d (decode step %.1f ms)",
		most, lm.loop.budget.tokens(), lm.loop.budget.decode/1e6)
	if most == 0 || most > 64 {
		t.Fatalf("measured budget: %d prompt tokens beside decoding rows", most)
	}

	lm.loop.budget = newStepBudget(64, 64)
	_, most = run()
	if most <= bound {
		t.Fatalf("the bound lifted to 64, a step still carried at most %d beside decoding rows: "+
			"this gate cannot see the bound", most)
	}
	t.Logf("violation: the bound lifted, %d prompt tokens beside decoding rows", most)
}
