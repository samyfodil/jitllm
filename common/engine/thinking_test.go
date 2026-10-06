package engine

import (
	"strings"
	"testing"
)

// The reasoning and the answer must come apart cleanly.
//
// They are split rather than marked because they are shown differently: the
// transcript collapses the reasoning so it does not push the answer away.
func TestTheReasoningComesApartFromTheAnswer(t *testing.T) {
	for _, c := range []struct{ name, in, think, answer string }{
		{"closed", "<think>weighing it up</think>\nthe answer", "weighing it up", "the answer"},
		{"still open", "<think>still weighing", "still weighing", ""},
		{"no block", "just an answer", "", "just an answer"},
		{"text before the block", "hi <think>hmm</think>there", "hmm", "hi there"},
		{"two blocks", "<think>a</think>one<think>b</think>two", "a\nb", "onetwo"},
		{"empty", "", "", ""},
	} {
		t.Run(c.name, func(t *testing.T) {
			think, answer := SplitThinking(c.in)
			if think != c.think {
				t.Errorf("think = %q, want %q", think, c.think)
			}
			if answer != c.answer {
				t.Errorf("answer = %q, want %q", answer, c.answer)
			}
		})
	}
}

// While the block is still open there is no answer, and the reasoning is all
// there is.
//
// That is the true state, not a placeholder: no reply exists yet.
func TestAnUnclosedBlockIsAllThinkingAndNoAnswer(t *testing.T) {
	const monologue = "Okay, the user is asking how to build an http server"
	think, answer := SplitThinking("<think>" + monologue)
	if think != monologue {
		t.Errorf("think = %q, want the monologue", think)
	}
	if answer != "" {
		t.Errorf("answer = %q, want empty: there is no answer yet", answer)
	}
}

// No prefix may leak a half-typed tag into either half.
func TestNoPrefixLeaksATagFragment(t *testing.T) {
	const full = "<think>reasoning here</think>the answer"
	for i := 1; i <= len(full); i++ {
		think, answer := SplitThinking(holdPartialTag(full[:i]))
		for _, s := range []string{think, answer} {
			if strings.ContainsAny(s, "<>") {
				t.Fatalf("prefix %d (%q) leaked a tag fragment: think=%q answer=%q",
					i, full[:i], think, answer)
			}
		}
	}
}

// The answer must never contain the reasoning, at any prefix.
//
// This is the property the transcript depends on: the answer is shown
// expanded and the reasoning collapsed.
func TestTheAnswerNeverCarriesTheReasoning(t *testing.T) {
	const full = "<think>SECRETREASONING</think>the visible answer"
	for i := 1; i <= len(full); i++ {
		_, answer := SplitThinking(holdPartialTag(full[:i]))
		if strings.Contains(answer, "SECRET") {
			t.Fatalf("prefix %d put the reasoning in the answer: %q", i, answer)
		}
	}
}

// A block the template opened has only a closing tag, and everything before it
// is reasoning.
//
// A template that pre-opens the block makes the output start inside it with
// only </think> to find; a parser that looked for the opener first would call
// the whole monologue the answer.
func TestATemplateOpenedBlockIsStillReasoning(t *testing.T) {
	for _, c := range []struct{ name, in, think, answer string }{
		{"closed only", "weighing it up</think>the answer", "weighing it up", "the answer"},
		{"closed only, still open", "weighing it up", "", "weighing it up"},
		{"a real opener later still works",
			"reasoning</think>answer<think>more</think>tail",
			"reasoning\nmore", "answertail"},
		{"an ordinary pair is unaffected",
			"<think>a</think>b", "a", "b"},
	} {
		t.Run(c.name, func(t *testing.T) {
			think, answer := SplitThinking(c.in)
			if think != c.think {
				t.Errorf("think = %q, want %q", think, c.think)
			}
			if answer != c.answer {
				t.Errorf("answer = %q, want %q", answer, c.answer)
			}
		})
	}
}

// pipeline is the exact pair of lines the generate loop runs on every token
// and once at the end (see Engine.generate).
func pipeline(rawDecode string, preOpened bool) (think, answer string) {
	return splitThinking(holdPartialTag(display(rawDecode)), preOpened)
}

// TestThePipelineKeepsTheReasoning is the gate on the whole path from decoded
// tokens to the two strings the transcript renders.
//
// Every other test here calls SplitThinking directly; the engine does not, and
// display() once split and discarded the reasoning before the caller saw it.
func TestThePipelineKeepsTheReasoning(t *testing.T) {
	for _, c := range []struct {
		name      string
		raw       string
		preOpened bool
		think     string
		answer    string
	}{{
		name:   "the model emits both tags",
		raw:    "<think>Okay, the user wants a capital.</think>Paris.",
		think:  "Okay, the user wants a capital.",
		answer: "Paris.",
	}, {
		name:      "the template opened the block, so only the closer arrives",
		raw:       "Okay, the user wants a capital.</think>Paris.",
		preOpened: true,
		think:     "Okay, the user wants a capital.",
		answer:    "Paris.",
	}, {
		name:   "still inside the block: reasoning grows, no answer yet",
		raw:    "<think>Okay, the user wants",
		think:  "Okay, the user wants",
		answer: "",
	}, {
		name:   "a model that does not reason has no reasoning",
		raw:    "Paris.",
		answer: "Paris.",
	}} {
		t.Run(c.name, func(t *testing.T) {
			think, answer := pipeline(c.raw, c.preOpened)
			if think != c.think {
				t.Errorf("think = %q, want %q", think, c.think)
			}
			if answer != c.answer {
				t.Errorf("answer = %q, want %q", answer, c.answer)
			}
		})
	}
}

// TestDisplayOnlyHoldsAPartialRune pins display() to the one job its doc
// describes. A second responsibility in it is what deleted the reasoning.
func TestDisplayOnlyHoldsAPartialRune(t *testing.T) {
	if got := display("<think>reasoning</think>answer"); got != "<think>reasoning</think>answer" {
		t.Fatalf("display rewrote its input: %q", got)
	}
	// "é" is two bytes; half of it must be held back and nothing else cut.
	if got := display("caf\xc3"); got != "caf" {
		t.Fatalf("display(%q) = %q, want %q", "caf\xc3", got, "caf")
	}
}
