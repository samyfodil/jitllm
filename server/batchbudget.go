package server

import "time"

// stepBudget is how many prompt tokens a step of the loop carries beside its
// decoding rows: vLLM's max_num_batched_tokens under chunked prefill, measured
// rather than set.
//
// A prompt admitted beside decoding rows costs each of them its chunk's time
// at their next token. Fed at the step's full width (a device or host chunk,
// 512 rows) that is a whole prefill chunk between two tokens, and the decoding
// rows' inter-token tail grows by it at every admission. So the loop feeds a
// prompt in pieces sized to cost about one decode step more: it times the
// steps it runs anyway -- a decode-only step gives a decode step's time, a
// step that carries prompt tokens the extra those tokens cost -- and scales the
// budget so the extra comes to one decode step. Under the compute crossover a
// prompt token is nearly free beside decoding rows (the step is bound by the
// weight read either way), so the budget grows, at most doubling a step; past
// it, it shrinks in proportion. Prompts are fed oldest first (promptUnits), so
// a long one still finishes, a budget a step.
//
// Config.StepPromptTokens fixes the budget instead. Only the loop goroutine
// touches it.
type stepBudget struct {
	fixed, ceiling int
	// cur is the measured budget.
	cur int
	// decode is a decode-only step's time, a moving average; zero until
	// measured.
	decode float64
}

const (
	// budgetStart is the budget before anything is measured: small, so the
	// first admission beside decoding rows is cheap.
	budgetStart = 32
	// budgetFloor keeps a prompt moving however slow a token measures.
	budgetFloor = 8
	// budgetAlpha weighs a new decode step in the moving average.
	budgetAlpha = 0.2
)

func newStepBudget(fixed, ceiling int) *stepBudget {
	ceiling = max(ceiling, 1)
	return &stepBudget{fixed: fixed, ceiling: ceiling, cur: min(budgetStart, ceiling)}
}

// tokens is the prompt tokens the next step beside decoding rows may carry.
func (b *stepBudget) tokens() int {
	if b.fixed > 0 {
		return b.fixed
	}
	return b.cur
}

// observe takes a step's time: the decoding rows and prompt tokens it carried.
func (b *stepBudget) observe(decoding, prompt int, d time.Duration) {
	if decoding == 0 || b.fixed > 0 {
		return
	}
	t := float64(d)
	if prompt == 0 {
		if b.decode == 0 {
			b.decode = t
		} else {
			b.decode += budgetAlpha * (t - b.decode)
		}
		return
	}
	if b.decode == 0 {
		return
	}
	// The prompt tokens that would have cost one decode step, at this
	// step's rate; a step the budget did not limit says nothing about more.
	extra := t - b.decode
	want := 2 * prompt
	if extra > 0 {
		want = min(want, int(float64(prompt)*b.decode/extra))
	}
	if prompt < b.cur && want >= prompt {
		return
	}
	b.cur = min(max(want, budgetFloor), b.ceiling)
}
