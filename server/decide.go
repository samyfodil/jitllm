package server

import (
	"context"
	"fmt"
	"time"

	"github.com/samyfodil/jitllm/engine/model"
	"github.com/samyfodil/jitllm/format/jlm"
	"github.com/samyfodil/jitllm/tok/jinja"
)

// Decisions: TypeSafe's System One request, answered by a decision model
// (docs/design/decision-models.md). /v1/systemone calls Engine.Decide and
// nothing else, so an answer served here is the answer `jitllm decide` prints
// for the same model and request.
//
// A decision runs in one turn on the model's gates (its devices' and the
// host's, as a generate does), like an embedding: its prompts are prefilled
// on the model's Decider, one State the model keeps and reuses, its blocks
// placed on the model's device as a session's are (decideMaxBlocks), and a
// second request for the same model waits for the first.

// DecideOptions is Decide's input.
type DecideOptions struct {
	// Model is a loaded model's id, or the file name it was loaded from.
	Model string
	// State is what every question is about: a string or any JSON value,
	// object keys in the order the request wrote them.
	State jinja.Value
	// Questions are the request's questions in its order.
	Questions []model.DecisionQuestion
}

// DecideResult is one answer per question, in question order.
type DecideResult struct {
	ModelID     string
	Answers     []model.DecisionAnswer
	InputTokens int
	QueuedFor   time.Duration
	Took        time.Duration
}

// Decide answers o's questions with a decision model.
func (e *Engine) Decide(ctx context.Context, o DecideOptions) (*DecideResult, error) {
	lm, err := e.resolveModel(o.Model)
	if err != nil {
		return nil, err
	}
	if lm.m.Decision() == jlm.DecisionNone {
		return nil, fmt.Errorf("%w: model %q (%s) is not a decision model: its container states no "+
			"decision readout, so no score of its forward pass is an answer the weights were trained for",
			ErrInvalid, lm.id, lm.m.Cfg.Arch)
	}
	gs := e.gatesFor(lm.gateIDs())
	t0 := time.Now()
	gs.acquire(e.nextID("dec"))
	defer gs.release()
	queued := time.Since(t0)

	lm.decMu.Lock()
	defer lm.decMu.Unlock()
	if lm.decClosed {
		return nil, fmt.Errorf("%w: model %q was unloaded", ErrNotFound, lm.id)
	}
	if lm.dec == nil {
		d, err := lm.m.NewDecider(e.defaultMaxSeq(lm))
		if err != nil {
			return nil, err
		}
		if lm.dev != nil {
			if err := d.SetDevice(lm.dev, lm.deviceBlocks()); err != nil {
				d.Close()
				return nil, err
			}
		}
		lm.dec = d
	}
	if err := lm.dec.Prepare(o.State, o.Questions); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalid, err)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	t1 := time.Now()
	ans, err := lm.dec.Decide(o.State, o.Questions)
	if err != nil {
		return nil, err
	}
	n := lm.dec.InputTokens()
	lm.tokensPrefilled.Add(int64(n))
	return &DecideResult{ModelID: lm.id, Answers: ans, InputTokens: n, QueuedFor: queued, Took: time.Since(t1)}, nil
}

// deviceBlocks is how many blocks a State, Decider or Embedder of this model
// offers its device: the load's cap, or every block (-1).
func (lm *LoadedModel) deviceBlocks() int {
	if lm.maxBlocks != 0 {
		return lm.maxBlocks
	}
	return -1
}

// closeDecider closes the model's Decider; a decision in flight holds decMu
// and finishes first.
func (lm *LoadedModel) closeDecider() {
	lm.decMu.Lock()
	defer lm.decMu.Unlock()
	lm.decClosed = true
	if lm.dec != nil {
		lm.dec.Close()
		lm.dec = nil
	}
}
