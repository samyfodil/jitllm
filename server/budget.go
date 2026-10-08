package server

import "github.com/samyfodil/jitllm/engine/sched"

// backgroundShare is the fraction of the host budget that all backgrounded
// models divide between them when the active model is given priority.
//
// A backgrounded model is not closed, so it still needs bytes: it keeps its
// pager and buys residency back by paging when next used. An eighth keeps the
// dense region and a few block pages.
const backgroundShare = 8

// shares divides the host weight budget between the open models.
//
// Something here has to divide it because nothing in the engine does:
// sched.MemBudget() is stateless and model.Open, tier.OpenWith and NewState
// each read it independently, so two models would each believe they hold the
// machine. The cgroup then throttles into reclaim, and neither pager can see
// the other model. (The vision tower avoided this by sharing one container
// and one budget; two models are two files and cannot.)
// pins are per-model caps the user set. They are honoured exactly and come off
// the top; the rest divide what is left.
//
// total is the budget, read once by the caller: sched.MemBudget follows the
// machine's free memory, so two reads of it are two different numbers, and a
// division checked against another moment's budget can exceed it.
func shares(total uint64, paths []string, active string, priority bool, pins map[string]uint64) map[string]uint64 {
	out := make(map[string]uint64, len(paths))
	if len(paths) == 0 {
		return out
	}

	// Pins first, so an unpinned model can never be handed bytes a pin has
	// already claimed.
	var free uint64 = total
	var unpinned []string
	for _, p := range paths {
		if n := pins[p]; n > 0 {
			if n > free {
				n = free
			}
			out[p] = n
			free -= n
			continue
		}
		unpinned = append(unpinned, p)
	}
	if len(unpinned) == 0 {
		return out
	}

	// Without priority the unpinned models split what is left evenly.
	if !priority {
		each := free / uint64(len(unpinned))
		for _, p := range unpinned {
			out[p] = each
		}
		return out
	}

	// Priority gives the active model the rest. Switching re-runs this, so the
	// bytes follow the model in use.
	bg := free / backgroundShare
	if len(unpinned) == 1 {
		out[unpinned[0]] = free
		return out
	}
	perBg := bg / uint64(len(unpinned)-1)
	var activeUnpinned bool
	for _, p := range unpinned {
		if p == active {
			activeUnpinned = true
			continue
		}
		out[p] = perBg
	}
	if activeUnpinned {
		out[active] = free - perBg*uint64(len(unpinned)-1)
		return out
	}
	// The active model is pinned, so the remainder goes back to the others.
	each := free / uint64(len(unpinned))
	for _, p := range unpinned {
		out[p] = each
	}
	return out
}

// hostTotal is the host budget every loaded model's share is divided from,
// read once: sched.MemBudget follows MemAvailable, so a second read after the
// weights are resident would count the engine's own footprint as taken.
// Guarded by e.mu.
func (e *Engine) hostTotal() uint64 {
	if e.total == 0 {
		e.total = sched.MemBudget()
	}
	return e.total
}

// sharesLocked divides the host budget between the loaded models, plus extra
// when a load is about to add one. e.mu held.
func (e *Engine) sharesLocked(extra string) map[string]uint64 {
	ids := append([]string(nil), e.order...)
	if extra != "" {
		ids = append(ids, extra)
	}
	pins := map[string]uint64{}
	for id, lm := range e.models {
		if lm.pin > 0 {
			pins[id] = lm.pin
		}
	}
	return shares(e.hostTotal(), ids, e.favored, e.priority, pins)
}

// rebudget re-divides the host budget and applies each model's share. A share
// is a function of the whole set, so it runs on every load, unload, pin and
// change of priority.
func (e *Engine) rebudget() {
	e.mu.Lock()
	sh := e.sharesLocked("")
	models := make([]*LoadedModel, 0, len(e.models))
	for id, lm := range e.models {
		lm.mu.Lock()
		lm.budget = sh[id]
		lm.mu.Unlock()
		models = append(models, lm)
	}
	e.mu.Unlock()
	for _, lm := range models {
		e.applyPageBudget(lm, nil)
	}
}

// Pin caps one model's host budget at bytes, taken off the top before the
// others divide the rest; 0 releases it back to the division.
func (e *Engine) Pin(id string, bytes uint64) error {
	lm, err := e.Model(id)
	if err != nil {
		return err
	}
	lm.mu.Lock()
	lm.pin = bytes
	lm.mu.Unlock()
	e.rebudget()
	return nil
}

// SetPriority gives the favoured model (Favor) everything but an eighth of the
// budget, the rest dividing that eighth; off, the models divide it evenly.
func (e *Engine) SetPriority(on bool) {
	e.mu.Lock()
	e.priority = on
	e.mu.Unlock()
	e.rebudget()
}

// Favor names the model in use, the one priority gives the budget to.
func (e *Engine) Favor(id string) {
	e.mu.Lock()
	e.favored = id
	e.mu.Unlock()
	e.rebudget()
}

// Budget is the total this model's pager is divided from now.
func (lm *LoadedModel) Budget() uint64 {
	lm.mu.Lock()
	defer lm.mu.Unlock()
	return lm.budget
}

// Pinned reports whether the model's budget is a pin rather than a share.
func (lm *LoadedModel) Pinned() bool {
	lm.mu.Lock()
	defer lm.mu.Unlock()
	return lm.pin > 0
}
