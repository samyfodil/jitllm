package engine

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
