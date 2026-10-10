package server

import (
	"fmt"
	"math"
	"slices"
	"time"
)

// Fairness is the step loop's policy between throughput and the spread of
// the requests' waits, one level from 0 to 100 (Config.Fairness).
//
// At 0 the loop is first come first served: a request waits for a free row
// and the card's room for its history (kvRoom), a row runs until its
// generation ends, and the admitted prompts are fed oldest first. That is the
// most tokens a second, and under load the worst wait: a request behind a full
// card gets nothing until whole generations ahead of it finish (on a V100 at
// 64 concurrent, a first token minutes out while the rows ahead decoded at
// 12-16 ms a token).
//
// Above 0 three mechanisms engage, each monotonic in the level:
//
//   - Time slices. A decoding row that has generated its quantum of tokens
//     since it was admitted or resumed is parked when a request is waiting
//     for its row or its room (model.State.Park: its blocks' KV and recurrent
//     state home from the card, its sealed pages into a store), and the
//     waiting request is admitted in its place. The parked row waits in the
//     same queue and resumes byte for byte (State.Resume) when its turn comes;
//     its next token is sampled from the logits it held, so its answer is the
//     one it would have given unparked. The quantum shrinks with the level,
//     from fairQuantumMax tokens at 1 to fairQuantumMin at 100.
//   - Fair prompt feeding. The level's share of a step's prompt budget is
//     split evenly across every admitted prompt, the rest fed oldest first as
//     at 0, so a long prompt no longer holds every later one at zero.
//   - Ordering. The queue -- new requests and parked rows alike -- is served
//     by the time each began waiting (a parked row's is when it was parked),
//     a high-priority request (jitllm_priority) counted as having waited
//     fairBoost longer. Age dominates past that head start, so nothing
//     starves, and the head start shrinks as the level rises.
//
// A parked row's history is on the host while it waits, so a swap costs the
// bus its history twice; that is the throughput the level spends.
const (
	// DefaultFairness is Config.Fairness's default.
	DefaultFairness = 50
	// fairQuantumMax and fairQuantumMin are the time slice at level 1 and at
	// level 100, in generated tokens; between them it falls geometrically.
	fairQuantumMax = 256
	fairQuantumMin = 8
	// fairBoostMax is a high-priority request's head start at level 1; it
	// falls linearly to fairBoostMin at 100.
	fairBoostMax = 60 * time.Second
	fairBoostMin = time.Second
)

// fairLevel clamps a configured level; nil is the default.
func fairLevel(p *int) int {
	if p == nil {
		return DefaultFairness
	}
	return min(max(*p, 0), 100)
}

// fairQuantum is the time slice at level f, in tokens; 0 is none.
func fairQuantum(f int) int {
	if f <= 0 {
		return 0
	}
	t := float64(f-1) / 99
	return int(math.Round(fairQuantumMax * math.Pow(float64(fairQuantumMin)/fairQuantumMax, t)))
}

// fairBoost is the head start a priority level is worth at level f.
func fairBoost(f int) time.Duration {
	if f <= 0 {
		return 0
	}
	t := float64(f-1) / 99
	return fairBoostMax - time.Duration(t*float64(fairBoostMax-fairBoostMin))
}

// rowPriority is a request's priority in the queue: 1 for PriorityHigh.
func rowPriority(p string) int {
	if p == PriorityHigh {
		return 1
	}
	return 0
}

// fairShare is the prompt tokens of a budget the level splits evenly.
func fairShare(f, budget int) int { return budget * f / 100 }

// key is when r counts as having begun to wait, its priority's head start
// taken off: the queue is served smallest first.
func (lp *stepLoop) key(r *row) time.Time {
	return r.since.Add(-time.Duration(r.prio) * fairBoost(lp.fair))
}

// order sorts the queue for admission. lp.mu held. At level 0 the queue
// stays in arrival order.
func (lp *stepLoop) order() {
	if lp.fair <= 0 {
		return
	}
	slices.SortStableFunc(lp.waiting, func(a, b *row) int { return lp.key(a).Compare(lp.key(b)) })
}

// victims is the decoding rows to park so that r is admitted: those past
// their quantum, the lowest priority and longest served first, as few as
// free a row and room for need positions. It is nil when they cannot, so a
// swap that would not admit r parks nobody. lp.mu held.
func (lp *stepLoop) victims(r *row, need, room, page int) []*row {
	q := fairQuantum(lp.fair)
	if q == 0 {
		return nil
	}
	var cand []*row
	for _, x := range lp.rows {
		if !x.prompting() && !x.cancelled.Load() && x.slice >= q && x.n < x.maxTokens {
			cand = append(cand, x)
		}
	}
	slices.SortStableFunc(cand, func(a, b *row) int {
		if a.prio != b.prio {
			return a.prio - b.prio
		}
		return b.slice - a.slice
	})
	rows := len(lp.rows)
	for i, x := range cand {
		rows--
		room += rowPages(x, 0, page) * page
		if rows < lp.width && (page == 0 || rows == 0 || need <= room || lp.admitAll) {
			return cand[:i+1]
		}
	}
	return nil
}

// parkRow takes a decoding row out of the step and parks its session, the
// row going back into the queue. lp.mu held.
func (lp *stepLoop) parkRow(r *row) error {
	// The next token is sampled from these logits after the resume; the
	// State's buffer behind them may be reused meanwhile.
	r.held = append(r.held[:0], r.logits...)
	r.logits = r.held
	if lp.parkDropsLogits {
		clear(r.held)
	}
	before := lp.room()
	if err := lp.e.parkLocked(r.s); err != nil {
		return err
	}
	lp.stats.parkFreed.Add(int64(lp.room() - before))
	for i, x := range lp.rows {
		if x == r {
			lp.rows = append(lp.rows[:i], lp.rows[i+1:]...)
			break
		}
	}
	r.parked = true
	r.since = time.Now()
	lp.waiting = append(lp.waiting, r)
	lp.shape++
	lp.stats.parks.Add(1)
	return nil
}

// room is the card's room for history in positions (tier.GPU.KVRoom), 0 on
// the host.
func (lp *stepLoop) room() int {
	if lp.lm.gpu == nil {
		return 0
	}
	n, _ := lp.lm.gpu.KVRoom()
	return n
}

// resumeRow brings a parked row back into the step. lp.mu held.
func (lp *stepLoop) resumeRow(r *row) error {
	if err := lp.e.resumeLocked(r.s); err != nil {
		return fmt.Errorf("server: resuming a time-sliced row: %w", err)
	}
	r.parked = false
	r.resumedAt = r.n
	lp.stats.resumes.Add(1)
	return nil
}

// dropParked retires the parked rows whose requests left. Their sessions stay
// parked, and the next generate on one resumes it (admitKV). lp.mu held.
func (lp *stepLoop) dropParked() {
	for i := 0; i < len(lp.waiting); i++ {
		r := lp.waiting[i]
		if !r.parked || !r.cancelled.Load() {
			continue
		}
		lp.waiting = append(lp.waiting[:i], lp.waiting[i+1:]...)
		i--
		lp.mu.Unlock()
		lp.finish(r, FinishCancelled, "", nil)
		lp.mu.Lock()
	}
}
