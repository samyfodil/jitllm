package tier

import "slices"

// Submissions in flight.
//
// A session's submission runs outside g.mu (layersOnce), so another session's
// bookkeeping goes on beside it. Most of that bookkeeping touches only what
// its own session reads -- its pages, its table run, its lane -- and needs
// nothing more. What changes something every submission may read -- a buffer
// a recording names, a block's weights, a page another sequence's descriptor
// points at, a recurrent pool's slots -- waits for every submission in flight
// to complete first (quiesce), and holds g.mu while it changes it, so none
// starts until it is done. A submission whose range writes what is the
// device's rather than the session's (a vision tower's set and k/v pair, a
// streamed block's expert cache, a recurrent pool, the rows of several
// sessions) runs alone: it quiesces and keeps g.mu across its Session, which
// is the whole device as it was before sessions ran at once
// (exclusiveRange).
//
// A KV page released while submissions are in flight is fenced with the
// ticket the next submission will take (subClock.seq), and is reused only
// once every submission holding a ticket up to it has completed -- or at once
// when none is in flight -- so a page a running submission might still read
// is never handed to another sequence under it (kvLayerPool.fence).

// subClock is the device's submission tickets: seq is the next one to be
// taken, live how many are in flight and oldest the smallest of them (seq
// when none is).
type subClock struct {
	seq, oldest uint64
	live        int
}

// beginSub takes a ticket for a submission about to run outside g.mu, after
// any quiesce in progress has had its turn. Callers hold g.mu.
func (g *devTier) beginSub() uint64 {
	for g.quiet > 0 {
		g.idle.Wait()
	}
	t := g.clk.seq
	g.clk.seq++
	g.tickets = append(g.tickets, t)
	g.clk.live = len(g.tickets)
	g.clk.oldest = g.tickets[0]
	return t
}

// endSub gives ticket t back. Callers hold g.mu.
func (g *devTier) endSub(t uint64) {
	if i := slices.Index(g.tickets, t); i >= 0 {
		g.tickets = slices.Delete(g.tickets, i, i+1)
	}
	g.clk.live = len(g.tickets)
	g.clk.oldest = g.clk.seq
	if len(g.tickets) > 0 {
		g.clk.oldest = slices.Min(g.tickets)
	}
	if len(g.tickets) == 0 {
		g.idle.Broadcast()
	}
}

// quiesce waits until no submission is in flight, and keeps new ones from
// starting while it waits; the caller then changes what they read before it
// lets g.mu go. It must not be called from inside a submission -- it would
// wait for itself. Callers hold g.mu.
func (g *devTier) quiesce() {
	if g.inSub {
		panic("tier: quiesce from inside a submission")
	}
	if len(g.tickets) == 0 {
		return
	}
	g.quiet++
	for len(g.tickets) > 0 {
		g.idle.Wait()
	}
	if g.quiet--; g.quiet == 0 {
		g.idle.Broadcast()
	}
}
