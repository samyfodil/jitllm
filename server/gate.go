package server

import (
	"context"
	"errors"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

// HostGateID is the gate every host-only session queues on. The host is a
// placement target like any other, so it gets a gate like any other.
const HostGateID = "cpu"

// gate serialises sessions onto one device (or the host).
//
// The tier holds one scratch set per device (g.bs, g.vbs, g.bbs in
// jit/gpu/tier), so two sessions on one card are correct but cannot run at
// once. A session holds its devices' gates for a whole generate; the wait is
// timed and reported in GenerateStarted.queued_millis, and the depth is
// readable through GetDeviceQueue.
//
// A model's step loop (batch.go) is one holder like any other: it holds its
// model's gates while it has rows, so every generate on a session wholly on the
// device shares the loop's turn instead of queueing for one of its own.
//
// The host has a gate too: every nn.JIT runs its own pool of decode cores, so
// concurrent host sessions oversubscribe them. Its width is
// Config.HostConcurrency, default 1.
//
// Nothing is refused for being second: a session waits until admitted, and
// only a per-request queue timeout or a cancelled context ends the wait.
type gate struct {
	id   string
	sem  chan struct{}
	mode ExecutionMode
	note string

	waiting atomic.Int32
	// holders is the sessions currently inside, and queue is every session
	// that has asked, in arrival order. Both are reported, not inferred.
	mu      sync.Mutex
	holders map[string]bool
	queue   []string
}

// ExecutionMode mirrors the proto enum without the server package depending on
// generated code for its own internals.
type ExecutionMode int

const (
	ExecutionUnspecified ExecutionMode = iota
	ExecutionParallel
	ExecutionSerialised
)

// ErrQueueTimeout is returned when a caller declined to keep waiting. It is
// NOT a refusal of the session: the session stays open and may try again.
var ErrQueueTimeout = errors.New("server: timed out waiting for the device's scratch")

func newGate(id string, width int, note string) *gate {
	if width < 1 {
		width = 1
	}
	mode := ExecutionSerialised
	if width > 1 {
		mode = ExecutionParallel
	}
	g := &gate{
		id:      id,
		sem:     make(chan struct{}, width),
		mode:    mode,
		note:    note,
		holders: map[string]bool{},
	}
	return g
}

// acquire blocks until the gate admits this session. It returns how long the
// wait was and how many sessions were ahead on entry, because those two
// numbers are the whole of what a caller cannot see from the outside.
func (g *gate) acquire(ctx context.Context, sessionID string, timeout time.Duration) (waited time.Duration, depth int32, err error) {
	depth = g.waiting.Add(1)
	g.mu.Lock()
	g.queue = append(g.queue, sessionID)
	g.mu.Unlock()

	start := time.Now()
	defer func() {
		g.waiting.Add(-1)
		if err != nil {
			g.dequeue(sessionID)
		}
	}()

	// The fast path costs no timer.
	select {
	case g.sem <- struct{}{}:
		g.enter(sessionID)
		return 0, depth - 1, nil
	default:
	}

	var timer <-chan time.Time
	if timeout > 0 {
		t := time.NewTimer(timeout)
		defer t.Stop()
		timer = t.C
	}
	select {
	case g.sem <- struct{}{}:
		g.enter(sessionID)
		return time.Since(start), depth - 1, nil
	case <-ctx.Done():
		return time.Since(start), depth - 1, ctx.Err()
	case <-timer:
		return time.Since(start), depth - 1, ErrQueueTimeout
	}
}

func (g *gate) enter(sessionID string) {
	g.mu.Lock()
	g.holders[sessionID] = true
	g.mu.Unlock()
}

func (g *gate) dequeue(sessionID string) {
	g.mu.Lock()
	for i, s := range g.queue {
		if s == sessionID {
			g.queue = append(g.queue[:i], g.queue[i+1:]...)
			break
		}
	}
	g.mu.Unlock()
}

func (g *gate) release(sessionID string) {
	g.mu.Lock()
	delete(g.holders, sessionID)
	g.mu.Unlock()
	g.dequeue(sessionID)
	<-g.sem
}

// snapshot is what GetDeviceQueue reports. waiting is g.waiting as it stands:
// acquire decrements it as soon as a session gets in, so subtracting the
// holders again would undercount.
func (g *gate) snapshot() (queue []string, running int, waiting int) {
	g.mu.Lock()
	defer g.mu.Unlock()
	queue = append(queue, g.queue...)
	return queue, len(g.holders), int(g.waiting.Load())
}

// holds reports whether id is inside the gate now.
func (g *gate) holds(id string) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.holders[id]
}

// gateSet is the several gates one generate has to hold at once: a session
// split across two cards needs both cards' scratch. Gates are taken in sorted
// id order so two sessions wanting the same pair cannot deadlock.
type gateSet struct {
	gates []*gate
	held  []*gate
	sid   string
}

func (e *Engine) gatesFor(ids []string) *gateSet {
	seen := map[string]bool{}
	sorted := append([]string(nil), ids...)
	sort.Strings(sorted)
	gs := &gateSet{}
	for _, id := range sorted {
		if seen[id] {
			continue
		}
		seen[id] = true
		gs.gates = append(gs.gates, e.gate(id))
	}
	return gs
}

// acquire takes every gate in order. A failure part way releases what it has:
// holding one card's scratch while failing to get the other's would stall the
// first card for a request that is not going to run.
func (gs *gateSet) acquire(ctx context.Context, sessionID string, timeout time.Duration) (time.Duration, int32, error) {
	gs.sid = sessionID
	var total time.Duration
	var depth int32
	deadline := time.Time{}
	if timeout > 0 {
		deadline = time.Now().Add(timeout)
	}
	for _, g := range gs.gates {
		var left time.Duration
		if !deadline.IsZero() {
			if left = time.Until(deadline); left <= 0 {
				gs.release()
				return total, depth, ErrQueueTimeout
			}
		}
		w, d, err := g.acquire(ctx, sessionID, left)
		total += w
		if d > depth {
			depth = d
		}
		if err != nil {
			gs.release()
			return total, depth, err
		}
		gs.held = append(gs.held, g)
	}
	return total, depth, nil
}

func (gs *gateSet) release() {
	for i := len(gs.held) - 1; i >= 0; i-- {
		gs.held[i].release(gs.sid)
	}
	gs.held = nil
}

// contended reports whether anyone is waiting for one of these gates. A step
// loop holding them stops admitting rows when it is, so it hands them over
// rather than holding them for as long as requests keep arriving.
func (gs *gateSet) contended() bool {
	for _, g := range gs.gates {
		if g.waiting.Load() > 0 {
			return true
		}
	}
	return false
}

// mode is the weakest mode of the gates held: a session that needs a
// serialised device is serialised however parallel the rest are.
func (gs *gateSet) mode() ExecutionMode {
	m := ExecutionParallel
	for _, g := range gs.gates {
		if g.mode == ExecutionSerialised {
			m = ExecutionSerialised
		}
	}
	return m
}
