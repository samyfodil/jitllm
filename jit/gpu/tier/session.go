package tier

import (
	"sync/atomic"

	"github.com/samyfodil/jitllm/engine/nn"
	"github.com/samyfodil/jitllm/format/quant"
)

// A session is one sequence; a tier is one model. The per-sequence state (the
// KV cache, the recurrent state) belongs to the session, so several model.States
// can share one tier's weights without writing each other's history. See
// docs/design/device-sessions.md.
//
// nn.LayerDevice does not grow a session argument: model.State obtains its
// device through a type assertion, so Attach hands back a view that satisfies
// the same interface and carries the session with it.
//
// The scratch (g.bs, g.bbs and their geometry sets) is still one per device, so calls from
// different sessions are serialised on a device rather than parallel.

var nextSession atomic.Uint64

// Attach returns a device view bound to a fresh session. Every State that wants
// its own attention history calls it; two views on one GPU share the model's
// weights and kernels and nothing else.
//
// The zero session -- a caller that uses *GPU directly -- still works and is
// exactly the single-State behaviour every existing gate measures.
func (g *GPU) Attach() nn.LayerDevice {
	return &gpuSession{g: g, sid: nextSession.Add(1)}
}

type gpuSession struct {
	g   *GPU
	sid uint64
	// held is enter's snapshot of the devices, kept between calls so taking it
	// allocates nothing; nil while a call holds it (guarded by g.mu).
	held []*devTier
	// sidBuf and slotBuf are LayersSessions' row lists, reused.
	sidBuf  []uint64
	slotBuf []int
	// ple is the rows' per-layer inputs for the next call (SetLayerInputs),
	// ids their token ids (SetTokenIDs).
	ple []float32
	ids []int32
}

// PlanBlocks forwards the placement's block count; see GPU.PlanBlocks.
func (s *gpuSession) PlanBlocks(n int, extra uint64) { s.g.PlanBlocks(n, extra) }

// BeginPlacement/EndPlacement bracket a session's whole offer sequence, so two
// sessions placing at once do not interleave their offers and each see a
// budget the other is spending. The lock is on the router, not a device,
// because which device takes a block is the decision being serialised.
func (s *gpuSession) BeginPlacement() { s.g.place.Lock() }
func (s *gpuSession) EndPlacement() {
	// Auto-streamed blocks get their expert caches from the room placement
	// left (GPU.sizeAutoCaches), before another session can place.
	s.g.sizeAutoCaches()
	s.g.place.Unlock()
}

// enter makes this session current on every device and takes their call locks,
// so a concurrent session's call waits rather than interleaving into the shared
// scratch. It returns the devices it locked, for leave.
func (s *gpuSession) enter() []*devTier {
	s.g.mu.Lock()
	ds := append(s.held[:0], s.g.devs...)
	s.held = nil
	s.g.mu.Unlock()
	for _, d := range ds {
		d.busy.Lock()
		if d.cur != s.sid {
			// The captured graph belongs to the session that recorded it: it
			// bakes in buffer pointers, so replaying it would read the other
			// session's KV cache. Dropping on a switch costs a re-capture; a
			// session running several tokens in a row keeps its graph. The
			// capacity and its scratch are the session's too (switchTo).
			d.mu.Lock()
			d.switchTo(s.sid)
			d.mu.Unlock()
		}
	}
	return ds
}

// leave releases what enter took, in reverse, and keeps the snapshot for the
// next call.
func (s *gpuSession) leave(ds []*devTier) {
	for i := len(ds) - 1; i >= 0; i-- {
		ds[i].busy.Unlock()
	}
	s.g.mu.Lock()
	s.held = ds
	s.g.mu.Unlock()
}

func (s *gpuSession) Layers(lo, hi, pos, n int, x, cs, csSWA []float32, head *nn.Head) bool {
	defer s.leave(s.inputsTo(s.enter()))
	return s.g.Layers(lo, hi, pos, n, x, cs, csSWA, head)
}

func (s *gpuSession) PrepLayer(li int, p *nn.LayerPlan, w *nn.LayerWeights) bool {
	defer s.leave(s.enter())
	return s.g.PrepLayer(li, p, w)
}

func (s *gpuSession) PrewarmLayer(li int, p *nn.LayerPlan, w *nn.LayerWeights) bool {
	// Deliberately not under enter: PrewarmLayer must be safe to call while
	// Layers is running, and it touches only the host-side pack.
	return s.g.PrewarmLayer(li, p, w)
}

func (s *gpuSession) MigrateKV(li int, k, v []float32, pos int, toDevice bool) bool {
	defer s.leave(s.enter())
	return s.g.MigrateKV(li, k, v, pos, toDevice)
}

// PerSequenceKV forwards the tier's answer (nn.SeqKVDevice); it carries no
// per-sequence state.
func (s *gpuSession) PerSequenceKV() bool { return s.g.PerSequenceKV() }

func (s *gpuSession) ReserveKVSeqs(bases, ends []int) bool {
	defer s.leave(s.enter())
	return s.g.ReserveKVSeqs(bases, ends)
}

func (s *gpuSession) MigrateKVSeq(li, base int, k, v []float32, pos int, toDevice bool) bool {
	defer s.leave(s.enter())
	return s.g.MigrateKVSeq(li, base, k, v, pos, toDevice)
}

func (s *gpuSession) MigrateEnt(li, base int, ent []float32, n int, toDevice bool) bool {
	defer s.leave(s.enter())
	return s.g.MigrateEnt(li, base, ent, n, toDevice)
}

// RecSteps forwards the device's count; see devTier.recSteps. It takes no
// session lock because the caller reads it immediately after its own Layers
// call returned, on the same goroutine.
func (s *gpuSession) RecSteps() int { return s.g.RecSteps() }

// PipelineDepth forwards the GPU's answer (nn.Pipeliner).
func (s *gpuSession) PipelineDepth() int { return s.g.PipelineDepth() }

// HoldsBlock forwards the tier's answer (nn.BlockHolder).
func (s *gpuSession) HoldsBlock(li int) bool { return s.g.HoldsBlock(li) }

func (s *gpuSession) MigrateRec(li int, conv, state []float32, toDevice bool) bool {
	defer s.leave(s.enter())
	return s.g.MigrateRec(li, conv, state, toDevice)
}

func (s *gpuSession) ReserveKV(pos int) bool {
	defer s.leave(s.enter())
	return s.g.ReserveKV(pos)
}

// Refused forwards the tier's answer; see GPU.Refused.
func (s *gpuSession) Refused() []int { return s.g.Refused() }

// SpillAfter forwards; see GPU.SpillAfter.
func (s *gpuSession) SpillAfter(li int) bool { return s.g.spillAfter(li, s.sid) }

// RoomGen forwards the tier's counter; see GPU.RoomGen.
func (s *gpuSession) RoomGen() uint64 { return s.g.RoomGen() }

// TrimKV shrinks the history on the devices this session is alone on; see
// devTier.TrimKV.
func (s *gpuSession) TrimKV(pos int) bool {
	defer s.leave(s.enter())
	return s.g.TrimKV(pos)
}

// Detach gives back everything this session holds on every device -- its
// attention history and its recurrent state on every block -- and leaves the
// model's weights for the sessions still using them. Without it a closed State
// would keep its history on the card for the life of the tier.
func (s *gpuSession) Detach() {
	defer s.leave(s.enter())
	s.g.mu.Lock()
	ds := append([]*devTier(nil), s.g.devs...)
	s.g.mu.Unlock()
	for _, d := range ds {
		d.dropSession(s.sid)
	}
}

// HeldBytes is this session's history on every device (nn.Session).
func (s *gpuSession) HeldBytes() uint64 {
	s.g.mu.Lock()
	ds := append([]*devTier(nil), s.g.devs...)
	s.g.mu.Unlock()
	var n uint64
	for _, d := range ds {
		n += d.heldBy(s.sid)
	}
	return n
}

func (s *gpuSession) ReleaseLayers(lo, hi int) {
	defer s.leave(s.enter())
	s.g.ReleaseLayers(lo, hi)
}

// EmbedRows is nn.EmbedDevice: the rows are gathered on the card holding the
// head, under this session's device locks like any other call.
func (s *gpuSession) EmbedRows(ids []int32, dst []float32) bool {
	defer s.leave(s.enter())
	return s.g.EmbedRows(ids, dst)
}

func (s *gpuSession) PrepHead(h *nn.Head) bool {
	defer s.leave(s.enter())
	return s.g.PrepHead(h)
}

// The rest is the tier's, not the session's: capability and accounting queries
// carry no per-sequence state and must not queue behind another session's token.
func (s *gpuSession) MatVec(out []float32, t quant.Type, w []byte, x []float32, nrows, k int) bool {
	return s.g.MatVec(out, t, w, x, nrows, k)
}

// Reserve sizes the shared scratch: a model property, not a sequence's.
func (s *gpuSession) Reserve(maxRows, maxK int) bool {
	defer s.leave(s.enter())
	return s.g.Reserve(maxRows, maxK)
}

func (s *gpuSession) Err() string { return s.g.Err() }

func (s *gpuSession) PrepLayerOn(name string, li int, p *nn.LayerPlan, w *nn.LayerWeights) (bool, error) {
	defer s.leave(s.enter())
	return s.g.PrepLayerOn(name, li, p, w)
}

func (s *gpuSession) PrepHeadOn(name string, h *nn.Head) (bool, error) {
	defer s.leave(s.enter())
	return s.g.PrepHeadOn(name, h)
}

func (s *gpuSession) DeviceOf(li int) (string, bool) { return s.g.DeviceOf(li) }
func (s *gpuSession) HeadDeviceName() (string, bool) { return s.g.HeadDeviceName() }
func (s *gpuSession) HeadWith(li int) bool           { return s.g.HeadWith(li) }
func (s *gpuSession) Pin(li int, on bool) bool       { return s.g.Pin(li, on) }
func (s *gpuSession) Stream(li int, on bool) bool    { return s.g.Stream(li, on) }
