package tier

import (
	"unsafe"

	"github.com/samyfodil/jitllm/engine/nn"
	"github.com/samyfodil/jitllm/jit/gpu/backend"
)

// A device serves several sessions at once. What it holds for all of them --
// the blocks, the kernels, the paged history, the recurrent pools, the budget
// -- is its devShared. What a session's calls write is the session's own: the
// per-call fields below (devSess), and the scratch a call runs in (a lane). A
// devTier is one session's view of one device: the three together, and the
// counters its writes go to.
//
// The scratch is the expensive part, so a lane is held for a call, not for a
// session's life. lane0 is the scratch placement builds; a call takes it when
// it is free, and otherwise the lane its session ran in last, any other free
// lane, or a new clone of lane0 (takeLane) -- built from lane0's plans and
// charged to the budget as scratch. Sessions that step one after another run
// in one lane, as they did when the scratch was the device's; each session
// stepping at the same time as another costs one more.
//
// lane0 is what placement, PrepHead, a capacity growth and every other change
// of shape act on, through the device's own view (devShared.dv). A clone is
// not kept in step with it: laneGen moves when lane0 changes shape, and a
// clone built at an older generation is dropped rather than run.
//
// A vision tower's set is not cloned: it is one per device (visrows.go), as
// the tower's k/v pair is (transientkv.go), and a call whose lane is a clone
// runs a tower's range in lane0 (lane0View).

// devSess is one session's per-call state on one device.
type devSess struct {
	sid uint64
	// cur is the lane the session's current call holds, nil between calls,
	// and depth how many entries deep that call is (as, done); last is the
	// lane its previous call ran in, which the next prefers -- its
	// recordings are there. view is the session's devTier, over cur during a
	// call; v0 its devTier over lane0, for the ranges only lane0 runs.
	cur, last *lane
	depth     int
	view, v0  *devTier
	// embPend is the prompt chunk EmbedRows promised and the next submission
	// gathers (embed.go). Guarded by mu.
	embPend *embPend
	// bidir is the full key runs the next causal Layers calls honour
	// (keyruns.go). Guarded by mu.
	bidir []nn.KeyRun
	// winSegs is setWindows' [lo, hi) pairs, kept so a picture's windows
	// allocate nothing past the first. Guarded by mu.
	winSegs []int
	// pleHost is the rows' per-layer embedding inputs for the next call
	// (SetLayerInputs) and pleStage their padded staging.
	pleHost, pleStage []float32
	// tokIDs is the rows' token ids for the next call (SetTokenIDs), which a
	// DeepSeek V4 block routing by token reads (ds4.go).
	tokIDs []int32
	// recSteps counts the linear blocks whose emitLinear completed with the
	// submission still good, for the current Layers call: the measured answer
	// to "may a failed submission be restarted on the host". Positional proxies
	// (the run's start index, or "a linear block is placed") are wrong in one
	// direction or the other.
	recSteps int
	// sub is the submission layersOnce hands layersSession, and subBytes its
	// staging bytes; see submitArgs and stagingBytes.
	sub      submitArgs
	subBytes stagingBytes
	// launchTo is the submission's launch target; see launcher.
	launchTo launchTo
	// recDescW is recDesc's words between submissions.
	recDescW []uint32
	// stepRS is the ragStep GPU.layersRows hands this device, its slices
	// reused from step to step.
	stepRS ragStep
	// rag is the rows of an in-flight LayersRows; see rows.go.
	rag *ragStep
	// recSess and rec are recorderOf's cache.
	recSess backend.Session
	rec     backend.Recorder
	// pgSeqs is pagedAppendRows' and ReserveKVSeqs' staging, reused so a
	// token allocates nothing.
	pgSeqs []seqEnd
	pgRows []pagedRow
	// graphOff is the tier deciding, not the caller: this backend has no
	// capture, or a capture failed once. Either way, do not keep asking.
	graphOff bool
}

// lane is a scratch: the block scratches of every geometry set, the batched
// ones by width, the staging a submission launches against, and the
// recordings made over them. Every kernel in it bakes its buffers, so a call
// holds the lane it runs in until it returns.
type lane struct {
	bs *blockScratch
	// bbs is the batched-prefill scratch, one per chunk width: a prompt asks
	// for the full chunk and then a narrower tail, and each width bakes its own
	// row count into every kernel. See prepBatch.
	bbs map[int]*blockScratch
	// geoSets is the scratch sets of the geometries not in use and geoCur the
	// one in use (gemma4.go). promptW is the prompt width reserved at
	// placement, 0 where none is; stepW the ragged step's.
	geoSets         map[geoKey]geomSet
	geoCur          geoKey
	promptW, stepW  int
	partBuf, f16Buf backend.Buf
	partCap, f16Cap int
	// argmaxOut is the device argmax's one word (PrepHead).
	argmaxOut backend.Buf
	// recs is a map, not one slot, because a token can issue more than one
	// sequence (blocks under {0,gl} and a head-only call under {gl,gl}); one
	// slot would have them evict each other, a capture per call. A recording
	// bakes this lane's buffers and its session's pages: it is keyed by the
	// session (graphKey.sid) and kept with the lane.
	recs map[graphKey]backend.Recording
	// recUsed is when each recording last ran, on the recTick clock, so the
	// per-session bound (maxGraphs) retires the least recently used rather
	// than every one. An entry whose recording is gone is pruned there.
	recUsed map[graphKey]uint64
	recTick uint64
	// stale holds recordings that were retired from inside a Session and so
	// could not be destroyed there; see dropGraph.
	stale []backend.Recording
	// clone says the lane was built from lane0 (cloneLane), at lane0's
	// generation gen; held says a call runs in it. A clone borrows lane0's
	// head (shareHead) and owns everything else.
	clone, held bool
	gen         uint64
}

// devTier is one session's view of one device: the device's shared state, the
// session's per-call state, the lane its call runs in and the counters its
// writes go to. The device's own view (devShared.dv) is lane0's and the zero
// session's.
type devTier struct {
	*devShared
	*devSess
	*lane
	*Stats
	// subFn and mvcFn are layersSession and matVecSession bound to this view,
	// built once (layersOnce, matVec).
	subFn, mvcFn func(backend.Session)
}

// newDevice builds the device's view over sh: lane0, the zero session, and
// the device's counters.
func newDevice(sh *devShared) *devTier {
	l0 := &lane{bbs: map[int]*blockScratch{}}
	ds := &devSess{}
	dv := &devTier{devShared: sh, devSess: ds, lane: l0, Stats: &sh.tot}
	sh.lane0, sh.dv = l0, dv
	sh.sess = map[uint64]*devSess{0: ds}
	return dv
}

// as is session sid's view of this device for one call, holding a lane; the
// caller pairs it with done. A call already holding one (an entry reached
// from inside another, for the same session) keeps it. nil when no lane can
// be had -- every lane busy and a clone does not fit the budget: the caller
// refuses, and the State runs on the host. Callers must not hold g.mu.
func (g *devTier) as(sid uint64) *devTier {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.asLocked(sid)
}

// asLocked is as with g.mu held.
func (g *devTier) asLocked(sid uint64) *devTier {
	ds := g.sessOf(sid)
	if ds.cur != nil {
		ds.depth++
		return ds.view
	}
	l := g.takeLane(ds)
	if l == nil {
		return nil
	}
	ds.cur, ds.last, ds.depth = l, l, 1
	if ds.view == nil {
		ds.view = &devTier{devShared: g.devShared, devSess: ds, Stats: &g.tot}
	}
	ds.view.lane = l
	return ds.view
}

// done ends what as began, giving the lane back when the outermost entry
// returns. Callers must not hold g.mu.
func (g *devTier) done() {
	g.mu.Lock()
	defer g.mu.Unlock()
	ds := g.devSess
	if ds.depth--; ds.depth > 0 {
		return
	}
	ds.cur.held = false
	ds.cur = nil
}

// sessOf is session sid's state on this device, made at its first call.
// Callers hold g.mu.
func (g *devTier) sessOf(sid uint64) *devSess {
	if ds := g.sess[sid]; ds != nil {
		return ds
	}
	ds := &devSess{sid: sid}
	g.sess[sid] = ds
	return ds
}

// takeLane holds a lane for one of ds's calls: the lane it ran in last, if
// free and current, then lane0, then any free clone, then a new clone. A clone
// lane0 has moved past is dropped on the way. Callers hold g.mu.
func (g *devTier) takeLane(ds *devSess) *lane {
	g.tidyLanes()
	pick := func(l *lane) *lane {
		l.held = true
		return l
	}
	if l := ds.last; l != nil && !l.held && g.liveLane(l) {
		return pick(l)
	}
	if !g.lane0.held {
		return pick(g.lane0)
	}
	for _, l := range g.lanes {
		if !l.held {
			return pick(l)
		}
	}
	l, ok := g.dv.cloneLane()
	if !ok {
		return nil
	}
	g.lanes = append(g.lanes, l)
	return pick(l)
}

// liveLane reports that l is lane0 or one of its current clones. Callers hold
// g.mu.
func (g *devTier) liveLane(l *lane) bool {
	if l == g.lane0 {
		return true
	}
	for _, c := range g.lanes {
		if c == l {
			return true
		}
	}
	return false
}

// tidyLanes drops every free clone built before lane0's last change of shape.
// A held one is dropped when its call gives it back and the next call comes
// here. Callers hold g.mu.
func (g *devTier) tidyLanes() {
	kept := g.lanes[:0]
	for _, l := range g.lanes {
		if !l.held && l.gen != g.laneGen {
			g.dv.dropLane(l)
			continue
		}
		kept = append(kept, l)
	}
	clear(g.lanes[len(kept):])
	g.lanes = kept
	for _, ds := range g.sess {
		if ds.last != nil && !g.liveLane(ds.last) {
			ds.last = nil
		}
	}
}

// lane0View is the session's view over lane0, for a range only lane0 runs: a
// vision tower's, whose set is one per device. It is the session's own view
// when its call holds lane0. Callers hold g.mu.
func (g *devTier) lane0View() *devTier {
	if g.lane == g.lane0 {
		return g
	}
	if g.v0 == nil {
		g.v0 = &devTier{devShared: g.devShared, devSess: g.devSess, lane: g.lane0, Stats: &g.tot}
	}
	return g.v0
}

// shapeChanged records that lane0 changed shape -- a scratch built, rebuilt
// or dropped, the head placed or dropped -- so no clone of the old shape runs
// again. A change to a clone moves nothing. Callers hold g.mu.
func (g *devTier) shapeChanged() {
	if g.lane == g.lane0 {
		g.laneGen++
	}
}

// homeSet is lane l's home geometry set's decode scratch: the one holding
// the head.
func homeSet(l *lane) *blockScratch {
	if l.geoCur == 0 {
		return l.bs
	}
	return l.geoSets[0].bs
}

// cloneLane builds a lane from lane0's plans: every causal set's decode
// scratch, the batched widths lane0 holds, the staging at lane0's sizes, and
// lane0's head shared rather than copied. A vision tower's set is not cloned
// (lane0View). Everything it allocates is charged as scratch. g is the
// device's view. Callers hold g.mu.
func (g *devTier) cloneLane() (*lane, bool) {
	defer g.scratchWin().close()
	l0 := g.lane0
	nl := &lane{bbs: map[int]*blockScratch{}, clone: true, gen: g.laneGen}
	v := &devTier{devShared: g.devShared, devSess: g.devSess, lane: nl, Stats: g.Stats}
	fail := func() (*lane, bool) {
		g.dropLane(nl)
		g.ScratchRefused++
		return nil, false
	}
	sets := map[geoKey]geomSet{l0.geoCur: {bs: l0.bs, bbs: l0.bbs, promptW: l0.promptW, stepW: l0.stepW}}
	for k, s := range l0.geoSets {
		sets[k] = s
	}
	for k, s := range sets {
		if s.bs == nil || k == geoNonCausal {
			continue
		}
		v.useGeom(k)
		if v.bs = v.initScratch(&s.bs.p, 1); v.bs == nil {
			return fail()
		}
		for w, src := range s.bbs {
			if src == nil || !v.prepBatch(w) || !v.reservePaged(v.bbs[w]) {
				return fail()
			}
			if src.ragLogits != nil && !v.reserveRowsHead(v.bbs[w], &src.p) {
				return fail()
			}
		}
		v.promptW, v.stepW = s.promptW, s.stepW
	}
	v.useGeom(0)
	if h := homeSet(l0); h != nil && h.head != nil && nl.bs != nil {
		if !v.shareHead(nl.bs, h) {
			return fail()
		}
	}
	if l0.partCap > 0 && !v.sizePart(l0.partCap) {
		return fail()
	}
	if l0.f16Cap > 0 && !v.sizeF16(l0.f16Cap) {
		return fail()
	}
	if l0.argmaxOut != nil {
		b, err := g.dev.Alloc(4)
		if err != nil {
			g.LastErr = "a lane's argmax word: " + err.Error()
			return fail()
		}
		nl.argmaxOut = b
	}
	return nl, true
}

// shareHead gives dst the head src holds: the projection, its norm and its
// cap are the model's and stay src's, and dst gets logits of its own to write.
// Callers hold g.mu inside a scratch window.
func (g *devTier) shareHead(dst, src *blockScratch) bool {
	dst.head, dst.mvHead = src.head, src.mvHead
	dst.hNorm, dst.hNormB, dst.headCap = src.hNorm, src.hNormB, src.headCap
	dst.headEmbeds, dst.embScale = src.headEmbeds, src.embScale
	dst.headShared = true
	n := src.mvHead.rows
	var err error
	if dst.logits, err = g.dev.Alloc(n * 4); err != nil {
		g.LastErr = "a lane's logits: " + err.Error()
		return false
	}
	if src.logitsCap != nil {
		if dst.logitsCap, err = g.dev.Alloc(n * 4); err != nil {
			g.LastErr = "a lane's capped logits: " + err.Error()
			return false
		}
	}
	dst.hRaw = make([]byte, n*4)
	dst.hOut = unsafe.Slice((*float32)(unsafe.Pointer(&dst.hRaw[0])), n)
	return true
}

// unshareHead forgets a borrowed head, so freeing dst frees only what is
// dst's own.
func unshareHead(dst *blockScratch) {
	if dst == nil || !dst.headShared {
		return
	}
	dst.head, dst.mvHead = nil, mv{}
	dst.hNorm, dst.hNormB, dst.headCap = nil, nil, nil
	dst.headShared = false
}

// dropLane frees a clone, its recordings and its scratch, and refunds it. g is
// the device's view. Callers hold g.mu, outside any Session.
func (g *devTier) dropLane(l *lane) {
	if l == nil || !l.clone {
		return
	}
	defer g.scratchWin().close()
	v := &devTier{devShared: g.devShared, devSess: g.devSess, lane: l, Stats: g.Stats}
	v.dropLaneGraph()
	for _, s := range l.geoSets {
		unshareHead(s.bs)
	}
	unshareHead(l.bs)
	v.dropLaneScratch()
	if l.argmaxOut != nil {
		l.argmaxOut.Free()
		l.argmaxOut = nil
	}
}

// dropLanes frees every clone: what a device dropping its scratch, or closing,
// does. Callers hold g.mu, outside any Session, with no call holding a clone.
func (g *devTier) dropLanes() {
	for _, l := range g.lanes {
		g.dv.dropLane(l)
	}
	clear(g.lanes)
	g.lanes = g.lanes[:0]
	for _, ds := range g.sess {
		if ds.last != nil && ds.last != g.lane0 {
			ds.last = nil
		}
	}
}

// forgetSess forgets session sid on this device: its recordings in every
// lane, and its state. The zero session is the device's and stays. Callers
// hold g.mu, outside any Session.
func (g *devTier) forgetSess(sid uint64) {
	g.eachLane(func(l *lane) {
		for k, r := range l.recs {
			if k.sid == sid {
				l.stale = append(l.stale, r)
				delete(l.recs, k)
				delete(l.recUsed, k)
			}
		}
		freeStaleOf(l)
	})
	if sid != 0 {
		delete(g.sess, sid)
	}
}

// eachLane calls f with lane0 and every clone. Callers hold g.mu.
func (g *devTier) eachLane(f func(*lane)) {
	f(g.lane0)
	for _, l := range g.lanes {
		f(l)
	}
}

// dropLaneGraph retires the recordings made in this view's lane. Callers hold
// g.mu, outside any Session.
func (g *devTier) dropLaneGraph() {
	for k, r := range g.recs {
		g.stale = append(g.stale, r)
		delete(g.recs, k)
	}
	freeStaleOf(g.lane)
}

// freeStaleOf destroys a lane's retired recordings. Callers must be outside a
// Session.
func freeStaleOf(l *lane) {
	for i, r := range l.stale {
		r.Free()
		l.stale[i] = nil
	}
	l.stale = l.stale[:0]
}
