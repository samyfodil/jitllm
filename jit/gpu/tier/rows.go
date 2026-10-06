package tier

import (
	"fmt"
	"math"
	"slices"

	"github.com/samyfodil/jitllm/engine/nn"
	"github.com/samyfodil/jitllm/jit/gpu/backend"
	"github.com/samyfodil/jitllm/jit/gpu/ir"
	"github.com/samyfodil/jitllm/jit/gpu/kernels"
)

// ragStep is one LayersRows call's rows, read by layersOnce while it runs.
type ragStep struct {
	pos, slot []int
	// sid, when set, is each row's session: a step whose rows belong to
	// different sessions (gpuSession.LayersSessions). Nil is the current
	// session's.
	sid []uint64
	// seqLen is the positions one sequence of a session's batch spans
	// (nn.RowsDevice): row r is sequence (slot[r]-pos[r])/seqLen of its
	// session.
	seqLen int
	// runs is the rows grouped by sequence, each run's rows in position order,
	// with each run's session and seat row (groupRuns), and sess the step's
	// distinct sessions: what a linear block's recurrent descriptor is built
	// from. Set for a step that runs a linear block or crosses sessions.
	runs    [][]int
	runSid  []uint64
	runSeat []int
	sess    []uint64
	// keys is groupRuns' scratch, kept with the rest.
	keys []runKey
	// again says an earlier run of this step ran on the same device (a cut at
	// a change of geometry, gemma4.go), so the step's rows are counted once.
	again bool
}

// LayersRows runs one step for len(pos) rows, every block and the head, and
// fills head.Tokens with each row's greedy token and, with head.RowLogits, reads
// each row's logits into head.Logits (see nn.RowsDevice). The rows may be
// independent sequences or one sequence's prompt.
//
// It is batched prefill with different attention: the batched scratch runs the
// matvecs and writes each row's K and V at its own offset, and each row then
// reads its own sequence's keys (kernels.AttnScoresRagged/AttnAccRagged),
// softmaxes over its own count and ends in its own argmax. N sequences cost
// one pass over the weights instead of N.
func (g *devTier) LayersRows(lo, hi int, pos, slot []int, seqLen int, x, cs, csSWA []float32, head *nn.Head) bool {
	return g.layersRows(lo, hi, &ragStep{pos: pos, slot: slot, seqLen: seqLen}, x, cs, csSWA, head)
}

// layersRows is LayersRows for rs's rows, each of its own session where
// rs.sid is set and of the current session's otherwise.
func (g *devTier) layersRows(lo, hi int, rs *ragStep, x, cs, csSWA []float32, head *nn.Head) bool {
	fail := func(f string, a ...any) bool {
		g.mu.Lock()
		g.LastErr = fmt.Sprintf("rows: "+f, a...)
		g.mu.Unlock()
		return false
	}
	pos, slot, sid := rs.pos, rs.slot, rs.sid
	n := len(pos)
	// Gemma 4: the range's geometry's scratch set (gemma4.go), as layersCall.
	g.mu.Lock()
	prev := g.geoCur
	if g.geoUsed {
		g.useGeom(g.geomFor(lo, hi))
	}
	g.mu.Unlock()
	defer g.restoreGeom(prev)
	switch {
	case rowsFaulted("swatab") && csSWA != nil:
		csSWA = cs
	case rowsFaulted("noswa"):
		csSWA = nil
	}
	if n < 1 || len(slot) != n || lo < 0 || lo > hi || (lo == hi && head == nil) {
		return fail("%d positions, %d slots, blocks [%d,%d)", n, len(slot), lo, hi)
	}
	if g.NoBatch {
		return fail("NoBatch is set")
	}
	// A sequence's rows -- a prompt chunk beside the decoding rows -- must be
	// its next positions in order, which a linear block's run chains through
	// and which a step across sessions checks for its attention too: a repeat
	// would write one position twice.
	g.mu.Lock()
	linear := g.linearIn(lo, hi)
	var err error
	if linear || sid != nil {
		err = g.groupRuns(rs)
	}
	g.mu.Unlock()
	if err != nil {
		return fail("%v", err)
	}
	// Each row's sequence takes its own pages.
	g.pgRows = g.pagedRows(g.pgRows, n, n, 0, rs)
	if !g.pagedAppendRows(g.pgRows) {
		return false
	}
	g.mu.Lock()
	// The row grain is the matrix instruction's 32 only where there is one;
	// the dp4a twin (ragMV) and every attention tile take 8, and padding to
	// 32 would do 32 rows of work. One compile settles which this device has.
	grain := batchGrain
	if l := g.layers[lo]; l != nil && l.ok {
		// The block's first projection: a latent block with a query latent
		// has no mvq, and a linear block none of the attention's.
		for _, m := range []mv{l.mvq, l.mvQA, l.mvKVA, l.mvSO} {
			if m.kern != nil {
				g.batchMV(m, batchGrain, 1)
				break
			}
		}
	}
	if g.mmaOff || g.NoMMA {
		grain = 8
	}
	w := (n + grain - 1) / grain * grain
	if w > batchWidth {
		g.mu.Unlock()
		return fail("%d rows, more than the batch width %d", n, batchWidth)
	}
	for li := lo; li < hi; li++ {
		l := g.layers[li]
		switch {
		case l == nil || !l.ok:
			g.mu.Unlock()
			return fail("block %d is not on this device", li)
		case l.nonCausal || l.stream != nil:
			g.mu.Unlock()
			return fail("block %d is non-causal or streamed", li)
		}
	}
	// A final softcap (gemma2) caps every row the head projects, as decode's
	// head caps its one (prepRowsHead builds it from the step's head). A head
	// whose cap disagrees with the one the device was given would be a
	// different function from the sessions' decode.
	if head != nil && g.bs != nil && (head.Softcap != 0) != (g.bs.headCap != nil) {
		g.mu.Unlock()
		return fail("the step's head has a final softcap of %g and the device's head was built "+
			"with a cap %v", head.Softcap, g.bs.headCap != nil)
	}
	// The step's own width, or the reserved one when that does not fit.
	ok := g.bs != nil && (head == nil || g.bs.head != nil && g.bs.head.ok)
	if ok {
		w = g.batchFor(w)
		ok = w > 0 && g.prepRagged(g.bbs[w], head, n)
	}
	bs := g.bbs[w]
	if ok && linear {
		ok = g.prepRagLinear(bs, n) && g.prepRecRows(lo, hi, rs)
	}
	g.rag = rs
	g.mu.Unlock()
	defer func() {
		g.mu.Lock()
		g.rag = nil
		g.mu.Unlock()
	}()
	if !ok {
		return false
	}
	if head != nil && len(head.Tokens) < n {
		head.Tokens = make([]int32, n)
	}
	if !g.submit(bs, lo, hi, 0, x, cs, csSWA, head) {
		return false
	}
	g.mu.Lock()
	if sid != nil && !rs.again {
		g.SessionRows += n
	}
	if linear {
		nl := 0
		for li := lo; li < hi; li++ {
			if g.layers[li].linear {
				nl++
			}
		}
		g.RecChainRows += (n - len(rs.runs)) * nl
	}
	// Counted here rather than in emitMLARows, which a replayed recording
	// does not run.
	for li := lo; li < hi; li++ {
		// l.mla is the model's: a hybrid's linear blocks carry it too.
		if l := g.layers[li]; l.mla && !l.linear {
			g.RowsLatent++
		}
	}
	g.mu.Unlock()
	return true
}

// prepRagged builds bs's ragged-decode set once, outside any session: Compile
// and Alloc are driver calls a capture forbids. Callers hold g.mu. The head's
// part (the batched projection and the per-row argmax) is built only for a
// step that runs the head, and over the rows it wants (nn.Head.LogitRows).
func (g *devTier) prepRagged(bs *blockScratch, rowsHead *nn.Head, n int) bool {
	if bs == nil {
		return false
	}
	defer g.scratchWin().close()
	// Every block's matvecs, on every call: a map lookup once compiled, and
	// the one place they can be compiled (Compile is refused inside a session).
	for _, l := range g.layers {
		if l == nil || !l.ok {
			continue // a block of a split placement that another tier runs
		}
		// A block of another scratch set is built when its set's step is
		// (gemma4.go): this set's batched scratch may not run it at all (a
		// mixture block beside MiniMax-M3's dense lead).
		if g.geoUsed && l.geo != g.geoCur {
			continue
		}
		// A linear block's projections and the shared expert's too: every
		// matvec a ragged block can launch, as prepBatch's list.
		// A latent block's query, latent and output projections too; its
		// absorb banks run grouped (prepBatch builds them).
		for _, m := range []mv{l.mvq, l.mvk, l.mvv, l.mvo, l.mvg, l.mvu, l.mvd,
			l.mvsg, l.mvsu, l.mvsd, l.mvSG, l.mvBA, l.mvSO, l.mvFA, l.mvFB, l.mvGA, l.mvGB,
			l.mvpg, l.mvpp,
			l.mvQA, l.mvQB, l.mvKVA,
			l.mvIQ, l.mvIK, l.mvIW, l.mvLL, l.mvLR, l.mvMQ,
			l.mvCKV, l.mvCG, l.mvICKV, l.mvICG} {
			if m.kern == nil || m.slots > 0 {
				continue
			}
			if _, ok := g.ragStepMV(m, bs.rows, n); !ok {
				return false
			}
		}
		if !g.ds4Twins(l, bs.rows) {
			return false
		}
		// Decode's fusions over the step, where decode has them: built here
		// because a session cannot compile, and declined (the separate
		// launches serve) where a shape refuses.
		g.groupKern(l.mvo, n, bs.rows, groupRes, 0)
		g.groupKern(l.mvd, n, bs.rows, groupRes, 0)
		g.groupKern(l.mvu, n, bs.rows, groupGate, bs.p.Act)
		g.groupQKV(l, n, bs.rows)
	}
	return g.prepRowsHead(bs, rowsHead, n)
}

// prepRowsHead builds bs's per-row head -- the batched projection over the
// rows a step wants and a per-row argmax -- once, outside any session. A
// ragged step needs it, and so does a chunk whose head wants every row
// (nn.Head.RowLogits on Layers). Callers hold g.mu.
func (g *devTier) prepRowsHead(bs *blockScratch, rowsHead *nn.Head, n int) bool {
	defer g.scratchWin().close()
	p := &bs.p
	R := bs.rows
	fail := func(err error) bool { g.LastErr = "rows: " + err.Error(); return false }
	comp := func(k *ir.Kernel, err error) backend.Kernel {
		if err != nil {
			fail(err)
			return nil
		}
		c, err := g.dev.Compile(k)
		if err != nil {
			fail(err)
			return nil
		}
		return c
	}
	// A batch's rows read their history through the scratch's paged set
	// (pagedDecode); what is left to build is the head.
	if rowsHead == nil {
		return true
	}
	head := g.bs.head
	g.groupMV(g.headMV(), rowsHead.WantedRows(n), R)
	g.headOneMV(R)
	if bs.ragArgmax == nil {
		amk := comp(kernels.ArgmaxRows(head.nrows, R))
		if amk == nil {
			return false
		}
		hm, ok := g.ragMV(mv{q: head.t, k: p.NEmbd, rows: head.nrows}, R)
		if !ok {
			return fail(fmt.Errorf("no batched head matvec: %s", g.LastErr))
		}
		// reserveRowsHead may have allocated them at placement, from the
		// plan's vocabulary; a head of another width replaces them.
		if bs.ragLogits != nil && bs.ragVocab != head.nrows {
			freeRagOut(bs)
		}
		var err error
		if bs.ragLogits == nil {
			if bs.ragLogits, err = g.dev.Alloc(R * head.nrows * 4); err == nil {
				bs.ragVocab = head.nrows
				bs.ragTok, err = g.dev.Alloc(R * 4)
			}
		}
		if err != nil {
			return fail(err)
		}
		bs.ragArgmax, bs.ragHead = amk, hm
	}
	// The final softcap over every row the head can project: the kernel is
	// sized for R rows and launched over the wanted ones.
	if c := rowsHead.Softcap; c != 0 && (bs.ragCapK == nil || bs.ragCapC != c) {
		k := comp(kernels.Softcap(R*head.nrows, c))
		if k == nil {
			return false
		}
		if bs.ragCap == nil {
			b, err := g.dev.Alloc(R * head.nrows * 4)
			if err != nil {
				k.Close()
				return fail(err)
			}
			bs.ragCap = b
		}
		if bs.ragCapK != nil {
			bs.ragCapK.Close()
		}
		bs.ragCapK, bs.ragCapC = k, c
	}
	return true
}

// LayersRows runs blocks [lo, hi) for the rows, one call per device run, the
// residual crossing to the next device through x. Each row's history for a
// block lives in the cache of the device that holds the block. A head is only
// taken on the device that runs the last block; a head-only step (the blocks
// elsewhere, the projection here) is refused rather than run.
func (g *GPU) LayersRows(lo, hi int, pos, slot []int, seqLen int, x, cs, csSWA []float32, head *nn.Head) bool {
	return g.layersRows(lo, hi, pos, slot, nil, seqLen, x, cs, csSWA, head)
}

// layersRows is LayersRows with each row's session, or nil for the current
// one's.
func (g *GPU) layersRows(lo, hi int, pos, slot []int, sid []uint64, seqLen int, x, cs, csSWA []float32, head *nn.Head) bool {
	g.resetRecSteps()
	g.mu.Lock()
	rs, ok := g.runsInto(g.runBuf, lo, hi)
	g.runBuf = nil
	why := ""
	switch {
	case !ok || len(rs) == 0:
		why = fmt.Sprintf("rows: blocks [%d,%d) are not all on devices", lo, hi)
	case head != nil && rs[len(rs)-1].dev != g.head:
		why = "rows: the head is not on the device that runs the last block"
	}
	// A last run outside the home scratch set (gemma4.go) leaves the head to
	// a step of its own on the same device, as Layers does.
	if why == "" && head != nil && rs[len(rs)-1].geo != 0 {
		r := rs[len(rs)-1]
		rs = append(rs, run{dev: r.dev, lo: hi, hi: hi, split: r.split})
	}
	g.rowsErr = why
	g.mu.Unlock()
	defer g.putRuns(rs)
	if why != "" {
		return false
	}
	for i, r := range rs {
		var h *nn.Head
		if i == len(rs)-1 {
			h = head
		}
		// A step of its own on each device: its rows' grouping is the
		// device's. The ragStep is the device's own, its slices reused.
		st := &r.dev.stepRS
		st.pos, st.slot, st.sid, st.seqLen = pos, slot, sid, seqLen
		st.again = slices.ContainsFunc(rs[:i], func(o run) bool { return o.dev == r.dev })
		c0, c1 := r.tables(cs, csSWA)
		ok := r.dev.layersRows(r.lo, r.hi, st, x, c0, c1, h)
		// The runs go too: a step that groups none (no linear block in its
		// range, one session) would otherwise hand recDesc the last step's,
		// whose rows index past this one's.
		st.pos, st.slot, st.sid = nil, nil, nil
		st.runs, st.runSid, st.runSeat, st.sess = st.runs[:0], st.runSid[:0], st.runSeat[:0], st.sess[:0]
		if !ok {
			why = fmt.Sprintf("%s, blocks [%d,%d): %s", label(r.dev.dev, r.dev.ord), r.lo, r.hi, r.dev.stats().LastErr)
			g.mu.Lock()
			g.rowsErr = why
			g.mu.Unlock()
			return false
		}
	}
	return true
}

func (s *gpuSession) LayersRows(lo, hi int, pos, slot []int, seqLen int, x, cs, csSWA []float32, head *nn.Head) bool {
	defer s.leave(s.inputsTo(s.enter()))
	return s.g.LayersRows(lo, hi, pos, slot, seqLen, x, cs, csSWA, head)
}

// LayersSessions runs one step for rows of different sessions on this GPU, row
// i being sess[i]'s sequence at pos[i] (nn.SessionStepper): every block's
// weights are read once for all of them, where one session after another
// reads them once each. Each row's history is its own session's. A session may
// take several rows -- a prompt chunk beside the others' decoding -- which
// must be its next positions, each once (devTier.groupRuns refuses a gap or a
// repeat); every row's K and V are written before any row attends, so a
// chunk's rows see each other causally.
func (s *gpuSession) LayersSessions(lo, hi int, sess []nn.LayerDevice, pos []int, x, cs, csSWA []float32, head *nn.Head) bool {
	s.g.resetRecSteps()
	// The session's own lists, reused: a session steps from one goroutine.
	s.sidBuf = slices.Grow(s.sidBuf[:0], len(sess))[:len(sess)]
	s.slotBuf = slices.Grow(s.slotBuf[:0], len(sess))[:len(sess)]
	sid, slot := s.sidBuf, s.slotBuf
	for i, d := range sess {
		o, ok := d.(*gpuSession)
		if !ok || o.g != s.g {
			s.g.mu.Lock()
			s.g.rowsErr = fmt.Sprintf("sessions: row %d is not a session of this GPU", i)
			s.g.mu.Unlock()
			return false
		}
		sid[i], slot[i] = o.sid, pos[i]
	}
	defer s.leave(s.inputsTo(s.enter()))
	// A session is one sequence whose slots are its positions, so the span a
	// sequence takes is no matter: every row's sequence is its session's 0.
	return s.g.layersRows(lo, hi, pos, slot, sid, 0, x, cs, csSWA, head)
}

// ragKey names a ragged step's compiled matvec in devTier.ragK: the kind of
// kernel and the shape fields it is keyed on. A struct, where a formatted
// string was built for every lookup -- every matvec of every step.
type ragKey struct {
	kind                               string
	q                                  kernels.Quant
	k, rows, ntok, r, tok, rowt, split int
	tile                               [5]int
	grp, bias                          bool
	// epi and act are a few-sequence matvec's epilogue (groupKern).
	epi groupEpi
	act kernels.ActKind
}

// ragMV is a ragged step's batched twin of m, compiled on first use -- which
// prepRagged makes happen outside any session, so inside one it is a lookup.
//
// On a device with a matrix instruction it is the prefill twin. Without one it
// is not: the prefill twin's dp4a tile is sized for 128-row chunks and leaves a
// small step's grid nearly empty. Here every token of the step is in one
// thread, so a weight is read once, and k is split until the grid fills the
// card, with a Reduce after.
func (g *devTier) ragMV(m mv, ntok int) (mv, bool) {
	if _, _, ok := g.mmaTile(m.rows, ntok); ok && !kernels.IsFloat(m.q) {
		return g.batchMV(m, ntok, 1)
	}
	if bm, ok := g.voltaMV(m, ntok); ok {
		return bm, true
	}
	// Tok 8; Rowt 2 below 64 rows and 4 from there, swept on a V100 (Tok 4
	// and 16 lost at every width).
	tok, rowt := min(ntok, 8), 2
	if ntok >= 64 {
		rowt = 4
	}
	if b := g.kb.batch; b.RagTok > 0 && ntok%b.RagTok == 0 {
		tok = b.RagTok
	}
	if b := g.kb.batch; b.RagRowT > 0 {
		rowt = b.RagRowT
	}
	for ntok%tok != 0 {
		tok /= 2
	}
	for rowt > 1 && m.rows%rowt != 0 {
		rowt /= 2
	}
	return g.dot4Split(m, ntok, tok, rowt, g.kb.batch.RagSplit)
}

// headMV is the decode head projection as groupMV reads it: its split, its
// format and its k.
func (g *devTier) headMV() mv {
	m := g.bs.mvHead
	m.q, m.k = g.bs.head.t, g.bs.p.NEmbd
	return m
}

// headOneMV is decode's head matvec -- its split and its Reduce -- reading the
// first of R quantized rows, which is the one row of a ragged step that wants
// the head (rows that want logits lead): one projection where a batched twin
// would project every row of the step to read back one. The activation's
// sums sit after every row's scales (kernels.MatVecShape.ActRows), which is
// the only way it differs from decode's own kernel. Compiled on first use,
// which prepRagged makes outside any session; false where the shape is
// refused, and the batched twin serves.
func (g *devTier) headOneMV(R int) (mv, bool) {
	m := g.bs.mvHead
	if g.NoRagHeadOne || m.kern == nil {
		return mv{}, false
	}
	q, k := g.bs.head.t, g.bs.p.NEmbd
	key := ragKey{kind: "headone", q: q, k: k, rows: m.rows, r: R, split: m.split}
	c, ok := g.ragK[key]
	if !ok {
		ker, err := kernels.MatVec(kernels.MatVecShape{Center: g.center, T: q, K: k, Rows: m.rows,
			Split: m.split, ActRows: R})
		if err == nil {
			c, err = g.dev.Compile(ker)
		}
		if err != nil {
			g.LastErr = "rows: the one-row head: " + err.Error()
			c = nil
		}
		if g.ragK == nil {
			g.ragK = map[ragKey]backend.Kernel{}
		}
		g.ragK[key] = c
	}
	if c == nil {
		return mv{}, false
	}
	m.kern, m.alt = c, nil
	return m, true
}

// ragStepMV is the matvec a ragged step of n live rows in a scratch of R
// runs: the decode matvec over the n rows where it applies (groupMV), else
// the batched twin over all R, padding included.
func (g *devTier) ragStepMV(m mv, R, n int) (mv, bool) {
	if bm, ok := g.groupMV(m, n, R); ok {
		return bm, true
	}
	return g.ragMV(m, R)
}

// groupTokMax is the widest step groupMV takes: one thread carries every
// token, so its dot products grow with the step and past four sequences the
// tiled twins win. Llama-3.1-8B Q4_K_M on a V100, tok/s together, groupMV
// against the tiled twin: 2 sessions 203 / 146, 4 307 / 270, 8 363 / 488.
const groupTokMax = 4

// groupMV is m's decode matvec carrying every token of a small step in one
// thread: each weight word is read once for all the step's tokens. That is
// llama.cpp's mul_mat_vec_q at ncols > 1, and on a card with no integer
// matrix instruction it is what a step of a few sequences wants -- the
// tensor-core and dp4a tiles read the weights at three quarters the rate,
// pad to their tile and add a Reduce (and an f16 conversion) per matvec.
//
// It takes decode's own split, so each token sums its k segments as decode
// does: unsplit where decode is unsplit (the head), and otherwise reduced in
// the group whether decode reduces there or in a second launch -- the
// partials are added in segment order either way, and the launch is saved. A
// split the group cannot hold is declined, and the tiled twin serves it. The
// activation holds R quantized rows, of which the step's ntok come first.
func (g *devTier) groupMV(m mv, ntok, R int) (mv, bool) {
	return g.groupKern(m, ntok, R, groupPlain, 0)
}

// groupEpi is the epilogue a groupMV launch carries, decode's own fusions
// over every token of the step: the plain row, the residual added per token
// (decode's mvres: the projection writes x + W*a), or act(gate)*up (decode's
// mvgate). Without them a step pays an Add or an ActMul launch per block that
// a decode token does not.
type groupEpi int

const (
	groupPlain groupEpi = iota
	groupRes
	groupGate
)

// groupKern is groupMV with an epilogue; act is the gate's activation. The
// residual form goes where decode built one (m.res: the attention output and
// the FFN down projection, with no bias of their own). The gate form goes
// wherever the gate is SiLU or GELU and the up projection has no bias: decode
// fuses it only where its split writes the final row, and this kernel always
// does, reducing its split in the group.
func (g *devTier) groupKern(m mv, ntok, R int, epi groupEpi, act kernels.ActKind) (mv, bool) {
	if ntok < 2 || ntok > groupTokMax || m.slots > 0 || kernels.IsFloat(m.q) || g.NoRagGroup {
		return mv{}, false
	}
	switch {
	case epi == groupRes && (m.res == nil || m.bias != nil || g.NoRagFuse),
		epi == groupGate && (m.bias != nil || g.NoRagFuse || (act != kernels.ActSiLU && act != kernels.ActGELU)):
		return mv{}, false
	}
	split := max(m.split, 1)
	grp := split > 1
	if grp && !kernels.GroupSplitOK(m.rows, split) {
		return mv{}, false
	}
	key := ragKey{kind: "grp", q: m.q, k: m.k, rows: m.rows, ntok: ntok, r: R, split: split, grp: grp,
		bias: m.bias != nil, epi: epi, act: act}
	k, ok := g.ragK[key]
	if !ok {
		ker, err := kernels.MatVec(kernels.MatVecShape{Center: g.center, T: m.q, K: m.k, Rows: m.rows, NTok: ntok,
			Tok: ntok, ActRows: R, Split: split, GroupSplit: grp, Bias: m.bias != nil || epi == groupRes,
			BiasRows: epi == groupRes, Gate: epi == groupGate, GateAct: act})
		if err == nil {
			k, err = g.dev.Compile(ker)
		}
		if err != nil {
			k = nil
		}
		if g.ragK == nil {
			g.ragK = map[ragKey]backend.Kernel{}
		}
		g.ragK[key] = k
	}
	if k == nil {
		return mv{}, false
	}
	if epi == groupPlain {
		g.RagGroupMV++
	}
	return mv{kern: k, rows: m.rows, split: split, groups: 1, group: grp, bias: m.bias, q: m.q, k: m.k}, true
}

// groupQKV is block l's q, k and v for a step of ntok rows in a scratch of R
// as one kernels.MatVecSegments launch with every token in one thread --
// decode's l.qkv over the step, at q's split as there, reduced in the group
// -- or nil where the three cannot share one (fuseQKV's conditions, less the
// one on decode's own split: this kernel always reduces in the group).
func (g *devTier) groupQKV(l *layer, ntok, R int) (segLaunch, bool) {
	if ntok < 2 || ntok > groupTokMax || g.NoRagGroup || g.NoRagFuse || g.NoSegFuse ||
		l.mla || l.linear || l.nonCausal || l.wq == nil || l.wk == nil || l.wv == nil {
		return segLaunch{}, false
	}
	split := max(l.mvq.split, 1)
	grp := split > 1
	ms := [3]mv{l.mvq, l.mvk, l.mvv}
	sk := segShapeKey{k: l.mvq.k, ntok: ntok, r: R, split: split, grp: grp}
	for i, m := range ms {
		if m.kern == nil || m.slots > 0 || m.k != l.mvq.k || kernels.IsFloat(m.q) {
			return segLaunch{}, false
		}
		sk.t[i], sk.rows[i], sk.bias[i] = m.q, m.rows, m.bias != nil
	}
	// A lookup on a struct, since an encode runs this every block of every
	// step and a decode token allocates nothing.
	if sl, ok := g.ragSeg[sk]; ok {
		return sl, sl.kern != nil
	}
	var segs []kernels.MatVecShape
	for _, m := range ms {
		segs = append(segs, kernels.MatVecShape{Center: g.center, T: m.q, K: m.k, Rows: m.rows,
			Bias: m.bias != nil, NTok: ntok, Tok: ntok, ActRows: R})
	}
	if g.ragSeg == nil {
		g.ragSeg = map[segShapeKey]segLaunch{}
	}
	blocks, w, err := kernels.SegmentsGrid(segs, split, grp)
	if err != nil {
		g.ragSeg[sk] = segLaunch{}
		return segLaunch{}, false
	}
	key := fmt.Sprintf("%v/%d/%v", segs, split, grp)
	k, ok := g.segKerns[key]
	if !ok {
		ker, err := kernels.MatVecSegments(segs, split, grp)
		if err == nil {
			k, err = g.dev.Compile(ker)
		}
		if err != nil {
			g.LastErr = err.Error()
			k = nil
		}
		if g.segKerns == nil {
			g.segKerns = map[string]backend.Kernel{}
		}
		g.segKerns[key] = k
	}
	n := 0
	for _, b := range blocks {
		n += b
	}
	sl := segLaunch{kern: k, blocks: n, w: w}
	g.ragSeg[sk] = sl
	return sl, k != nil
}

// segShapeKey names a few-sequence step's q/k/v launch (groupQKV) by its three
// shapes, the step and the split.
type segShapeKey struct {
	t                 [3]kernels.Quant
	rows              [3]int
	bias              [3]bool
	k, ntok, r, split int
	grp               bool
}

// dot4Split is the dp4a batched twin of m with k split until the grid holds
// ~65536 threads (or split pinned when > 0), and the Reduce that sums it. The
// Tok x Rowt tile divides the grid, so without the split a narrow projection
// (1024 rows) runs nearly as long as one fourteen times its work.
func (g *devTier) dot4Split(m mv, ntok, tok, rowt, split int) (mv, bool) {
	if split < 1 {
		split = 1
		for m.rows/rowt*(ntok/tok)*split < 1<<16 && split < 64 {
			split *= 2
		}
	}
	var c backend.Kernel
	// A refused split is the descent working, not a failure: only the last
	// refusal is recorded, and only when no split compiled. Recorded as it
	// went, a narrow projection (Gemma 3n's LAuReL, k=64) left
	// "Split=4 does not divide 2 sub-blocks" in LastErr on a model that ran
	// fine.
	refusal := ""
	for ; split >= 1; split /= 2 {
		key := ragKey{kind: "dot4", q: m.q, k: m.k, rows: m.rows, ntok: ntok, tok: tok, rowt: rowt, split: split,
			bias: m.bias != nil}
		k, ok := g.ragK[key]
		if !ok {
			// A split that does not divide the row's sub-blocks is refused
			// by MatVec, and the next one down is tried.
			ker, err := kernels.MatVec(kernels.MatVecShape{Center: g.center, T: m.q, K: m.k, Rows: m.rows, NTok: ntok,
				Tok: tok, Rowt: rowt, Split: split, Bias: m.bias != nil})
			if err == nil {
				k, err = g.dev.Compile(ker)
			}
			if err != nil {
				refusal = err.Error()
				k = nil
			}
			if g.ragK == nil {
				g.ragK = map[ragKey]backend.Kernel{}
			}
			g.ragK[key] = k
		}
		if c = k; c != nil {
			break
		}
	}
	if c == nil {
		if refusal != "" {
			g.LastErr = refusal
		}
		return mv{}, false
	}
	out := mv{kern: c, rows: m.rows / rowt, split: split, groups: ntok / tok, bias: m.bias,
		q: m.q, k: m.k, redN: m.rows * ntok}
	if split > 1 {
		if out.red = g.reduceKernel(m.rows*ntok, split); out.red == nil ||
			!g.sizePart(m.rows*ntok*split) {
			return mv{}, false
		}
	}
	return out, true
}

// voltaMV is m's batched twin on sm_70's f16 tensor cores (kernels.MatVecMMA70)
// and the ActF16 conversion it reads, or false where it does not apply: a card
// with an integer matrix instruction takes that instead (batchMV asks first),
// and a backend that does not lower the m8n8k4 shape refuses the compile, which
// is remembered. On a V100 it is ~2.3x the dp4a twin (MT=2 NT=4, swept).
func (g *devTier) voltaMV(m mv, ntok int) (mv, bool) {
	if g.NoVolta || !(g.mmaOff || g.NoMMA) || m.slots > 0 || !kernels.Volta70OK(m.q) {
		return mv{}, false
	}
	if bm, ok := g.voltaGemm(m, ntok); ok {
		return bm, true
	}
	mt, nt := 2, 4
	for nt > 1 && ntok%(8*nt) != 0 {
		nt /= 2
	}
	for mt > 1 && m.rows%(32*mt) != 0 {
		mt /= 2
	}
	if ntok%(8*nt) != 0 || m.rows%(32*mt) != 0 {
		return mv{}, false
	}
	// Split k until the grid holds ~1280 warps (sixteen an SM on a V100): the
	// tile alone left the narrow projections 64-256 of them.
	warps := (m.rows / (32 * mt)) * (ntok / (8 * nt))
	split := g.kb.batch.Split
	if split < 1 {
		split = 1
		for warps*split < 1280 && split < 16 {
			split *= 2
		}
	}
	var c backend.Kernel
	for ; split >= 1; split /= 2 {
		key := ragKey{kind: "volta", q: m.q, k: m.k, rows: m.rows, ntok: ntok, tok: mt, rowt: nt, split: split,
			bias: m.bias != nil}
		k, ok := g.ragK[key]
		if !ok {
			// A split that does not divide the sub-blocks is refused and the
			// next one down tried; a backend without the shape refuses them all.
			ker, err := kernels.MatVecMMA70(kernels.MatVecShape{Center: g.center, T: m.q, K: m.k, Rows: m.rows, NTok: ntok,
				MT: mt, NT: nt, Split: split, Bias: m.bias != nil})
			if err == nil {
				k, err = g.dev.Compile(ker)
			}
			if err != nil {
				k = nil
			}
			if g.ragK == nil {
				g.ragK = map[ragKey]backend.Kernel{}
			}
			g.ragK[key] = k
		}
		if c = k; c != nil {
			break
		}
	}
	if c == nil {
		return mv{}, false
	}
	akey := ragKey{kind: "actf16", q: m.q, ntok: ntok, k: m.k}
	a, ok := g.ragK[akey]
	if !ok {
		ker, err := kernels.ActF16(m.q, ntok, m.k)
		if err == nil {
			a, err = g.dev.Compile(ker)
		}
		if err != nil {
			a = nil
		}
		g.ragK[akey] = a
	}
	if a == nil || !g.sizeF16(ntok*m.k*2) {
		return mv{}, false
	}
	out := mv{kern: c, act: a, actN: ntok * m.k / 4, rows: m.rows, split: split, bias: m.bias,
		q: m.q, k: m.k, threads: warps * split * 32, redN: m.rows * ntok}
	if split > 1 {
		if out.red = g.reduceKernel(m.rows*ntok, split); out.red == nil ||
			!g.sizePart(m.rows*ntok*split) {
			return mv{}, false
		}
	}
	g.VoltaMV++
	return out, true
}

// tileGemms are GemmTile's blockings, widest first, all four subgroups (runBatched
// launches 128-thread workgroups); the narrower ones exist for a row or token
// count the wider blocks do not divide.
//
// 64x64 leads 64x32 (llama.cpp's kernel_mul_mm block): a few percent faster on
// the M4 by TestGemmTileSpeed and in pp512.
var tileGemms = []kernels.TileGemm{
	{MT: 4, NT: 4, WM: 2, WN: 2}, {MT: 4, NT: 2, WM: 2, WN: 2}, {MT: 2, NT: 2, WM: 2, WN: 2}, {MT: 1, NT: 2, WM: 2, WN: 2},
}

// tileMV is m's batched twin as kernels.GemmTile (binary16 tiles on the
// collective matrix unit, staged through workgroup memory) with the ActF16T
// conversion it reads; false where the device does not lower it or no tile
// divides the shape, and the next twin is asked.
//
// Only Metal lowers this form, so it is gated on the API rather than probed by
// a refused compile per shape on every other device. A Metal refusal is
// recorded once (tileOff): it is a property of the device, not the shape.
func (g *devTier) tileMV(m mv, ntok int) (mv, bool) {
	if g.tileOff || g.NoVolta || g.dev.API() != "msl" || m.slots > 0 || !kernels.Volta70OK(m.q) {
		return mv{}, false
	}
	sub, _, _, _ := kernels.Layout(m.q)
	for _, tl := range tileGemms {
		tl.KB = max(32/sub, 1)
		if m.rows%tl.Rows() != 0 || ntok%tl.Toks() != 0 {
			continue
		}
		s := kernels.MatVecShape{Center: g.center, T: m.q, K: m.k, Rows: m.rows, NTok: ntok, Bias: m.bias != nil}
		key := ragKey{kind: "gemmtile", q: m.q, k: m.k, rows: m.rows, ntok: ntok,
			tile: [5]int{tl.MT, tl.NT, tl.WM, tl.WN, tl.KB}, bias: s.Bias}
		c, ok := g.ragK[key]
		if !ok {
			ker, err := kernels.GemmTile(s, tl)
			if err == nil {
				c, err = g.dev.Compile(ker)
			}
			if err != nil {
				g.tileOff, g.TileWhy = true, err.Error()
				return mv{}, false
			}
			if g.ragK == nil {
				g.ragK = map[ragKey]backend.Kernel{}
			}
			g.ragK[key] = c
		}
		akey := ragKey{kind: "actf16t", q: m.q, ntok: ntok, k: m.k}
		a, ok := g.ragK[akey]
		if !ok {
			ker, err := kernels.ActF16T(m.q, ntok, m.k)
			if err == nil {
				a, err = g.dev.Compile(ker)
			}
			if err != nil {
				a = nil
			}
			g.ragK[akey] = a
		}
		if a == nil || !g.sizeF16(ntok*m.k*2) {
			return mv{}, false
		}
		g.TileMV++
		return mv{kern: c, act: a, actN: ntok * m.k / sub, rows: m.rows, split: 1, bias: m.bias,
			q: m.q, k: m.k, threads: kernels.GemmTileGroups(s, tl) * tl.Threads(), redN: m.rows * ntok}, true
	}
	return mv{}, false
}

// voltaTiles are GemmVolta's blockings, widest first. All are four warps
// (runBatched launches 128-thread workgroups), all four along the tokens: a
// 2x2 warp grid dequantizes twice the weights per MMA and measured slower on a
// V100. The 96-row tiles serve matrices no 128-row block divides (gpt-oss's
// 2880 rows).
var voltaTiles = []kernels.VoltaTile{
	{MT: 4, NT: 4, WM: 1, WN: 4}, {MT: 4, NT: 2, WM: 1, WN: 4}, {MT: 4, NT: 1, WM: 1, WN: 4},
	{MT: 3, NT: 4, WM: 1, WN: 4}, {MT: 3, NT: 2, WM: 1, WN: 4},
}

// voltaGemm is m's batched twin as kernels.GemmVolta, the shared-memory-staged
// form of MatVecMMA70, with the ActF16T conversion it reads; false where no
// tile divides the shape, and voltaMV then takes MatVecMMA70.
//
// k is staged 32 elements a trip (one sub-block of a 32-element format, two of
// Q6_K's 16): four 16-byte chunks a row, the width the swizzle is built for,
// keeping a double-buffered 128x128 block at 32 KiB. k is split only where the
// grid is thin (kb0Split); a well-filled grid is fastest unsplit.
func (g *devTier) voltaGemm(m mv, ntok int) (mv, bool) {
	sub, _, _, _ := kernels.Layout(m.q)
	kbN := max(32/sub, 1)
	for _, tl := range voltaTiles {
		tl.KB = kbN
		s := kernels.MatVecShape{Center: g.center, T: m.q, K: m.k, Rows: m.rows, NTok: ntok, Bias: m.bias != nil}
		if m.rows%tl.Rows() != 0 || ntok%tl.Toks() != 0 {
			continue
		}
		split := kb0Split(m.k/sub/kbN, (m.rows/tl.Rows())*(ntok/tl.Toks()))
		if f := g.kb.batch.Split; f >= 1 {
			split = f
		}
		var c backend.Kernel
		for ; split >= 1; split /= 2 {
			s.Split = split
			key := ragKey{kind: "gemmvolta", q: m.q, k: m.k, rows: m.rows, ntok: ntok,
				tile: [5]int{tl.MT, tl.NT, tl.WM, tl.WN, tl.KB}, split: split, bias: s.Bias}
			k, ok := g.ragK[key]
			if !ok {
				ker, err := kernels.GemmVolta(s, tl)
				if err == nil {
					k, err = g.dev.Compile(ker)
				}
				if err != nil {
					k = nil
				}
				if g.ragK == nil {
					g.ragK = map[ragKey]backend.Kernel{}
				}
				g.ragK[key] = k
			}
			if c = k; c != nil {
				break
			}
		}
		if c == nil {
			continue
		}
		akey := ragKey{kind: "actf16t", q: m.q, ntok: ntok, k: m.k}
		a, ok := g.ragK[akey]
		if !ok {
			ker, err := kernels.ActF16T(m.q, ntok, m.k)
			if err == nil {
				a, err = g.dev.Compile(ker)
			}
			if err != nil {
				a = nil
			}
			g.ragK[akey] = a
		}
		if a == nil || !g.sizeF16(ntok*m.k*2) {
			return mv{}, false
		}
		out := mv{kern: c, act: a, actN: ntok * m.k / sub, rows: m.rows, split: split, bias: m.bias,
			q: m.q, k: m.k, threads: kernels.GemmVoltaGroups(s, tl) * tl.Threads(), redN: m.rows * ntok}
		if split > 1 {
			if out.red = g.reduceKernel(m.rows*ntok, split); out.red == nil ||
				!g.sizePart(m.rows*ntok*split) {
				return mv{}, false
			}
		}
		g.VoltaMV++
		g.VoltaGemm++
		return out, true
	}
	return mv{}, false
}

// kb0Split is GemmVolta's k-split for a grid of groups workgroups over trips
// k-trips: doubled while under 64 workgroups (a V100 has 80 SMs and holds two
// of these each) and while the halves still divide the trips.
func kb0Split(trips, groups int) int {
	split := 1
	for groups*split < 64 && trips%(2*split) == 0 && split < 8 {
		split *= 2
	}
	return split
}

// prepRagLinear builds bs's linear-block kernels for a step of n rows, once per
// size and outside any session. Callers hold g.mu.
//
// The rows are the step's runs (kernels' run forms): each sequence's rows at
// consecutive positions chain through its one state, so a prompt chunk riding
// beside decoding rows steps once a row, as a prefill would. Sized to the real
// rows, not the padded width: the padding never reaches a state, and the next
// states (recNext, in the shared form) are one a run.
func (g *devTier) prepRagLinear(bs *blockScratch, n int) bool {
	r := bs.rec
	if r.Conv == 0 || bs.ragLin[n] != nil {
		return true
	}
	fail := func(err error) bool { g.LastErr = "rows: " + err.Error(); return false }
	// The scan's lanes are the device's promise: a lane group a state row
	// where the subgroup is guaranteed, and where it is not -- llvmpipe, a
	// Vulkan driver that will not pin the width -- one thread a row, the whole
	// row in its registers and both dots summed serially. That is the same
	// generator at one lane, with no shuffle in it, so it lowers on every
	// backend. ScalarLinearRows takes it on a device that has the subgroup.
	lanes := bs.qkLanes
	if g.ScalarLinearRows {
		lanes = 1
	}
	// A short convolution (LFM2) has no rule to scan: its run is the
	// convolution and the shift alone.
	var d kernels.DeltaScan
	rk := &ragLinear{}
	if !r.ShortConv {
		d = deltaScanOf(r, n, lanes)
		d.Runs = true
		rk.perRun, rk.lanes = d.Threads()/n, d.ScanLanes()
	}
	eps := bs.p.RMSEps
	if r.ChanDecay {
		eps = 1e-6
	}
	var err error
	for _, k := range []struct {
		dst *backend.Kernel
		mk  func() (*ir.Kernel, error)
	}{
		{&rk.delta, func() (*ir.Kernel, error) {
			if r.ShortConv {
				return nil, nil
			}
			if r.SSD {
				return kernels.GatedDeltaFused(d, kernels.DeltaFuse{Chans: r.Chans, SSD: true})
			}
			if r.Mamba1 {
				return kernels.GatedDeltaFused(d, kernels.DeltaFuse{Mamba1: true})
			}
			return kernels.GatedDeltaFused(d, kernels.DeltaFuse{Chans: r.Chans,
				Eps: float32(eps / float64(r.KDim)), Scale: float32(1 / math.Sqrt(float64(r.KDim))),
				BARep: r.VHeads / r.KHeads, Bound: r.DecayBound})
		}},
		{&rk.conv, func() (*ir.Kernel, error) { return kernels.Conv1dRowsRuns(r.Conv, r.Chans, n) }},
		{&rk.shift, func() (*ir.Kernel, error) { return kernels.Conv1dShiftRuns(r.Conv, r.Chans, n) }},
	} {
		var kk *ir.Kernel
		if kk, err = k.mk(); err == nil && kk != nil {
			*k.dst, err = g.dev.Compile(kk)
		}
		if err != nil {
			for _, c := range []backend.Kernel{rk.conv, rk.shift, rk.delta} {
				if c != nil {
					c.Close()
				}
			}
			return fail(err)
		}
	}
	if bs.ragLin == nil {
		bs.ragLin = map[int]*ragLinear{}
	}
	bs.ragLin[n] = rk
	return true
}

// ResetRecRows zeroes the recurrent state (the conv window and the delta
// state, in every half) of the given rows in every linear block of this
// session: a row a new sequence takes must not start from the last one's
// summary. It is a write per buffer, outside any submission.
func (g *devTier) ResetRecRows(rows []int) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	st, seated := g.recSeats[g.cur]
	if !seated || len(rows) == 0 {
		return true
	}
	for _, l := range g.layers {
		if l == nil || !l.linear || l.recOf(g.cur) == nil {
			continue
		}
		for _, row := range rows {
			if row < 0 || row >= st.rows {
				continue // a row the batch has not grown to holds nothing yet
			}
			if err := g.recZero(l, st.base+row, 1); err != nil {
				g.LastErr = "rows: " + err.Error()
				return false
			}
		}
	}
	return true
}

// RecMark marks the session's state on every device (nn.RecRewinder).
func (g *GPU) RecMark() {
	for _, d := range g.devs {
		d.RecMark()
	}
}

// RecRewind rewinds every device or none: every device is checked before any
// flips, so a refusal on the second cannot leave the first rewound.
func (g *GPU) RecRewind() bool {
	for _, d := range g.devs {
		if _, ok := d.rewindable(); !ok {
			return false
		}
	}
	for _, d := range g.devs {
		if !d.RecRewind() {
			return false // checked above; a session's own state does not move between
		}
	}
	return true
}

func (s *gpuSession) RecMark() {
	defer s.leave(s.enter())
	s.g.RecMark()
}

func (s *gpuSession) RecRewind() bool {
	defer s.leave(s.enter())
	return s.g.RecRewind()
}

// ResetRecRows resets the rows on every device; see devTier.ResetRecRows.
func (g *GPU) ResetRecRows(rows []int) bool {
	ok := true
	for _, d := range g.devs {
		ok = d.ResetRecRows(rows) && ok
	}
	return ok
}

func (s *gpuSession) ResetRecRows(rows []int) bool {
	defer s.leave(s.enter())
	return s.g.ResetRecRows(rows)
}
