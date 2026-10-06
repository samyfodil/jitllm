package tier

import (
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"sort"
	"sync"
	"time"

	"github.com/samyfodil/jitllm/jit/gpu/backend"
	"github.com/samyfodil/jitllm/jit/gpu/ir"
	"github.com/samyfodil/jitllm/jit/gpu/kernels"
)

// Split width, measured on the device that will run it rather than tabulated:
// chooseSplit's fitted rule missed the optimum on most shapes, even on the
// card it was fitted on.
//
// The sweep runs at load against the real resident weights, so it measures the
// footprint that will actually be read, and costs a few milliseconds once.
//
// A dispersed measurement is refused, not rounded: above IQR/median 0.10 the
// table's answer stands.
const splitTuneIQR = 0.10

type splitKey struct {
	q       kernels.Quant
	rows, k int
}

// tuneSplit returns the best split for one shape, measuring once per shape.
func (g *devTier) tuneSplit(q kernels.Quant, rows, k int, r *resident) int {
	if g.kb.split >= 1 || g.kb.tune == TuneOff {
		return chooseSplit(rows, k/32, g.kb.split)
	}
	// Metal is measured like every other backend; this depends on Session's
	// Sync actually waiting for the GPU (see backend/metal.go).
	key := splitKey{q, rows, k}
	if s, ok := g.splits[key]; ok {
		return s
	}
	fallback := chooseSplit(rows, k/32, g.kb.split)
	g.splits[key] = fallback // so a failure below is not retried per layer

	nb := k / 32
	out, err := g.dev.Alloc(rows * 4)
	if err != nil {
		return fallback
	}
	defer out.Free()

	// A float weight reads a K-float activation where a quant's scale plane
	// goes; its own d buffer is one word.
	d := r.d
	if kernels.IsFloat(q) {
		xf, err := g.dev.Alloc(k * 4)
		if err != nil {
			return fallback
		}
		defer xf.Free()
		if xf.Write(make([]byte, k*4)) != nil {
			return fallback
		}
		d = xf
	}
	best, bestRate, bestGroup := 0, 0.0, false
	for _, split := range splitCands(nb, 32) {
		if rows*split > 1<<22 {
			continue
		}
		// Both reductions are timed at every split: which wins is a property of
		// the shape, not the device (on Metal in-group wins big on q/o and
		// loses on a Q6_K head).
		for _, grp := range []bool{false, true} {
			if grp && g.NoGroupSplit {
				continue
			}
			if grp && !kernels.GroupSplitOK(rows, split) {
				continue
			}
			// The split tuner times the unbiased shape; a bias is one add per
			// row and does not move the split.
			kern := g.kernelMode(q, k, rows, split, 1, false, grp)
			if kern == nil {
				continue
			}
			var red backend.Kernel
			if split > 1 && !grp {
				if red = g.reduceKernel(rows, split); red == nil {
					continue
				}
				if !g.sizePart(rows * split) {
					continue
				}
			}
			// The in-group kernel writes the final row, so it needs no partial
			// plane, but it launches the same grid as the arm it is compared
			// against: `grp` says the output contract, `split` the grid.
			rate, ok := g.timeSplitMode(kern, red, r, d, out, rows, split, grp)
			if ok && rate > bestRate {
				best, bestRate, bestGroup = split, rate, grp
			}
		}
	}
	if best == 0 {
		return fallback
	}
	g.groupSplit[key] = bestGroup
	g.splits[key] = best
	if backend.Verbose() {
		fmt.Fprintf(os.Stderr, "tier: %s %dx%d split %d%s measured (%.0f/s), table says %d\n",
			q, rows, k, best, map[bool]string{true: " in-group"}[bestGroup], bestRate, fallback)
	}
	return best
}

func (g *devTier) timeSplitMode(kern, red backend.Kernel, r *resident, d, out backend.Buf, rows, split int, grp bool) (float64, bool) {
	const rounds, iters = 9, 4
	rates := make([]float64, 0, rounds)
	for round := 0; round < rounds+1; round++ {
		t0 := time.Now()
		var err error
		g.dev.Session(func(s backend.Session) {
			for i := 0; i < iters; i++ {
				dst := out
				if split > 1 && !grp {
					dst = g.partBuf
				}
				if err = s.Launch(kern, groupsOf(rows*split, split, grp), widthOf(split, grp),
					r.qs, d, r.sc, g.bs.a, g.bs.ax, dst); err != nil {
					return
				}
				if split > 1 && !grp {
					if err = s.Launch(red, (rows+127)/128, 128, g.partBuf, out); err != nil {
						return
					}
				}
			}
			err = s.Sync()
		})
		if err != nil {
			return 0, false
		}
		if round == 0 {
			continue // the first round pays compilation and first-touch
		}
		rates = append(rates, float64(iters)/time.Since(t0).Seconds())
	}
	sort.Float64s(rates)
	med := rates[len(rates)/2]
	iqr := rates[len(rates)*3/4] - rates[len(rates)/4]
	if med <= 0 || iqr/med > splitTuneIQR {
		return 0, false
	}
	return med, true
}

// retuneDecode re-picks each dense decode matvec's split once every block is
// placed, by timing it the way a token runs it: one launch per placed block's
// OWN weights, in order, so each launch streams a matrix the one before it did
// not touch.
//
// tuneSplit times one matrix repeatedly, which a small matrix serves from L2,
// while a decode token reads every weight from DRAM; the cache-resident answer
// can be the slowest in a token. A sequence over the real blocks is the regime
// itself (flushing L2 per trial measured worse; see
// docs/engineering-history/gpu-kernels.md, "Flushing L2 before each
// split-tuner trial was tried").
//
// It runs from Layers, before the first decode token is recorded, and again
// whenever the number of placed blocks has doubled since. Callers hold no lock
// and are outside any backend Session.
func (g *devTier) retuneDecode() {
	g.mu.Lock()
	defer g.mu.Unlock()
	// Doubling rather than every change: -gpu-grow adopts one block a token,
	// and a pass is tens of milliseconds a shape.
	if g.kb.split >= 1 || g.kb.tune == TuneOff || g.bs == nil || len(g.layers) < 2*g.retunedAt || len(g.layers) == 0 {
		return
	}
	g.retunedAt = len(g.layers)
	type use struct {
		m *mv
		r *resident
	}
	by := map[splitKey][]use{}
	for _, l := range g.layers {
		if l == nil || !l.ok || l.nonCausal || l.linear || (l.pg != nil && !l.pg.in) {
			continue
		}
		for _, u := range []use{{&l.mvq, l.wq}, {&l.mvk, l.wk}, {&l.mvv, l.wv}, {&l.mvo, l.wo},
			{&l.mvg, l.gate}, {&l.mvu, l.up}, {&l.mvd, l.down},
			// MLA's projections and the shared expert are dense decode shapes
			// too, and the single-matrix trial picks them unreliably.
			{&l.mvQA, l.wqa}, {&l.mvQB, l.wqb}, {&l.mvKVA, l.wkva},
			{&l.mvsg, l.shGate}, {&l.mvsu, l.shUp}, {&l.mvsd, l.shDown}} {
			m, r := u.m, u.r
			if r == nil || m.kern == nil || m.slots > 0 || m.groups > 0 || m.threads > 0 ||
				m.rowt > 1 || m.bias != nil || kernels.IsFloat(m.q) || r.nrows != m.rows {
				continue
			}
			key := splitKey{m.q, m.rows, m.k}
			by[key] = append(by[key], u)
		}
	}
	changed := false
	for key, us := range by {
		if len(us) < 2 {
			continue
		}
		// 16 blocks stream well past a mid-size card's L2 but not far past a large one's;
		// widen it if a card's L2 ever holds the whole sequence.
		rs := make([]*resident, 0, 16)
		for _, u := range us[:min(len(us), 16)] {
			rs = append(rs, u.r)
		}
		ck := ""
		if id := binaryID(); id != "" {
			ck = fmt.Sprintf("%s-%s-%v-%dx%d", id, deviceKey([]backend.Device{g.dev}), key.q, key.rows, key.k)
		}
		var cands []splitCand
		nb := key.k / 32
		for _, split := range splitCands(nb, 64) {
			if key.rows*split > 1<<22 {
				continue
			}
			for _, grp := range []bool{false, true} {
				if grp && (g.NoGroupSplit || !kernels.GroupSplitOK(key.rows, split)) {
					continue
				}
				c := splitCand{split: split, grp: grp}
				if c.kern = g.kernelMode(key.q, key.k, key.rows, split, 1, false, grp); c.kern == nil {
					continue
				}
				if split > 1 && !grp {
					if c.red = g.reduceKernel(key.rows, split); c.red == nil || !g.sizePart(key.rows*split) {
						continue
					}
				}
				cands = append(cands, c)
			}
		}
		// A candidate with a reduce launches twice per matvec and must win by
		// more than the timing can resolve (~1 us a launch) to be taken over one
		// that writes its row itself.
		cost := func(c splitCand) float64 {
			if c.split > 1 && !c.grp {
				return c.t * 1.03
			}
			return c.t
		}
		best, bestT, bestGrp := 0, 0.0, false
		if v, ok := lookupCache("gpusplit", ck); ok && ck != "" && g.kb.tune != TuneForce {
			var gi int
			if n, _ := fmt.Sscanf(v, "%d %d", &best, &gi); n != 2 {
				best = 0
			}
			bestGrp = gi == 1
		} else {
			for _, c := range g.timeSequences(cands, rs, key.rows, nil) {
				if c.t > 0 && (best == 0 || cost(c) < bestT) {
					best, bestT, bestGrp = c.split, cost(c), c.grp
				}
			}
			if best > 0 && ck != "" {
				gi := 0
				if bestGrp {
					gi = 1
				}
				saveCache("gpusplit", ck, fmt.Sprintf("%d %d", best, gi))
			}
		}
		m0 := us[0].m
		if best == 0 || (best == m0.split && bestGrp == m0.group) {
			continue
		}
		kern := g.kernelMode(key.q, key.k, key.rows, best, 1, false, bestGrp)
		var red backend.Kernel
		if best > 1 && !bestGrp {
			red = g.reduceKernel(key.rows, best)
		}
		if kern == nil || (best > 1 && !bestGrp && red == nil) {
			continue
		}
		if backend.Verbose() {
			fmt.Fprintf(os.Stderr, "tier: %s %dx%d retuned over %d blocks: split %d%s -> %d%s (%.1f us a launch)\n",
				key.q, key.rows, key.k, len(us), m0.split, map[bool]string{true: " in-group"}[m0.group],
				best, map[bool]string{true: " in-group"}[bestGrp], bestT*1e6)
		}
		for _, u := range us {
			u.m.kern, u.m.red, u.m.split, u.m.group = kern, red, best, bestGrp
			if u.m.res != nil {
				u.m.res = g.kernelMode(key.q, key.k, key.rows, best, 1, true, bestGrp)
			}
			if u.m.gated != nil {
				u.m.gated = g.gatedKernel(key.q, key.k, key.rows, best, bestGrp, g.bs.p.Act)
			}
		}
		g.splits[key], g.groupSplit[key] = best, bestGrp
		changed = true
	}
	if g.retuneIndexed() {
		changed = true
	}
	if changed {
		for _, l := range g.layers {
			if l != nil && l.ok {
				g.fuseQKV(l)
			}
		}
		g.dropGraph()
	}
}

// splitCands is every split up to max that divides nb activation blocks,
// ascending. Divisors, not powers of two: gpt-oss's k of 2880 is 90 blocks,
// which only 2 of the powers of two divide. For a power-of-two nb the lists
// are the same.
func splitCands(nb, max int) []int {
	var out []int
	for s := 1; s <= max && s <= nb; s++ {
		if nb%s == 0 {
			out = append(out, s)
		}
	}
	return out
}

// retuneIndexed does for a mixture's EXPERT matvecs what retuneDecode does for
// the dense ones: it times every split -- partial-writing and in-group -- over
// a sequence of placed blocks' own banks, the way a token runs them, and adopts
// the fastest, where the table (fitted to one small card, powers of two only)
// badly under-filled larger cards. It reports whether any matvec changed. A
// streamed bank and a biased one are left to the table. Caller holds g.mu.
func (g *devTier) retuneIndexed() bool {
	if g.bs == nil {
		return false
	}
	type idKey struct {
		q                       kernels.Quant
		rows, k, experts, slots int
		slotAct                 bool
	}
	type use struct {
		m *mv
		r *resident
	}
	by := map[idKey][]use{}
	var order []idKey
	for _, l := range g.layers {
		if l == nil || !l.ok || l.stream != nil || (l.pg != nil && !l.pg.in) {
			continue
		}
		// A mixture's three banks, and MLA's two per-head absorb banks, which
		// are the same indexed shape with the identity as the selection.
		var cand []use
		if l.router != nil {
			cand = append(cand, use{&l.mvg, l.gate}, use{&l.mvu, l.up}, use{&l.mvd, l.down})
		}
		if l.mla {
			cand = append(cand, use{&l.mvKB, l.wkb}, use{&l.mvVB, l.wvb})
		}
		for _, u := range cand {
			m, r := u.m, u.r
			if r == nil || m.kern == nil || m.slots < 1 || m.experts < 2 || m.bias != nil ||
				kernels.IsFloat(m.q) {
				continue
			}
			key := idKey{m.q, m.rows, m.k, m.experts, m.slots, m.slotAct}
			if by[key] == nil {
				order = append(order, key)
			}
			by[key] = append(by[key], u)
		}
	}
	changed := false
	for _, key := range order {
		us := by[key]
		if len(us) < 2 {
			continue
		}
		nout := key.rows * key.slots
		ck := ""
		if id := binaryID(); id != "" {
			ck = fmt.Sprintf("%s-%s-%v-%dx%d-e%dx%d-%v", id, deviceKey([]backend.Device{g.dev}),
				key.q, key.rows, key.k, key.experts, key.slots, key.slotAct)
		}
		mk := func(split int, grp bool) (backend.Kernel, backend.Kernel, bool) {
			kern, err := kernels.MatVec(kernels.MatVecShape{Center: g.center, T: key.q, K: key.k, Rows: key.rows,
				Split: split, Experts: key.experts, Slots: key.slots, SlotAct: key.slotAct, GroupSplit: grp})
			if err != nil {
				return nil, nil, false
			}
			c, err := g.dev.Compile(kern)
			if err != nil {
				return nil, nil, false
			}
			var red backend.Kernel
			if split > 1 && !grp {
				if red = g.reduceKernel(nout, split); red == nil || !g.sizePart(nout*split) {
					return nil, nil, false
				}
			}
			return c, red, true
		}
		best, bestGrp := 0, false
		if v, ok := lookupCache("gpusplitid", ck); ok && ck != "" && g.kb.tune != TuneForce {
			var gi int
			if n, _ := fmt.Sscanf(v, "%d %d", &best, &gi); n != 2 {
				best = 0
			}
			bestGrp = gi == 1
		} else {
			var cands []splitCand
			for _, split := range splitCands(key.k/32, 64) {
				if nout*split > 1<<22 {
					continue
				}
				for _, grp := range []bool{false, true} {
					if grp && (g.NoGroupSplit || !kernels.GroupSplitOK(key.rows, split)) {
						continue
					}
					if kern, red, ok := mk(split, grp); ok {
						cands = append(cands, splitCand{kern: kern, red: red, split: split, grp: grp})
					}
				}
			}
			// The selection: distinct experts spread over the bank, the same
			// for every block of the sequence -- each block's bank is its own
			// memory, so the sequence still streams from DRAM.
			ids := make([]uint32, key.slots)
			for i := range ids {
				ids[i] = uint32(i * key.experts / key.slots)
			}
			sel, err := g.dev.Alloc(len(ids) * 4)
			if err != nil {
				continue
			}
			if sel.Write(u32b(ids)) != nil {
				sel.Free()
				continue
			}
			rs := make([]*resident, 0, 16)
			for _, u := range us[:min(len(us), 16)] {
				rs = append(rs, u.r)
			}
			bestT := 0.0
			cost := func(c splitCand) float64 {
				if c.split > 1 && !c.grp {
					return c.t * 1.03
				}
				return c.t
			}
			for _, c := range g.timeSequences(cands, rs, nout, sel) {
				if c.t > 0 && (best == 0 || cost(c) < bestT) {
					best, bestT, bestGrp = c.split, cost(c), c.grp
				}
			}
			sel.Free()
			// The candidates were compiled for the trial alone; the winner is
			// compiled again below. Reductions are the tier's cache and stay.
			for _, c := range cands {
				c.kern.Close()
			}
			if best > 0 && ck != "" {
				gi := 0
				if bestGrp {
					gi = 1
				}
				saveCache("gpusplitid", ck, fmt.Sprintf("%d %d", best, gi))
			}
			if backend.Verbose() && best > 0 {
				fmt.Fprintf(os.Stderr, "tier: %s %dx%d e%dx%d indexed, retuned over %d blocks: split %d%s -> %d%s (%.1f us a launch)\n",
					key.q, key.rows, key.k, key.experts, key.slots, len(rs), us[0].m.split,
					map[bool]string{true: " in-group"}[us[0].m.group],
					best, map[bool]string{true: " in-group"}[bestGrp], bestT*1e6)
			}
		}
		m0 := us[0].m
		if best == 0 || (best == m0.split && bestGrp == m0.group) {
			continue
		}
		// The winner from the device's cache, which Close closes; mk's kernels
		// are the trial's own.
		kern, err := g.idKernel(kernels.MatVecShape{Center: g.center, T: key.q, K: key.k, Rows: key.rows,
			Split: best, Experts: key.experts, Slots: key.slots, SlotAct: key.slotAct, GroupSplit: bestGrp})
		if err != nil {
			continue
		}
		var red backend.Kernel
		if best > 1 && !bestGrp {
			if red = g.reduceKernel(nout, best); red == nil || !g.sizePart(nout*best) {
				continue
			}
		}
		for _, u := range us {
			if bestGrp && !u.m.group {
				g.IndexedGroup++
			}
			u.m.kern, u.m.red, u.m.split, u.m.group = kern, red, best, bestGrp
		}
		changed = true
	}
	return changed
}

// splitCand is one split retuneDecode considers, and t its median seconds a
// launch (0 when it could not be timed).
type splitCand struct {
	kern, red backend.Kernel
	split     int
	grp       bool
	t         float64
}

// timeSequences times every candidate run once over each of rs in turn, the
// candidates interleaved round by round so drift lands on all of them alike.
// Where the device records, each sequence is a captured graph and a trial is
// its replay -- the way a decode token is launched -- so host launch cost does
// not enter the comparison.
//
// sel is the expert selection an INDEXED matvec reads after its output, or nil
// for a dense one; rows is then the launch's output rows, Rows*Slots.
func (g *devTier) timeSequences(cs []splitCand, rs []*resident, rows int, sel backend.Buf) []splitCand {
	// Three replays a sample so a Sync's jitter is small against the work, and
	// seven rounds; the differences being decided are ~1 us a launch.
	const rounds, reps = 7, 3
	out, err := g.dev.Alloc(rows * 4)
	if err != nil {
		return nil
	}
	defer out.Free()
	seq := func(s backend.Session, c splitCand) error {
		for _, r := range rs {
			dst := out
			if c.split > 1 && !c.grp {
				dst = g.partBuf
			}
			args := []backend.Buf{r.qs, r.d, r.sc, g.bs.a, g.bs.ax, dst}
			if sel != nil {
				args = append(args, sel)
			}
			if err := s.Launch(c.kern, groupsOf(rows*c.split, c.split, c.grp), widthOf(c.split, c.grp),
				args...); err != nil {
				return err
			}
			if c.split > 1 && !c.grp {
				if err := s.Launch(c.red, (rows+127)/128, 128, g.partBuf, out); err != nil {
					return err
				}
			}
		}
		return nil
	}
	recs := make([]backend.Recording, len(cs))
	defer func() {
		for _, r := range recs {
			if r != nil {
				r.Free()
			}
		}
	}()
	// A failed capture is freed after the session, never inside it: the
	// callback runs on CUDA's owner goroutine, and cudaGraph.Free posts to
	// that same goroutine and waits, which deadlocks.
	var failed []backend.Recording
	defer func() {
		for _, r := range failed {
			r.Free()
		}
	}()
	g.dev.Session(func(s backend.Session) {
		rec, ok := s.(backend.Recorder)
		if !ok || g.NoGraph {
			return
		}
		for i, c := range cs {
			var inner error
			r, e := backend.Record(rec, func() { inner = seq(s, c) })
			if e == nil && inner == nil {
				recs[i] = r
			} else if r != nil {
				failed = append(failed, r)
			}
		}
	})
	per := make([][]float64, len(cs))
	dead := make([]bool, len(cs))
	for round := 0; round < rounds+1; round++ {
		for i, c := range cs {
			if dead[i] {
				continue
			}
			t0 := time.Now()
			g.dev.Session(func(s backend.Session) {
				for r := 0; r < reps && err == nil; r++ {
					if recs[i] != nil {
						err = s.(backend.Recorder).Replay(recs[i])
					} else {
						err = seq(s, c)
					}
				}
				if err == nil {
					err = s.Sync()
				}
			})
			if err != nil {
				dead[i], err = true, nil
				continue
			}
			if round > 0 { // the first pays first touch
				per[i] = append(per[i], time.Since(t0).Seconds()/float64(reps*len(rs)))
			}
		}
		if round == 1 {
			// A candidate half again slower than the best after one round is
			// not going to win, and a split-1 sequence over a 4096x14336 matrix
			// is 3 ms a replay: stop timing it.
			best := 0.0
			for i := range cs {
				if !dead[i] && (best == 0 || per[i][0] < best) {
					best = per[i][0]
				}
			}
			for i := range cs {
				if !dead[i] && per[i][0] > 1.5*best {
					dead[i] = true
				}
			}
		}
	}
	for i, p := range per {
		if len(p) < rounds {
			continue
		}
		sort.Float64s(p)
		if med := p[len(p)/2]; med > 0 && (p[len(p)*3/4]-p[len(p)/4])/med <= splitTuneIQR {
			cs[i].t = med
		}
		if backend.Verbose() {
			fmt.Fprintf(os.Stderr, "tier: sequence trial rows=%d split %d grp=%v: median %.2f us, IQR/median %.3f\n",
				rows, cs[i].split, cs[i].grp, p[len(p)/2]*1e6, (p[len(p)*3/4]-p[len(p)/4])/p[len(p)/2])
		}
	}
	return cs
}

// binaryID fingerprints the running executable, so a cached split does not
// outlive the kernels it was measured on: any rebuild tunes once again. Empty
// when the executable cannot be read, and then nothing is cached.
var binaryID = sync.OnceValue(func() string {
	p, err := os.Executable()
	if err != nil {
		return ""
	}
	f, err := os.Open(p)
	if err != nil {
		return ""
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return ""
	}
	return fmt.Sprintf("%x", h.Sum(nil)[:8])
})

// fuseMoE builds a mixture block's expert matvecs with their epilogues fused
// -- the idea of llama.cpp's MUL_MAT_ID with glu fusion (mmvq.cu,
// ggml_cuda_mm_fusion_args: the up projection applies the gate, the biases and
// the GLU in its own epilogue) -- or leaves them nil. It needs the final-row
// forms (split 1, or in-group): a partial-writing split has no row to apply an
// epilogue to until its Reduce.
//
// It removes one to four launches a layer: the gated activation always
// (kernels.ActMul over the slots), and with expert biases the three
// kernels.IndexedBiasAdd. A streamed bank is left alone: its weights are
// compacted 0..k-1 while its biases need the true ids.
func (g *devTier) fuseMoE(l *layer) {
	l.moeGate, l.moeUp, l.moeDown = nil, nil, nil
	if g.NoMoEFuse || l.router == nil || l.stream != nil || g.bs == nil {
		return
	}
	// Llama 4 weights the expert's input (ActMulWeighted), which the fused
	// up epilogue does not carry.
	if g.bs.p.ExpertWeightIn {
		return
	}
	act := g.bs.p.Act
	if act != kernels.ActSiLU && act != kernels.ActGELU && act != kernels.ActSwiGLUOAI {
		return
	}
	final := func(m mv) bool {
		return m.kern != nil && m.slots > 0 && m.experts > 1 && m.bias == nil &&
			(m.split <= 1 || m.group) && !kernels.IsFloat(m.q)
	}
	build := func(m mv, bias, gate bool) backend.Kernel {
		s := kernels.MatVecShape{Center: g.center, T: m.q, K: m.k, Rows: m.rows, Split: m.split, GroupSplit: m.group,
			Experts: m.experts, Slots: m.slots, SlotAct: m.slotAct, Bias: bias, Gate: gate, GateAct: act}
		key := fmt.Sprintf("%+v", s)
		if k, ok := g.segKerns[key]; ok {
			return k
		}
		var k backend.Kernel
		ker, err := kernels.MatVec(s)
		if err == nil {
			k, err = g.dev.Compile(ker)
		}
		if err != nil {
			g.LastErr, k = err.Error(), nil
		}
		if g.segKerns == nil {
			g.segKerns = map[string]backend.Kernel{}
		}
		g.segKerns[key] = k
		return k
	}
	// The up projection is the one that must fuse; the gate's bias only
	// matters if it does, since the fused up reads the gate's output as is.
	if !final(l.mvg) || !final(l.mvu) {
		return
	}
	up := build(l.mvu, l.expUpB != nil, true)
	if up == nil {
		return
	}
	var gate backend.Kernel
	if l.expGateB != nil {
		if gate = build(l.mvg, true, false); gate == nil {
			return
		}
	}
	l.moeGate, l.moeUp = gate, up
	if l.expDownB != nil && final(l.mvd) {
		l.moeDown = build(l.mvd, true, false)
	}
}

// warpNorm reports whether the whole-row RMSNorm kernels take the
// subgroup-shuffle reduction (kernels.RMSNormRowsWarp): a device that
// guarantees 32 lanes AND whose subgroups are contiguous runs of 32 threads,
// which CUDA's warps are by architecture and nothing promises elsewhere.
// Config.NoWarpNorm keeps the shared-memory tree.
func (g *devTier) warpNorm() bool {
	return !g.NoWarpNorm && g.dev.API() == "ptx" && g.pickLanes() == ir.SubgroupLanes
}

func (g *devTier) rmsRows() func(k, rows int, eps float32, addOne, add bool) (*ir.Kernel, error) {
	if g.warpNorm() {
		return kernels.RMSNormRowsWarp
	}
	return kernels.RMSNormRows
}

func (g *devTier) rmsQuantRows() func(k, rows int, eps float32, addOne, add bool, window int) (*ir.Kernel, error) {
	if g.warpNorm() {
		return kernels.RMSNormQuantRowsWarp
	}
	return kernels.RMSNormQuantRows
}

// segLaunch is a kernels.MatVecSegments launch: the kernel and its grid.
type segLaunch struct {
	kern      backend.Kernel
	blocks, w int
}

// fuseSegs builds one launch for several plain, unbiased decode matvecs over
// the SAME activation, at the first one's split -- fuseQKV's recipe for any
// such group. nil where they cannot share a launch. A pair saves a launch
// each time and the small member rides the big one's grid (MLA's 576-row kv_a
// alone reads far below the bandwidth wall).
func (g *devTier) fuseSegs(ms []mv) *segLaunch {
	if len(ms) < 2 || g.NoSegFuse {
		return nil
	}
	m0 := ms[0]
	if m0.split > 1 && !m0.group {
		return nil
	}
	var segs []kernels.MatVecShape
	for _, m := range ms {
		if m.kern == nil || m.slots > 0 || m.groups > 0 || m.threads > 0 || m.rowt > 1 ||
			m.k != m0.k || m.bias != nil || kernels.IsFloat(m.q) {
			return nil
		}
		segs = append(segs, kernels.MatVecShape{Center: g.center, T: m.q, K: m.k, Rows: m.rows})
	}
	blocks, w, err := kernels.SegmentsGrid(segs, m0.split, m0.group)
	if err != nil {
		return nil
	}
	key := fmt.Sprintf("%v/%d/%v", segs, m0.split, m0.group)
	k, ok := g.segKerns[key]
	if !ok {
		ker, err := kernels.MatVecSegments(segs, m0.split, m0.group)
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
	if k == nil {
		return nil
	}
	n := 0
	for _, b := range blocks {
		n += b
	}
	return &segLaunch{kern: k, blocks: n, w: w}
}

// widthOf is a matvec's workgroup width, and groupsOf the groups that cover
// threads at it; see kernels.GroupSplitWidth.
func widthOf(split int, grp bool) int {
	if grp {
		return kernels.GroupSplitWidth(split)
	}
	return 128
}

func groupsOf(threads, split int, grp bool) int {
	w := widthOf(split, grp)
	return (threads + w - 1) / w
}

// fuseQKV builds block l's q/k/v launch (kernels.MatVecSegments) at the split
// its q projection runs, or leaves l.qkv nil where the three cannot share one:
// a batched, indexed, tiled or float matvec, a split that needs a separate
// reduction, or projections that read different activations. MLA, recurrent
// and vision blocks never reach it with three plain decode matvecs.
func (g *devTier) fuseQKV(l *layer) {
	g.fuseMoE(l)
	l.mlaQK, l.shGU = nil, nil
	if l.mla {
		q, wq := l.mvq, l.wq
		if l.wqa != nil {
			q, wq = l.mvQA, l.wqa
		}
		if wq != nil && l.wkva != nil {
			l.mlaQK = g.fuseSegs([]mv{q, l.mvKVA})
		}
	}
	if l.shGate != nil && l.shUp != nil {
		l.shGU = g.fuseSegs([]mv{l.mvsg, l.mvsu})
	}
	l.qkv = nil
	if l.mla || l.linear || l.nonCausal || l.wq == nil || l.wk == nil || l.wv == nil {
		return
	}
	q := l.mvq
	if q.split > 1 && !q.group {
		return
	}
	var segs []kernels.MatVecShape
	for _, m := range []mv{l.mvq, l.mvk, l.mvv} {
		if m.kern == nil || m.slots > 0 || m.groups > 0 || m.threads > 0 || m.rowt > 1 || m.k != q.k {
			return
		}
		segs = append(segs, kernels.MatVecShape{Center: g.center, T: m.q, K: m.k, Rows: m.rows, Bias: m.bias != nil})
	}
	blocks, w, err := kernels.SegmentsGrid(segs, q.split, q.group)
	if err != nil {
		return
	}
	key := fmt.Sprintf("%v/%d/%v", segs, q.split, q.group)
	k, ok := g.segKerns[key]
	if !ok {
		ker, err := kernels.MatVecSegments(segs, q.split, q.group)
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
	if k == nil {
		return
	}
	n := 0
	for _, b := range blocks {
		n += b
	}
	l.qkv, l.qkvBlocks, l.qkvW = k, n, w
	// The launch appends a buffer per bias it finds, and CUDA does not check the
	// count; it runs this kernel only while these still match.
	l.qkvBias = [3]bool{segs[0].Bias, segs[1].Bias, segs[2].Bias}
}
