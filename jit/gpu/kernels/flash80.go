package kernels

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/gpu/ir"
)

// FlashPrefill80Rows is how many query rows one FlashPrefill80 workgroup
// takes: four warps of sixteen.
const FlashPrefill80Rows = 16 * flashWarps

// FlashPrefill80Threads is FlashPrefill80's workgroup width.
const FlashPrefill80Threads = 32 * flashWarps

// FlashPrefill80Groups is how many workgroups FlashPrefill80 launches.
func FlashPrefill80Groups(s FlashPrefill70Shape) int {
	return s.Heads * (s.Rows / FlashPrefill80Rows) * flashPrefillSplits(s)
}

// FlashPrefill80WhyNot is why FlashPrefill80 does not take a shape, or "".
// A head wider than 128 is the one geometry it does not express yet: its
// output accumulators alone (head/8 n-tiles of four) would pass the
// register file's share a thread can hold without spilling.
func FlashPrefill80WhyNot(s FlashPrefill70Shape) string {
	switch {
	case s.KVHeads <= 0 || s.Heads%s.KVHeads != 0:
		return fmt.Sprintf("%d heads over %d kv heads", s.Heads, s.KVHeads)
	case s.Dim%32 != 0 || s.Dim < 32 || s.Dim > 128:
		return fmt.Sprintf("head width %d (a multiple of 32 up to 128)", s.Dim)
	case s.Rows <= 0 || s.Rows%FlashPrefill80Rows != 0:
		return fmt.Sprintf("%d rows, want a multiple of %d", s.Rows, FlashPrefill80Rows)
	case s.Page <= 0:
		return "the contiguous cache (FlashPrefill80 reads paged KV)"
	case s.Softcap < 0:
		return fmt.Sprintf("softcap %v", s.Softcap)
	}
	if err := flashPrefillCheck("FlashPrefill80", s); err != nil {
		return err.Error()
	}
	return ""
}

// FlashPrefill80 is the batched attention of a prompt chunk over paged KV in
// one kernel on the m16n8k16 binary16 instruction (sm_80 on): scores, a
// running softmax and the weighted sum of V per 32-key tile, the
// probabilities never leaving registers. It replaces PagedAttnScoresMMA,
// PagedPrefillSoftmax and PagedAttnAccMMA, which wrote and re-read a rows x
// heads x keys float32 score plane and a weight plane between them, and it
// is FlashPrefill70's algorithm in FlashAttention-2's layout: queries are the
// A rows (sixteen a warp, loaded once into registers), the tile's keys the B
// columns, so the score fragment of two 8-key n-tiles is the A fragment of
// the 16-key step of P times V -- row g's keys 2q, 2q+1 and their +8 -- with
// no shared-memory round trip. A row's maximum and sum reduce over the four
// lanes of its quad (lanes xor 1 and 2).
//
// K is staged as binary16 key rows and V transposed as dim rows of key
// pairs, both padded four words so a fragment load's 32 lanes hit 32 banks.
// The per-element transform is the staged kernels': the scale, the
// attention softcap, then -inf for a key outside the row's [keyStart,
// keyEnd) -- the causal end, the window's start or a bidirectional run's
// count, all the descriptor's. A V row outside the workgroup's keys is
// staged as zero: the page there holds nothing written, and 0 * NaN is NaN.
//
// Parameters as FlashPrefill70's paged form: pQ ([row][head][dim] f32), pK
// and pV (the paged pools), pN (not read), pOut, pTab, pRow. Splits > 0
// writes FlashAttentionMerge's partials (each query's sums, maximum and
// sum), which also applies sinks; Splits 0 writes the output. Launch
// FlashPrefill80Groups groups of FlashPrefill80Threads.
func FlashPrefill80(s FlashPrefill70Shape) (*ir.Kernel, error) {
	if why := FlashPrefill80WhyNot(s); why != "" {
		return nil, fmt.Errorf("kernels: FlashPrefill80: %s", why)
	}
	hd := s.Dim
	P, S := s.Page, flashPrefillSplits(s)
	const tk = 32 // keys a tile
	const qw = 16 // queries a warp
	gqa := s.Heads / s.KVHeads
	kvDim := s.KVHeads * hd
	qs := hd/2 + 4 // u32 stride of a staged K row: four apart, a fragment load conflict-free
	const vs = tk/2 + 4
	nd := hd / 8 // output n-tiles
	ksteps := hd / 16
	sh := ir.MMAShape{M: 16, N: 8, K: 16, Kind: ir.MMAF16}

	b := ir.New(fmt.Sprintf("flashprefill80paged_d%d_s%d", hd, s.Splits), [3]int{FlashPrefill80Threads, 1, 1})
	pQ := b.Param("pQ", ir.F32)
	pK := b.Param("pK", ir.F32)
	pV := b.Param("pV", ir.F32)
	b.Param("pN", ir.U32)
	pOut := b.Param("pOut", ir.F32)
	pTab, pDesc := b.Param("pTab", ir.U32), b.Param("pRow", ir.U32)
	c := func(v int64) ir.Value { return b.Const(ir.U32, v) }
	f := b.ConstF32
	negInf := b.Bitcast(ir.F32, c(0xFF800000))
	zero := f(0)
	zeroU := c(0)

	shK := b.Shared("fp8K", ir.U32, tk*qs)
	shV := b.Shared("fp8V", ir.U32, hd*vs)

	tid := b.TID()
	wg := b.CTAID()
	lane := b.And(ir.U32, tid, c(31))
	warp := b.Shr(ir.U32, tid, c(5))
	lg := b.Shr(ir.U32, lane, c(2)) // the A row (and +8), the B column
	lt := b.And(ir.U32, lane, c(3)) // the k pair
	spl := urem(b, wg, S)           // the split varies fastest, then the head
	wg = udiv(b, wg, S)
	h := b.Rem(ir.U32, wg, c(int64(s.Heads)))
	qb := b.Div(ir.U32, wg, c(int64(s.Heads)))
	kvBase := b.Mul(ir.U32, b.Div(ir.U32, h, c(int64(gqa))), c(int64(hd)))
	row0 := b.Mul(ir.U32, qb, c(FlashPrefill80Rows))
	wrow0 := b.Add(ir.U32, row0, b.Mul(ir.U32, warp, c(qw)))
	// The workgroup's keys, [keyStart of its first row, keyEnd of its last),
	// in 32-key tiles from a multiple of 32, which lie in one page each; this
	// split's run of them.
	span := tileSpanOf(b, pDesc, row0, b.Add(ir.U32, row0, c(FlashPrefill80Rows-1)))
	kStart, ntiles := span.tileRun(b, spl, S, tk)
	hlo := b.Sub(ir.U32, span.hi, span.lo)

	// The lane's two query rows and their keys.
	var rows, kst, kend [2]ir.Value
	for x := 0; x < 2; x++ {
		rows[x] = b.Add(ir.U32, wrow0, b.Add(ir.U32, lg, c(int64(8*x))))
		kst[x], kend[x] = pagedDesc(b, pDesc, rows[x], PRowStart), pagedDesc(b, pDesc, rows[x], PRowEnd)
	}
	// The queries, once, as A fragments: rows g and g+8, dims 2q.. and +8 of
	// each 16-dim step.
	qa := make([][]ir.Value, ksteps)
	qAt := func(x int) ir.Value {
		return b.Add(ir.U32, b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, rows[x], c(int64(s.Heads))), h), c(int64(hd))),
			b.Shl(ir.U32, lt, c(1)))
	}
	q0, q1 := qAt(0), qAt(1)
	for ks := range qa {
		pk := func(base ir.Value, off int) ir.Value {
			e := b.LoadV(ir.F32, pQ, base, int64(16*ks+off), 2)
			return b.PackF16(e[0], e[1])
		}
		qa[ks] = []ir.Value{pk(q0, 0), pk(q1, 0), pk(q0, 8), pk(q1, 8)}
	}

	type state struct {
		o    [][]ir.Value
		m, l [2]ir.Value
	}
	newState := func(from *state) state {
		var t state
		for n := 0; n < nd; n++ {
			o := make([]ir.Value, 4)
			for i := range o {
				init := zero
				if from != nil {
					init = from.o[n][i]
				}
				o[i] = b.Phi(ir.F32, init)
			}
			t.o = append(t.o, o)
		}
		for x := 0; x < 2; x++ {
			mi, li := negInf, zero
			if from != nil {
				mi, li = from.m[x], from.l[x]
			}
			t.m[x], t.l[x] = b.Phi(ir.F32, mi), b.Phi(ir.F32, li)
		}
		return t
	}
	// The run's tiles a page at a time: an outer loop over the pages it
	// touches reads each page's id once, and the tiles inside it need no
	// lookup.
	kEnd := b.Add(ir.U32, kStart, b.Mul(ir.U32, ntiles, c(tk)))
	pg0 := pageDiv(b, kStart, P)
	npg := b.Select(ir.U32, b.Lt(ir.U32, zeroU, ntiles),
		b.Add(ir.U32, b.Sub(ir.U32, pageDiv(b, b.Sub(ir.U32, kEnd, c(1)), P), pg0), c(1)), zeroU)
	b.LoopN(npg)
	outer := newState(nil)
	pg := b.Phi(ir.U32, pg0)
	t0 := b.Max(ir.U32, kStart, b.Mul(ir.U32, pg, c(int64(P))))
	pid := pageOf(b, pTab, span.tab, t0, P)
	t1 := b.Min(ir.U32, kEnd, b.Mul(ir.U32, b.Add(ir.U32, pg, c(1)), c(int64(P))))
	b.LoopN(udiv(b, b.Sub(ir.U32, t1, t0), tk))
	st := newState(&outer)
	k0 := b.Phi(ir.U32, t0)
	o, m, l := st.o, st.m, st.l

	// Stage the tile. The barrier first: the last tile's fragments were being
	// read out of these same arrays.
	nth := int64(FlashPrefill80Threads)
	b.Barrier()
	for it := int64(0); it < int64(tk*hd/2)/nth; it++ {
		w := b.Add(ir.U32, tid, c(nth*it))
		key := b.And(ir.U32, w, c(tk-1)) // consecutive threads, consecutive keys: coalesced
		p := b.Shr(ir.U32, w, c(5))
		// A page's K is [kvRow][P]: the tile's keys are adjacent.
		e := b.Add(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, pid, c(int64(P*kvDim))),
			b.Mul(ir.U32, b.Add(ir.U32, kvBase, b.Shl(ir.U32, p, c(1))), c(int64(P)))), b.Add(ir.U32, pageRem(b, k0, P), key))
		lo, hi := b.Load(ir.F32, pK, e, 0), b.Load(ir.F32, pK, e, int64(P))
		b.Store(shK, b.Add(ir.U32, b.Mul(ir.U32, key, c(int64(qs))), p), b.PackF16(lo, hi), 0)
	}
	// V four dims at a time: two positions' 16-byte loads, four packed words
	// of key pairs into the dims' rows.
	for it := int64(0); it < int64(tk/2*hd/4)/nth; it++ {
		w := b.Add(ir.U32, tid, c(nth*it))
		dq := b.Rem(ir.U32, w, c(int64(hd/4))) // consecutive threads, consecutive dims: coalesced
		kp := b.Div(ir.U32, w, c(int64(hd/4)))
		p0 := b.Add(ir.U32, k0, b.Shl(ir.U32, kp, c(1)))
		p1 := b.Add(ir.U32, p0, c(1))
		// Both positions are in the tile's page; one outside the workgroup's
		// keys is a slot nothing wrote.
		ok0, ok1 := b.Lt(ir.U32, b.Sub(ir.U32, p0, span.lo), hlo), b.Lt(ir.U32, b.Sub(ir.U32, p1, span.lo), hlo)
		if pagedFault == "vmask" {
			ok0, ok1 = span.in(b, p0), span.in(b, p1)
		}
		col := b.Add(ir.U32, kvBase, b.Shl(ir.U32, dq, c(2)))
		v0, v1 := pagedV4(b, pV, pid, p0, col, P, kvDim, s.F16), pagedV4(b, pV, pid, p1, col, P, kvDim, s.F16)
		base := b.Add(ir.U32, b.Mul(ir.U32, b.Shl(ir.U32, dq, c(2)), c(vs)), kp)
		for x := 0; x < 4; x++ {
			b.Store(shV, base, b.PackF16(b.Select(ir.F32, ok0, v0[x], zero), b.Select(ir.F32, ok1, v1[x], zero)), int64(x*vs))
		}
	}
	b.Barrier()

	// S = Q K^T over the tile: n-tile j is keys 8j..8j+7, lane column lg.
	sacc := make([][]ir.Value, tk/8)
	for j := range sacc {
		sacc[j] = []ir.Value{zero, zero, zero, zero}
		kRow := b.Add(ir.U32, b.Mul(ir.U32, b.Add(ir.U32, lg, c(int64(8*j))), c(int64(qs))), lt)
		for ks := 0; ks < ksteps; ks++ {
			bf := []ir.Value{b.Load(ir.U32, shK, kRow, int64(8*ks)), b.Load(ir.U32, shK, kRow, int64(8*ks+4))}
			sacc[j] = b.MMA(sh, qa[ks], bf, sacc[j])
		}
	}
	// The transform, then each row's running maximum and sum over its quad.
	sc := f(s.Scale)
	kq := b.Add(ir.U32, k0, b.Shl(ir.U32, lt, c(1)))
	val := make([][]ir.Value, len(sacc))
	for j := range sacc {
		val[j] = make([]ir.Value, 4)
		for x := 0; x < 4; x++ {
			v := b.Mul(ir.F32, sacc[j][x], sc)
			if s.Softcap > 0 {
				e := b.Exp(b.Mul(ir.F32, f(2/s.Softcap), v))
				v = b.Mul(ir.F32, f(s.Softcap), b.Sub(ir.F32, f(1), b.Div(ir.F32, f(2), b.Add(ir.F32, e, f(1)))))
			}
			key := b.Add(ir.U32, kq, c(int64(8*j+x&1)))
			r := x >> 1
			val[j][x] = b.Select(ir.F32, rowAttends(b, key, kst[r], kend[r]), v, negInf)
		}
	}
	var mNext, lNext, alpha [2]ir.Value
	pf := make([][]ir.Value, len(sacc))
	for j := range pf {
		pf[j] = make([]ir.Value, 4)
	}
	for r := 0; r < 2; r++ {
		t := negInf
		for j := range val {
			t = b.Max(ir.F32, t, b.Max(ir.F32, val[j][2*r], val[j][2*r+1]))
		}
		for _, mk := range []int64{1, 2} {
			t = b.Max(ir.F32, t, b.ShuffleXor(ir.F32, t, mk))
		}
		mn := b.Max(ir.F32, m[r], t)
		// A row with no key yet stays at -inf; exp against a finite stand-in
		// keeps -inf - -inf out of it.
		mf := b.Max(ir.F32, mn, f(-1e30))
		alpha[r] = b.Exp(b.Sub(ir.F32, m[r], mf))
		sum := b.Mul(ir.F32, l[r], alpha[r])
		for j := range val {
			for x := 2 * r; x < 2*r+2; x++ {
				pf[j][x] = b.Exp(b.Sub(ir.F32, val[j][x], mf))
				sum = b.Add(ir.F32, sum, pf[j][x])
			}
		}
		mNext[r], lNext[r] = mn, sum
	}
	// O = O*alpha + P V: P's A fragment for the 16-key step kk is n-tiles 2kk
	// and 2kk+1 of the scores as they sit; V's B fragment is dim column lg,
	// key pairs q and q+4 of the step.
	od := make([][]ir.Value, nd)
	for n := range od {
		od[n] = make([]ir.Value, 4)
		for x := 0; x < 4; x++ {
			od[n][x] = b.Mul(ir.F32, o[n][x], alpha[x>>1])
		}
	}
	for kk := 0; kk < tk/16; kk++ {
		a := []ir.Value{
			b.PackF16(pf[2*kk][0], pf[2*kk][1]), b.PackF16(pf[2*kk][2], pf[2*kk][3]),
			b.PackF16(pf[2*kk+1][0], pf[2*kk+1][1]), b.PackF16(pf[2*kk+1][2], pf[2*kk+1][3]),
		}
		for n := 0; n < nd; n++ {
			vRow := b.Add(ir.U32, b.Mul(ir.U32, b.Add(ir.U32, lg, c(int64(8*n))), c(vs)), lt)
			bf := []ir.Value{b.Load(ir.U32, shV, vRow, int64(8*kk)), b.Load(ir.U32, shV, vRow, int64(8*kk+4))}
			od[n] = b.MMA(sh, a, bf, od[n])
		}
	}
	for n := range o {
		for x := range o[n] {
			b.SetPhi(o[n][x], od[n][x])
		}
	}
	for r := 0; r < 2; r++ {
		b.SetPhi(m[r], mNext[r])
		b.SetPhi(l[r], lNext[r])
	}
	b.SetPhi(k0, b.Add(ir.U32, k0, c(tk)))
	b.EndLoop()
	for n := range o {
		for x := range o[n] {
			b.SetPhi(outer.o[n][x], o[n][x])
		}
	}
	for r := 0; r < 2; r++ {
		b.SetPhi(outer.m[r], m[r])
		b.SetPhi(outer.l[r], l[r])
	}
	b.SetPhi(pg, b.Add(ir.U32, pg, c(1)))
	b.EndLoop()
	o, m, l = outer.o, outer.m, outer.l

	// From the phis: a workgroup whose keys are none runs no tile, and a body
	// value is then a register nothing wrote. A row's sum is its quad's.
	var lsum [2]ir.Value
	for r := 0; r < 2; r++ {
		t := l[r]
		for _, mk := range []int64{1, 2} {
			t = b.Add(ir.F32, t, b.ShuffleXor(ir.F32, t, mk))
		}
		lsum[r] = t
	}
	dimOf := func(n, x int) ir.Value { return b.Add(ir.U32, b.Shl(ir.U32, lt, c(1)), c(int64(8*n+x&1))) }
	if s.Splits > 0 {
		// This run's partials for FlashAttentionMerge: the unnormalised sums,
		// and each query's maximum (the finite stand-in its exponentials were
		// taken against) and sum. The four lanes holding a row write eight of
		// its 32 replicas each.
		stride := int64(hd + 64)
		pbOf := func(qr ir.Value) ir.Value {
			return b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, qr, c(int64(s.Heads))), h),
				c(int64(S))), spl), c(stride))
		}
		for r := 0; r < 2; r++ {
			partialML(b, pOut, pbOf(rows[r]), b.Max(ir.F32, m[r], f(-1e30)), lsum[r], lt, hd, 4)
		}
		for n := 0; n < nd; n++ {
			for x := 0; x < 4; x++ {
				b.Store(pOut, b.Add(ir.U32, pbOf(rows[x>>1]), dimOf(n, x)), o[n][x], 0)
			}
		}
		return b.Done(), nil
	}
	var inv [2]ir.Value
	for r := 0; r < 2; r++ {
		inv[r] = b.Div(ir.F32, f(1), b.Max(ir.F32, lsum[r], f(1e-30)))
	}
	for n := 0; n < nd; n++ {
		for x := 0; x < 4; x++ {
			at := b.Add(ir.U32, b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, rows[x>>1], c(int64(s.Heads))), h), c(int64(hd))), dimOf(n, x))
			b.Store(pOut, at, b.Mul(ir.F32, o[n][x], inv[x>>1]), 0)
		}
	}
	return b.Done(), nil
}
