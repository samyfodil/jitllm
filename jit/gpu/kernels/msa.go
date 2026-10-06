package kernels

import (
	"fmt"

	"github.com/samyfodil/jitllm/jit/gpu/ir"
)

// MiniMax Sparse Attention on a device (engine/model/msa.go has the graph).
// The indexer's key is cached as one more kv head of the paged pool's row, at
// column koff of a kvRow-wide K row ([kvRow][P] a page, paged.go), and each
// query row's history is its descriptor's [0, PRowEnd) through the page
// table. Seven basic kernels, each a loop of depth one:
//
//	MSAScoresPaged  score[r][h][t] = q[r][h] . k_t, per kv group h
//	MSABlockMax     each block's best score; the query's own IdxLocal blocks
//	                +inf, blocks past the history -inf
//	MSASelect       the IdxTopK best blocks of each (row, group), as a flag
//	MSACompact      the kept blocks in ascending order
//	MSAGatherK/V    each group's kept positions' K and V into scratch pages
//	MSADesc         the gathered pages' descriptors
//
// after which the attention runs unchanged over the gathered pages: kv head h
// of gathered position j of row r is the kept position j of group h. Every
// group keeps the same count (the query's own block is forced in and is the
// only partial one, and the last in ascending order), so one descriptor a row
// serves every head. Ties between blocks go to the lower block, as the host's
// ordering does.

// MSAScoresPaged scores every position of each row's history in each group.
// One thread per (row, group, position) up to tcap positions a row; a position
// past the row's end scores its last position (a clamp, never a read past the
// history), which MSABlockMax does not read. Parameters: pQ (rows x heads x
// dim, rotated), pK (the pool), pTab, pRow, pOut (rows x heads x tcap).
func MSAScoresPaged(rows, heads, dim, tcap, page, kvRow, koff int) (*ir.Kernel, error) {
	if err := checkPage("MSAScoresPaged", page); err != nil {
		return nil, err
	}
	if rows < 1 || heads < 1 || dim < 1 || tcap < 1 || koff < 0 || koff+dim > kvRow {
		return nil, fmt.Errorf("kernels: MSAScoresPaged: rows=%d heads=%d dim=%d tcap=%d row=%d col=%d",
			rows, heads, dim, tcap, kvRow, koff)
	}
	b := ir.New("msascores", [3]int{128, 1, 1})
	pQ := b.Param("pQ", ir.F32)
	pK := b.Param("pK", ir.F32)
	pTab := b.Param("pTab", ir.U32)
	pRow := b.Param("pRow", ir.U32)
	pOut := b.Param("pOut", ir.F32)
	flat := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(rows*heads*tcap-1)))
	one := b.Const(ir.U32, 1)
	t := b.Rem(ir.U32, flat, b.Const(ir.U32, int64(tcap)))
	rh := b.Div(ir.U32, flat, b.Const(ir.U32, int64(tcap)))
	r := b.Div(ir.U32, rh, b.Const(ir.U32, int64(heads)))
	end := b.Max(ir.U32, pagedDesc(b, pRow, r, PRowEnd), one)
	t = b.Min(ir.U32, t, b.Sub(ir.U32, end, one))
	pid := pageOf(b, pTab, pagedDesc(b, pRow, r, PRowTab), t, page)
	// K is [kvRow][P] a page: column e of position t at pid*P*kvRow + e*P + t%P.
	kb := b.Add(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, pid, b.Const(ir.U32, int64(page*kvRow))),
		b.Const(ir.U32, int64(koff*page))), pageRem(b, t, page))
	qb := b.Mul(ir.U32, rh, b.Const(ir.U32, int64(dim)))
	zu, zf := b.Const(ir.U32, 0), b.ConstF32(0)
	b.Loop(int64(dim))
	i := b.Phi(ir.U32, zu)
	acc := b.Phi(ir.F32, zf)
	k := b.Load(ir.F32, pK, b.Add(ir.U32, kb, b.Mul(ir.U32, i, b.Const(ir.U32, int64(page)))), 0)
	b.SetPhi(acc, b.Fma(b.Load(ir.F32, pQ, b.Add(ir.U32, qb, i), 0), k, acc))
	b.SetPhi(i, b.Add(ir.U32, i, one))
	b.EndLoop()
	b.Store(pOut, flat, acc, 0)
	return b.Done(), nil
}

// MSABlocks is how many blocks of blk positions tcap positions span.
func MSABlocks(tcap, blk int) int { return (tcap + blk - 1) / blk }

// msaOwn is the query's own block: the block of its row's last position.
func msaOwn(b *ir.Builder, pRow, r ir.Value, blk int) (end, own ir.Value) {
	one := b.Const(ir.U32, 1)
	end = b.Max(ir.U32, pagedDesc(b, pRow, r, PRowEnd), one)
	return end, b.Div(ir.U32, b.Sub(ir.U32, end, one), b.Const(ir.U32, int64(blk)))
}

// MSABlockMax ranks each block of each row's history in each group by its
// best position: +inf for the query's own local blocks (own-local < b <=
// own, the reference's clamped indices as a set), -inf for a block past the
// query's own, and the max of the block's scores otherwise. One thread per
// (row, group, block). Parameters: pScore (rows x heads x tcap), pRow, pOut
// (rows x heads x MSABlocks(tcap, blk)).
func MSABlockMax(rows, heads, tcap, blk, local int) (*ir.Kernel, error) {
	if rows < 1 || heads < 1 || tcap < 1 || blk < 1 || local < 1 {
		return nil, fmt.Errorf("kernels: MSABlockMax: rows=%d heads=%d tcap=%d blk=%d local=%d",
			rows, heads, tcap, blk, local)
	}
	nb := MSABlocks(tcap, blk)
	b := ir.New("msablockmax", [3]int{128, 1, 1})
	pScore := b.Param("pScore", ir.F32)
	pRow := b.Param("pRow", ir.U32)
	pOut := b.Param("pOut", ir.F32)
	flat := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(rows*heads*nb-1)))
	one := b.Const(ir.U32, 1)
	bi := b.Rem(ir.U32, flat, b.Const(ir.U32, int64(nb)))
	rh := b.Div(ir.U32, flat, b.Const(ir.U32, int64(nb)))
	r := b.Div(ir.U32, rh, b.Const(ir.U32, int64(heads)))
	end, own := msaOwn(b, pRow, r, blk)
	negInf := b.Bitcast(ir.F32, b.Const(ir.U32, 0xFF800000))
	posInf := b.Bitcast(ir.F32, b.Const(ir.U32, 0x7F800000))
	t0 := b.Mul(ir.U32, bi, b.Const(ir.U32, int64(blk)))
	sb := b.Mul(ir.U32, rh, b.Const(ir.U32, int64(tcap)))
	last := b.Const(ir.U32, int64(tcap-1))
	zu := b.Const(ir.U32, 0)
	b.Loop(int64(blk))
	i := b.Phi(ir.U32, zu)
	mx := b.Phi(ir.F32, negInf)
	t := b.Add(ir.U32, t0, i)
	s := b.Load(ir.F32, pScore, b.Add(ir.U32, sb, b.Min(ir.U32, t, last)), 0)
	b.SetPhi(mx, b.Select(ir.F32, b.Lt(ir.U32, t, end), b.Max(ir.F32, mx, s), mx))
	b.SetPhi(i, b.Add(ir.U32, i, one))
	b.EndLoop()
	// own - b < local, i.e. b + local > own, for a block at or below own.
	forced := b.Lt(ir.U32, own, b.Add(ir.U32, bi, b.Const(ir.U32, int64(local))))
	v := b.Select(ir.F32, forced, posInf, mx)
	v = b.Select(ir.F32, b.Lt(ir.U32, own, bi), negInf, v)
	b.Store(pOut, flat, v, 0)
	return b.Done(), nil
}

// MSASelect flags each row's and group's kept blocks: block b is kept when it
// is at or below the query's own block and fewer than min(k, own+1) blocks
// rank above it -- a higher score, or an equal one at a lower block. One
// thread per (row, group, block). Parameters: pBlk (MSABlockMax's), pRow,
// pFlag (rows x heads x blocks, u32 0 or 1).
func MSASelect(rows, heads, tcap, blk, k int) (*ir.Kernel, error) {
	if rows < 1 || heads < 1 || tcap < 1 || blk < 1 || k < 1 {
		return nil, fmt.Errorf("kernels: MSASelect: rows=%d heads=%d tcap=%d blk=%d k=%d",
			rows, heads, tcap, blk, k)
	}
	nb := MSABlocks(tcap, blk)
	b := ir.New("msaselect", [3]int{128, 1, 1})
	pBlk := b.Param("pBlk", ir.F32)
	pRow := b.Param("pRow", ir.U32)
	pFlag := b.Param("pFlag", ir.U32)
	flat := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(rows*heads*nb-1)))
	zero, one := b.Const(ir.U32, 0), b.Const(ir.U32, 1)
	bi := b.Rem(ir.U32, flat, b.Const(ir.U32, int64(nb)))
	rh := b.Div(ir.U32, flat, b.Const(ir.U32, int64(nb)))
	r := b.Div(ir.U32, rh, b.Const(ir.U32, int64(heads)))
	_, own := msaOwn(b, pRow, r, blk)
	n := b.Min(ir.U32, b.Add(ir.U32, own, one), b.Const(ir.U32, int64(nb)))
	keep := b.Min(ir.U32, n, b.Const(ir.U32, int64(k)))
	base := b.Mul(ir.U32, rh, b.Const(ir.U32, int64(nb)))
	sv := b.Load(ir.F32, pBlk, flat, 0)
	b.LoopN(n)
	u := b.Phi(ir.U32, zero)
	rank := b.Phi(ir.U32, zero)
	su := b.Load(ir.F32, pBlk, b.Add(ir.U32, base, u), 0)
	before := b.Select(ir.U32, b.Lt(ir.F32, sv, su), one,
		b.Select(ir.U32, b.Lt(ir.F32, su, sv), zero, b.Select(ir.U32, b.Lt(ir.U32, u, bi), one, zero)))
	b.SetPhi(rank, b.Add(ir.U32, rank, before))
	b.SetPhi(u, b.Add(ir.U32, u, one))
	b.EndLoop()
	kept := b.Select(ir.U32, b.Lt(ir.U32, rank, keep), b.Select(ir.U32, b.Lt(ir.U32, bi, n), one, zero), zero)
	b.Store(pFlag, flat, kept, 0)
	return b.Done(), nil
}

// MSACompact lists each row's and group's kept blocks in ascending order: a
// kept block b lands in slot (kept blocks below b) of the k+1 slots, and a
// block not kept in the spare slot k. One thread per (row, group, block).
// Parameters: pFlag (MSASelect's), pSel (rows x heads x (k+1), u32).
func MSACompact(rows, heads, tcap, blk, k int) (*ir.Kernel, error) {
	if rows < 1 || heads < 1 || tcap < 1 || blk < 1 || k < 1 {
		return nil, fmt.Errorf("kernels: MSACompact: rows=%d heads=%d tcap=%d blk=%d k=%d",
			rows, heads, tcap, blk, k)
	}
	nb := MSABlocks(tcap, blk)
	b := ir.New("msacompact", [3]int{128, 1, 1})
	pFlag := b.Param("pFlag", ir.U32)
	pSel := b.Param("pSel", ir.U32)
	flat := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(rows*heads*nb-1)))
	zero, one := b.Const(ir.U32, 0), b.Const(ir.U32, 1)
	bi := b.Rem(ir.U32, flat, b.Const(ir.U32, int64(nb)))
	rh := b.Div(ir.U32, flat, b.Const(ir.U32, int64(nb)))
	base := b.Mul(ir.U32, rh, b.Const(ir.U32, int64(nb)))
	b.LoopN(bi)
	u := b.Phi(ir.U32, zero)
	slot := b.Phi(ir.U32, zero)
	b.SetPhi(slot, b.Add(ir.U32, slot, b.Load(ir.U32, pFlag, b.Add(ir.U32, base, u), 0)))
	b.SetPhi(u, b.Add(ir.U32, u, one))
	b.EndLoop()
	kk := b.Const(ir.U32, int64(k))
	mine := b.Load(ir.U32, pFlag, flat, 0)
	at := b.Select(ir.U32, b.Lt(ir.U32, zero, mine), b.Min(ir.U32, slot, kk), kk)
	b.Store(pSel, b.Add(ir.U32, b.Mul(ir.U32, rh, b.Const(ir.U32, int64(k+1))), at), bi, 0)
	return b.Done(), nil
}

// MSAGatherPages is the gathered pages a row: k blocks of blk keys, pages of P.
func MSAGatherPages(k, blk, page int) int { return (k*blk + page - 1) / page }

// msaKept is a row's kept positions, the same in every group: every kept
// block whole but the query's own, which is the last and ends at the
// history's end.
func msaKept(b *ir.Builder, end, own ir.Value, blk, k int) ir.Value {
	one := b.Const(ir.U32, 1)
	nsel := b.Min(ir.U32, b.Add(ir.U32, own, one), b.Const(ir.U32, int64(k)))
	bl := b.Const(ir.U32, int64(blk))
	return b.Add(ir.U32, b.Mul(ir.U32, b.Sub(ir.U32, nsel, one), bl), b.Sub(ir.U32, end, b.Mul(ir.U32, own, bl)))
}

// msaGatherPos is the history position gathered key j of row r's group h
// reads, and the source page id holding it: its kept block's position, clamped
// to the kept count so a slot past it reads the last kept key, which no
// descriptor reads.
func msaGatherPos(b *ir.Builder, pTab, pRow, pSel, r, h, j ir.Value, heads, blk, k, page int) (t, pid ir.Value) {
	one := b.Const(ir.U32, 1)
	end, own := msaOwn(b, pRow, r, blk)
	jj := b.Min(ir.U32, j, b.Sub(ir.U32, msaKept(b, end, own, blk, k), one))
	bl := b.Const(ir.U32, int64(blk))
	rh := b.Add(ir.U32, b.Mul(ir.U32, r, b.Const(ir.U32, int64(heads))), h)
	blkID := b.Load(ir.U32, pSel, b.Add(ir.U32, b.Mul(ir.U32, rh, b.Const(ir.U32, int64(k+1))), b.Div(ir.U32, jj, bl)), 0)
	t = b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, blkID, bl), b.Rem(ir.U32, jj, bl)), b.Sub(ir.U32, end, one))
	return t, pageOf(b, pTab, pagedDesc(b, pRow, r, PRowTab), t, page)
}

// MSAGatherK copies each row's kept keys into its gathered K pages: column e
// (of the attention's heads*headDim) of gathered position j is column e of
// the kept position j of e's group, both [width][P] pages. Parameters: pSrc
// (the pool), pTab, pRow, pSel, pDst.
func MSAGatherK(rows, heads, headDim, blk, k, page, kvRow int) (*ir.Kernel, error) {
	if err := checkPage("MSAGatherK", page); err != nil {
		return nil, err
	}
	w := heads * headDim
	if rows < 1 || w < 1 || blk < 1 || k < 1 || w > kvRow {
		return nil, fmt.Errorf("kernels: MSAGatherK: rows=%d heads=%d dim=%d blk=%d k=%d row=%d",
			rows, heads, headDim, blk, k, kvRow)
	}
	kc := k * blk
	ppr := MSAGatherPages(k, blk, page)
	b := ir.New("msagatherk", [3]int{128, 1, 1})
	pSrc := b.Param("pSrc", ir.F32)
	pTab := b.Param("pTab", ir.U32)
	pRow := b.Param("pRow", ir.U32)
	pSel := b.Param("pSel", ir.U32)
	pDst := b.Param("pDst", ir.F32)
	flat := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(rows*w*kc-1)))
	// Position fastest, so a warp reads consecutive floats of one column.
	j := b.Rem(ir.U32, flat, b.Const(ir.U32, int64(kc)))
	er := b.Div(ir.U32, flat, b.Const(ir.U32, int64(kc)))
	e := b.Rem(ir.U32, er, b.Const(ir.U32, int64(w)))
	r := b.Div(ir.U32, er, b.Const(ir.U32, int64(w)))
	h := b.Div(ir.U32, e, b.Const(ir.U32, int64(headDim)))
	t, pid := msaGatherPos(b, pTab, pRow, pSel, r, h, j, heads, blk, k, page)
	src := b.Add(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, pid, b.Const(ir.U32, int64(page*kvRow))),
		b.Mul(ir.U32, e, b.Const(ir.U32, int64(page)))), pageRem(b, t, page))
	gp := b.Add(ir.U32, b.Mul(ir.U32, r, b.Const(ir.U32, int64(ppr))), pageDiv(b, j, page))
	dst := b.Add(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, gp, b.Const(ir.U32, int64(page*w))),
		b.Mul(ir.U32, e, b.Const(ir.U32, int64(page)))), pageRem(b, j, page))
	b.Store(pDst, dst, b.Load(ir.F32, pSrc, src, 0), 0)
	return b.Done(), nil
}

// MSAGatherV copies each row's kept values into its gathered V pages, both
// [P][width] row-major in 32-bit words (two binary16 values a word when f16,
// which never straddles a head since headDim is even). Parameters: pSrc (the
// pool), pTab, pRow, pSel, pDst.
func MSAGatherV(rows, heads, headDim, blk, k, page, kvRow int, f16 bool) (*ir.Kernel, error) {
	if err := checkPage("MSAGatherV", page); err != nil {
		return nil, err
	}
	w := heads * headDim
	if rows < 1 || w < 1 || blk < 1 || k < 1 || w > kvRow || (f16 && (headDim%2 != 0 || kvRow%2 != 0)) {
		return nil, fmt.Errorf("kernels: MSAGatherV: rows=%d heads=%d dim=%d blk=%d k=%d row=%d f16=%v",
			rows, heads, headDim, blk, k, kvRow, f16)
	}
	ew := 1
	if f16 {
		ew = 2
	}
	ww, rw := w/ew, kvRow/ew
	kc := k * blk
	ppr := MSAGatherPages(k, blk, page)
	b := ir.New("msagatherv", [3]int{128, 1, 1})
	pSrc := b.Param("pSrc", ir.F32)
	pTab := b.Param("pTab", ir.U32)
	pRow := b.Param("pRow", ir.U32)
	pSel := b.Param("pSel", ir.U32)
	pDst := b.Param("pDst", ir.F32)
	flat := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(rows*kc*ww-1)))
	// The word fastest: a row's words are contiguous on both sides.
	u := b.Rem(ir.U32, flat, b.Const(ir.U32, int64(ww)))
	rj := b.Div(ir.U32, flat, b.Const(ir.U32, int64(ww)))
	j := b.Rem(ir.U32, rj, b.Const(ir.U32, int64(kc)))
	r := b.Div(ir.U32, rj, b.Const(ir.U32, int64(kc)))
	h := b.Div(ir.U32, b.Mul(ir.U32, u, b.Const(ir.U32, int64(ew))), b.Const(ir.U32, int64(headDim)))
	t, pid := msaGatherPos(b, pTab, pRow, pSel, r, h, j, heads, blk, k, page)
	src := b.Add(ir.U32, b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, pid, b.Const(ir.U32, int64(page))),
		pageRem(b, t, page)), b.Const(ir.U32, int64(rw))), u)
	gp := b.Add(ir.U32, b.Mul(ir.U32, r, b.Const(ir.U32, int64(ppr))), pageDiv(b, j, page))
	dst := b.Add(ir.U32, b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, gp, b.Const(ir.U32, int64(page))),
		pageRem(b, j, page)), b.Const(ir.U32, int64(ww))), u)
	b.Store(pDst, dst, b.Load(ir.F32, pSrc, src, 0), 0)
	return b.Done(), nil
}

// MSADesc writes the gathered pages' descriptors: row r's table at
// r*MSAGatherPages, its keys 0..kept, written nowhere. Parameters: pRow (the
// history's), pOut.
func MSADesc(rows, blk, k, page int) (*ir.Kernel, error) {
	if rows < 1 || blk < 1 || k < 1 {
		return nil, fmt.Errorf("kernels: MSADesc: rows=%d blk=%d k=%d", rows, blk, k)
	}
	b := ir.New("msadesc", [3]int{128, 1, 1})
	pRow := b.Param("pRow", ir.U32)
	pOut := b.Param("pOut", ir.U32)
	flat := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(rows*PRowWords-1)))
	w := b.Rem(ir.U32, flat, b.Const(ir.U32, PRowWords))
	r := b.Div(ir.U32, flat, b.Const(ir.U32, PRowWords))
	zero := b.Const(ir.U32, 0)
	end, own := msaOwn(b, pRow, r, blk)
	kept := msaKept(b, end, own, blk, k)
	tab := b.Mul(ir.U32, r, b.Const(ir.U32, int64(MSAGatherPages(k, blk, page))))
	isTab := b.Lt(ir.U32, w, b.Const(ir.U32, PRowTab+1))
	isEnd := b.Select(ir.U32, b.Lt(ir.U32, w, b.Const(ir.U32, PRowEnd)), zero,
		b.Select(ir.U32, b.Lt(ir.U32, w, b.Const(ir.U32, PRowEnd+1)), b.Const(ir.U32, 1), zero))
	v := b.Select(ir.U32, isTab, tab, b.Mul(ir.U32, isEnd, kept))
	b.Store(pOut, flat, v, 0)
	return b.Done(), nil
}
