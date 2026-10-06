package model

import (
	"github.com/samyfodil/jitllm/engine/nn"
)

// Multi-head Latent Attention, the DeepSeek-V2/V3 attention and the one place
// in this engine where a key and a value are not the same width.
//
// The cache holds one row per position for the whole layer:
//
//	[ latent (KVLoraRank) | rotary key (NRot) ]
//
// and every head attends to that same row. The up-projections that would turn
// the latent back into per-head keys and values are absorbed into the operands
// either side of the softmax:
//
//	score_h  = q_nope_h . (W_k_h . c)  =  (W_k_hᵀ . q_nope_h) . c
//	out_h    = Σ a_j (W_v_h . c_j)     =  W_v_h . (Σ a_j c_j)
//
// so W_k_hᵀ folds into the query and W_v_h into the output, and neither ever
// materialises a key or a value. That is why l.wkb is stored already
// transposed (the converter does it) and l.wvb is not.
//
// The value is the key's prefix rather than a second tensor: the scores read
// all KVLoraRank+NRot floats of the row and the accumulation only the leading
// KVLoraRank. kvPages.latent aliases the two caches and nn.AddAttnKV emits the
// kernels at the two widths.

// allocMLA sizes the per-State buffers one MLA token needs, so a decode token
// allocates nothing here.
func (s *State) allocMLA() {
	c := s.c
	if !c.MLA() {
		return
	}
	row := c.KVLoraRank + c.NRot
	s.mlaAbs = make([]float32, c.NHead*row)
	s.mlaAcc = make([]float32, c.NHead*c.KVLoraRank)
	s.mlaPE = make([]float32, c.NHead*c.NRot)
	s.mlaOut = make([]float32, c.NHead*c.HeadDimV)
	if c.QLoraRank != 0 {
		s.mlaQA = make([]float32, c.QLoraRank)
	}
	// One output slice per head for each gather, built once so the per-token
	// path appends nothing.
	s.mlaAbsO = make([][]float32, c.NHead)
	s.mlaOutO = make([][]float32, c.NHead)
	s.mlaKB = make([]*nn.Packed, 0, c.NHead)
	s.mlaVB = make([]*nn.Packed, 0, c.NHead)
	for h := 0; h < c.NHead; h++ {
		s.mlaAbsO[h] = s.mlaAbs[h*row : h*row+c.KVLoraRank]
		s.mlaOutO[h] = s.mlaOut[h*c.HeadDimV : (h+1)*c.HeadDimV]
	}
	if s.m.mlaGated() {
		s.mlaG = make([]float32, c.NHead*c.HeadDimV)
	}
	s.allocIndexer()
}

// mlaWeights collects the per-head packed views for one layer's two gathers,
// and reports whether every head is uniform (same type, same shape, packed)
// enough to batch. It is not cached because a paged block's weights can be
// evicted between calls; moe.go's uniformity check does the same.
func (s *State) mlaWeights(l *layer) bool {
	s.mlaKB, s.mlaVB = s.mlaKB[:0], s.mlaVB[:0]
	if len(l.kbHead) == 0 {
		return false
	}
	k0, v0 := &l.kbHead[0], &l.vbHead[0]
	if k0.packed == nil || v0.packed == nil {
		return false
	}
	for i := range l.kbHead {
		k, v := &l.kbHead[i], &l.vbHead[i]
		if k.packed == nil || k.typ != k0.typ || k.rows != k0.rows || k.k != k0.k ||
			v.packed == nil || v.typ != v0.typ || v.rows != v0.rows || v.k != v0.k {
			return false
		}
		s.mlaKB = append(s.mlaKB, k.packed)
		s.mlaVB = append(s.mlaVB, v.packed)
	}
	return true
}

// mlaProject runs everything before the softmax: both projections, both latent
// norms, the rotary, the absorb, and the cache write. It leaves the absorbed
// query in s.mlaAbs and the position's row in the cache.
//
// x is the block's normed residual. cs is this position's rotary table.
func (s *State) mlaProject(li int, l *layer, x, cs []float32, kvSlot, kvPos int) error {
	c := s.c
	nh, hd, rot := c.NHead, c.HeadDim, c.NRot
	nope, lat := hd-rot, c.KVLoraRank
	row := lat + rot

	// ── the query, in one step or two ────────────────────────────────────
	//
	// DeepSeek-V3 compresses the query through a latent; V2-Lite has no
	// q_lora_rank and projects straight to NHead*HeadDim. Ask the config, not
	// `l.wqa.data != nil`: a paged matrix's data is nil whenever its page is
	// not resident, so presence of the tensor is only a fact for
	// always-resident tensors.
	if c.QLoraRank != 0 {
		if err := s.mv(s.mlaQA, l.wqa, x); err != nil {
			return err
		}
		s.rmsnorm(s.mlaQA, s.mlaQA, l.qaNorm, c.LatentNormEps)
		if err := s.mv(s.q[:nh*hd], l.wqb, s.mlaQA); err != nil {
			return err
		}
	} else if err := s.mv(s.q[:nh*hd], l.wq, x); err != nil {
		return err
	}

	// ── the shared latent and the shared rotary key ──────────────────────
	//
	// One projection for both k and v. The RMSNorm covers only the first
	// KVLoraRank floats; the rotary tail is not normed.
	if err := s.mv(s.kc[:row], l.wkva, x); err != nil {
		return err
	}
	s.rmsnorm(s.kc[:lat], s.kc[:lat], l.kvaNorm, c.LatentNormEps)
	// Kimi-K3's output gate reads the block's normed input, which the
	// attention's output overwrites; it is taken now (mlaOutProject applies
	// it).
	if l.mlaGate.rows != 0 {
		if err := s.mv(s.mlaG, l.mlaGate, x); err != nil {
			return err
		}
	}

	// ── rotary, on the pe halves only ────────────────────────────────────
	//
	// The query's rotary half is the tail of each head and so not contiguous;
	// it is compacted into s.mlaPE first, which the absorbed query needs
	// anyway. The key's half is already contiguous at s.kc[lat:].
	for h := 0; h < nh; h++ {
		copy(s.mlaPE[h*rot:(h+1)*rot], s.q[h*hd+nope:(h+1)*hd])
	}
	t0 := s.tick()
	// deepseek2 rotates adjacent pairs (not NEOX); the container flag says so.
	// A NoPE model (Kimi-Linear) skips both rotations. Roping anyway is exact at
	// position 0, where the rotation is the identity, and wrong after it.
	if !s.c.NoPosEnc {
		nn.RoPE32JIT(s.mlaPE[:nh*rot], rot, cs, s.m.rope.Neox)
		nn.RoPE32JIT(s.kc[lat:row], rot, cs, s.m.rope.Neox)
	}
	s.tock(opRoPE, t0)

	// ── absorb W_k into the query ────────────────────────────────────────
	//
	// The gather reads q's nope half in place: head h's activation is
	// q[h*HeadDim : +nope], and MatVecPackedGather takes the stride and k
	// separately.
	ok := s.mlaWeights(l)
	if ok {
		ok = s.jit.MatVecPackedGather(s.mlaAbsO, l.kbHead[0].typ, s.mlaKB,
			s.q, hd, lat, nope)
	}
	if !ok {
		// Per head, for a non-uniform or unpacked bank; bit-identical to the
		// batch by construction.
		s.mlaAbsLooped.Add(1)
		for h := 0; h < nh; h++ {
			s.jit.NewInput()
			if err := s.mv(s.mlaAbsO[h], l.kbHead[h], s.q[h*hd:h*hd+nope]); err != nil {
				return err
			}
		}
	}
	// The rotary half joins the absorbed latent to make the row-wide query.
	for h := 0; h < nh; h++ {
		copy(s.mlaAbs[h*row+lat:(h+1)*row], s.mlaPE[h*rot:(h+1)*rot])
	}

	// ── the indexer: its query, weights and the key cached in the row ─────
	if c.Indexer() {
		if err := s.idxQuery(s.idxQ, s.idxW, l, s.mlaQA, x, cs); err != nil {
			return err
		}
		if err := s.idxKey(s.kc, l, x, cs); err != nil {
			return err
		}
	}

	// ── the cache ────────────────────────────────────────────────────────
	//
	// The same slice twice: the value is the key's latent prefix and
	// kvPages.latent has made the two page arrays one buffer.
	s.kvWritable(li, kvPos)
	w := c.KVDim()
	s.kv.layers[li].write(s.kvl, kvSlot, kvPos, s.kc[:w], s.kc[:w])
	return nil
}

// mlaOutProject un-absorbs W_v and runs the output projection, turning the
// KVLoraRank-wide attention result of each head into HeadDimV and then into the
// residual's width.
func (s *State) mlaOutProject(out []float32, l *layer) error {
	c := s.c
	ok := s.mlaWeights(l)
	if ok {
		ok = s.jit.MatVecPackedGather(s.mlaOutO, l.vbHead[0].typ, s.mlaVB,
			s.mlaAcc, c.KVLoraRank, c.HeadDimV, c.KVLoraRank)
	}
	if !ok {
		s.mlaAbsLooped.Add(1)
		for h := 0; h < c.NHead; h++ {
			s.jit.NewInput()
			lo := h * c.KVLoraRank
			if err := s.mv(s.mlaOutO[h], l.vbHead[h], s.mlaAcc[lo:lo+c.KVLoraRank]); err != nil {
				return err
			}
		}
	}
	// No bias here: the caller adds l.bo for every architecture.
	o := s.mlaOut[:c.NHead*c.HeadDimV]
	if l.mlaGate.rows != 0 && c.k3Fault != k3FaultNoMLAGate {
		// Kimi-K3: sigma(g) times the attention output, before o_proj.
		nn.SigmoidMul32JIT(s.mlaG, o)
		s.jit.NewInput()
		o = s.mlaG
	}
	return s.mv(out, l.wo, o)
}

// ── the batched paths ──────────────────────────────────────────────────────────────
//
// The projections batch and the rest does not, the same split the dense path
// makes: q, the latent and their norms are row-independent; the rotary, the
// absorb and the cache write depend on the position and run per row. The
// operations and their order are mlaProject's.

// growMLABatch sizes the per-chunk MLA buffers.
func (s *State) growMLABatch(chunk int) {
	c := s.c
	if !c.MLA() || len(s.bmlaAbs) >= chunk*c.NHead*(c.KVLoraRank+c.NRot) {
		return
	}
	row := c.KVLoraRank + c.NRot
	s.bmlaAbs = make([]float32, chunk*c.NHead*row)
	s.bmlaAcc = make([]float32, chunk*c.NHead*c.KVLoraRank)
	s.bmlaKV = make([]float32, chunk*row)
	s.bmlaOut = make([]float32, chunk*c.NHead*c.HeadDimV)
	if s.m.mlaGated() {
		s.bmlaG = make([]float32, chunk*c.NHead*c.HeadDimV)
	}
	if c.QLoraRank != 0 {
		s.bmlaQA = make([]float32, chunk*c.QLoraRank)
	}
	// One output slice per (row, head) for the two gathers, built once.
	s.bmlaAbsO = make([][]float32, c.NHead)
	s.bmlaOutO = make([][]float32, c.NHead)
	s.growIndexerRows(chunk)
}

// mlaPrefillProject is mlaProject over a chunk of n rows: it leaves every
// row's absorbed query in s.bmlaAbs and every row's latent in the cache.
//
// at gives each row's slot and position, the only thing the two callers
// disagree about: Prefill's rows are successive positions of one sequence,
// ForwardBatch's are different sequences. Keeping one body avoids the paths
// drifting apart.
//
// cs and csSWA are the caller's rotary tables, n rows of NRot.
func (s *State) mlaProjectRows(li int, l *layer, n int, cs, csSWA []float32,
	at func(i int) (slot, pos int)) error {
	c := s.c
	nh, hd, rot := c.NHead, c.HeadDim, c.NRot
	nope, lat := hd-rot, c.KVLoraRank
	row := lat + rot
	qDim := nh * hd

	// ── the batched half ─────────────────────────────────────────────────
	if c.QLoraRank != 0 {
		if err := s.mm(s.bmlaQA, l.wqa, s.bh, n); err != nil {
			return err
		}
		for i := 0; i < n; i++ {
			t := s.bmlaQA[i*c.QLoraRank : (i+1)*c.QLoraRank]
			s.rmsnorm(t, t, l.qaNorm, c.LatentNormEps)
		}
		if err := s.mm(s.bq, l.wqb, s.bmlaQA, n); err != nil {
			return err
		}
	} else if err := s.mm(s.bq, l.wq, s.bh, n); err != nil {
		return err
	}
	if err := s.mm(s.bmlaKV, l.wkva, s.bh, n); err != nil {
		return err
	}
	if l.mlaGate.rows != 0 {
		if err := s.mm(s.bmlaG, l.mlaGate, s.bh, n); err != nil {
			return err
		}
	}
	hq := c.IdxHeads * c.IdxHeadDim
	if c.Indexer() {
		if err := s.mm(s.bidxQ, l.idxQB, s.bmlaQA, n); err != nil {
			return err
		}
		if err := s.mm(s.bidxW, l.idxProj, s.bh, n); err != nil {
			return err
		}
		if err := s.mm(s.bidxK, l.idxK, s.bh, n); err != nil {
			return err
		}
	}

	// ── the per-row half ─────────────────────────────────────────────────
	for i := 0; i < n; i++ {
		slot, pos := at(i)
		kv := s.bmlaKV[i*row : (i+1)*row]
		// The norm covers the latent only; the rotary tail is not normed.
		s.rmsnorm(kv[:lat], kv[:lat], l.kvaNorm, c.LatentNormEps)

		q := s.bq[i*qDim : (i+1)*qDim]
		abs := s.bmlaAbs[i*nh*row : (i+1)*nh*row]
		// q's rotary half is the tail of each head and is therefore not
		// contiguous; see mlaProject for why it is compacted first.
		for h := 0; h < nh; h++ {
			copy(s.mlaPE[h*rot:(h+1)*rot], q[h*hd+nope:(h+1)*hd])
		}
		// Only the rotation is skipped for a NoPE model; the compaction,
		// absorb and cache write below still run.
		if !s.c.NoPosEnc {
			csi := s.ropeTable(li, cs, csSWA, i)
			t0 := s.tick()
			nn.RoPE32JIT(s.mlaPE[:nh*rot], rot, csi, s.m.rope.Neox)
			nn.RoPE32JIT(kv[lat:row], rot, csi, s.m.rope.Neox)
			s.tock(opRoPE, t0)
		}

		for h := 0; h < nh; h++ {
			s.bmlaAbsO[h] = abs[h*row : h*row+lat]
		}
		ok := s.mlaWeights(l)
		if ok {
			ok = s.jit.MatVecPackedGather(s.bmlaAbsO, l.kbHead[0].typ, s.mlaKB, q, hd, lat, nope)
		}
		if !ok {
			s.mlaAbsLooped.Add(1)
			for h := 0; h < nh; h++ {
				s.jit.NewInput()
				if err := s.mv(s.bmlaAbsO[h], l.kbHead[h], q[h*hd:h*hd+nope]); err != nil {
					return err
				}
			}
		}
		for h := 0; h < nh; h++ {
			copy(abs[h*row+lat:(h+1)*row], s.mlaPE[h*rot:(h+1)*rot])
		}
		// The same slice twice: the value is the key's latent prefix.
		s.kvWritable(li, pos)
		if c.Indexer() {
			// The cached row is the latent and rotary key, then the indexer's
			// key, assembled in s.kc.
			csi := s.ropeTable(li, cs, csSWA, i)
			d := c.IdxHeadDim
			full := s.kc[:c.KVDim()]
			copy(full, kv)
			copy(full[row:], s.bidxK[i*d:(i+1)*d])
			s.idxKeyRest(full[row:], l, csi)
			s.idxQueryRest(s.bidxQ[i*hq:(i+1)*hq], s.bidxW[i*c.IdxHeads:(i+1)*c.IdxHeads], csi)
			kv = full
		}
		s.kv.layers[li].write(s.kvl, slot, pos, kv, kv)
		s.jit.NewInput()
	}
	return nil
}

// mlaOutRows un-absorbs W_v for every row and runs the output projection. It
// needs no slot or position: the un-absorb is per row and the rows are already
// separated by the attention that wrote bmlaAcc.
func (s *State) mlaOutRows(l *layer, n int) error {
	c := s.c
	lat, hdv := c.KVLoraRank, c.HeadDimV
	for i := 0; i < n; i++ {
		acc := s.bmlaAcc[i*c.NHead*lat : (i+1)*c.NHead*lat]
		out := s.bmlaOut[i*c.NHead*hdv : (i+1)*c.NHead*hdv]
		for h := 0; h < c.NHead; h++ {
			s.bmlaOutO[h] = out[h*hdv : (h+1)*hdv]
		}
		ok := s.mlaWeights(l)
		if ok {
			ok = s.jit.MatVecPackedGather(s.bmlaOutO, l.vbHead[0].typ, s.mlaVB, acc, lat, hdv, lat)
		}
		if !ok {
			s.mlaAbsLooped.Add(1)
			for h := 0; h < c.NHead; h++ {
				s.jit.NewInput()
				if err := s.mv(s.bmlaOutO[h], l.vbHead[h], acc[h*lat:(h+1)*lat]); err != nil {
					return err
				}
			}
		}
		s.jit.NewInput()
	}
	// Into s.bh, as the dense path's wo, so the residual add is shared.
	o := s.bmlaOut[:n*c.NHead*hdv]
	if l.mlaGate.rows != 0 && c.k3Fault != k3FaultNoMLAGate {
		// Kimi-K3's output gate; see mlaOutProject.
		nn.SigmoidMul32JIT(s.bmlaG[:n*c.NHead*hdv], o)
		s.jit.NewInput()
		o = s.bmlaG[:n*c.NHead*hdv]
	}
	return s.mm(s.bh, l.wo, o, n)
}

// attnWidths is the key and value widths the attention kernels are emitted
// for. They differ only under MLA, where a score covers the whole cached row
// and the accumulation only its latent prefix. NewState and stepDownKVToF32
// must both derive them here.
func (c *Config) attnWidths() (hdK, hdV int) {
	if c.KVLoraRank != 0 {
		return c.KVLoraRank + c.NRot, c.KVLoraRank
	}
	return c.HeadDim, c.HeadDim
}

// mlaGated reports whether any full block carries Kimi-K3's MLA output gate.
func (m *Model) mlaGated() bool {
	for i := range m.layers {
		if m.layers[i].mlaGate.rows != 0 {
			return true
		}
	}
	return false
}
