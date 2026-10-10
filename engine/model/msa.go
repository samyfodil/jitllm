package model

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
)

// MiniMax Sparse Attention: MiniMax-M3's block selection (transformers'
// MiniMaxM3VLIndexer, llama.cpp's minimax-m3.cpp).
//
// Per kv group g the query at p scores every cached t <= p by q_g . k_t, ranks
// the blocks of IdxBlock positions by their best position, forces in the
// IdxLocal blocks ending at its own, and keeps the IdxTopK best; the group's
// heads attend there alone, through a -inf mask added after the scale and
// before the softmax. The key is one more kv head of the attention row
// (Config.KVRowAt, kvlMSA), so it pages, relocates and is prefix-cached with
// k: the attention reads heads 0..NKVHead-1, the indexer head NKVHead, and
// the value row's extra head is zero. The ranking is the sampler's ordering
// over positions, keeping each block as it is first seen: the first position
// seen of a block is its maximum.
//
// The full description: docs/engineering-history/model-correctness.md,
// "engine/model/msa.go".

// msaFault is a violation of one block-selection piece, set only by a gate
// (Config.msaFault) to show the fixture can see it.
type msaFault uint8

const (
	msaFaultNone msaFault = iota
	// msaFaultNoLocal forces no block in: the query's own block competes.
	msaFaultNoLocal
	// msaFaultSharedGroups selects once, with group 0's head, for every
	// group.
	msaFaultSharedGroups
	// msaFaultFirstPosition ranks a block by its first position's score
	// instead of its best.
	msaFaultFirstPosition
	// msaFaultOneShort keeps one block fewer than the top-k.
	msaFaultOneShort
)

// loadMSA reads block bi's indexer tensors and checks their shapes.
func loadMSA(l *layer, bi int32, cfg *Config,
	get func(jlm.Role, int32, int32) (tensor, error), vec func(jlm.Role, int32) ([]float32, error)) error {
	var err error
	if l.idxQ, err = get(jlm.RoleIdxQ, bi, -1); err != nil {
		return err
	}
	if l.idxK, err = get(jlm.RoleIdxK, bi, -1); err != nil {
		return err
	}
	if l.idxQNorm, err = vec(jlm.RoleIdxQNorm, bi); err != nil {
		return err
	}
	if l.idxKNorm, err = vec(jlm.RoleIdxKNorm, bi); err != nil {
		return err
	}
	h, d := cfg.IdxHeads, cfg.IdxHeadDim
	if l.idxQ.k != cfg.NEmbd || l.idxQ.rows != h*d || l.idxK.k != cfg.NEmbd || l.idxK.rows != d ||
		len(l.idxQNorm) != d || len(l.idxKNorm) != d {
		return fmt.Errorf("model: block %d's indexer is q %dx%d, k %dx%d, norms %d/%d; "+
			"want %d heads of %d over a %d-wide input", bi, l.idxQ.k, l.idxQ.rows, l.idxK.k,
			l.idxK.rows, len(l.idxQNorm), len(l.idxKNorm), h, d, cfg.NEmbd)
	}
	return nil
}

// allocMSA sizes the decode token's selection buffers.
func (s *State) allocMSA() {
	c := s.m.Cfg
	if !c.MSA() {
		return
	}
	s.msaQ = make([]float32, c.IdxHeads*c.IdxHeadDim)
	s.msaTmp = make([]float32, s.attStride)
	s.msaBias = make([]float32, c.IdxHeads*s.attStride)
	s.msaKept = make([]bool, s.attStride/c.IdxBlock+1)
	s.msaOrd.Reserve(s.maxSeq)
}

// growMSARows sizes a chunk's: each row's query, key and masks.
func (s *State) growMSARows(n int) {
	c := s.m.Cfg
	if !c.MSA() || len(s.msaMasks) >= n {
		return
	}
	s.bmsaQ = make([]float32, n*c.IdxHeads*c.IdxHeadDim)
	s.bmsaK = make([]float32, n*c.IdxHeadDim)
	s.bmsaBias = make([]float32, n*c.IdxHeads*s.attStride)
	s.msaMasks = make([][]float32, n)
}

// msaKey writes one position's indexer key into dst (IdxHeadDim wide): the
// projection of the block's normed input h, its RMSNorm and the rotary at
// table cs.
func (s *State) msaKey(dst []float32, l *layer, h, cs []float32) error {
	if err := s.mv(dst, l.idxK, h); err != nil {
		return err
	}
	s.msaKeyRest(dst, l, cs)
	return nil
}

// msaKeyRest is msaKey after the projection.
func (s *State) msaKeyRest(k []float32, l *layer, cs []float32) {
	c := s.m.Cfg
	s.rmsnorm(k, k, l.idxKNorm, c.RMSEps)
	if c.NRot > 0 && !c.NoPosEnc {
		nn.RoPE32JIT(k, len(k), cs, s.m.rope.Neox)
	}
}

// msaQuery fills q (IdxHeads*IdxHeadDim) for one position: the projection
// of h, each head's RMSNorm and the rotary at cs.
func (s *State) msaQuery(q []float32, l *layer, h, cs []float32) error {
	if err := s.mv(q, l.idxQ, h); err != nil {
		return err
	}
	s.msaQueryRest(q, l, cs)
	return nil
}

// msaQueryRest is msaQuery after the projection.
func (s *State) msaQueryRest(q []float32, l *layer, cs []float32) {
	c := s.m.Cfg
	d := c.IdxHeadDim
	for g := 0; g < c.IdxHeads; g++ {
		t := q[g*d : (g+1)*d]
		s.rmsnorm(t, t, l.idxQNorm, c.RMSEps)
	}
	if c.NRot > 0 && !c.NoPosEnc {
		nn.RoPE32JIT(q[:c.IdxHeads*d], d, cs, s.m.rope.Neox)
	}
}

// msaMask fills bias (IdxHeads rows of stride, each n long) with the
// selection of a query whose window is the first n cached positions of
// sequence slot in layer li's history: 0 at the positions of each group's
// kept blocks, -inf everywhere else. It returns nil when every block is kept,
// where the attention is dense. The window's pages must be resident
// (kvEnsureWindow).
//
// tmp is n-long scratch, kept one flag a block, ord the ranking's state. A
// caller running rows in parallel hands each its own.
func (s *State) msaMask(li, slot, n, stride int, q, tmp, bias []float32, kept []bool, ord *nn.SampleOrder) []float32 {
	c := s.m.Cfg
	B, d := c.IdxBlock, c.IdxHeadDim
	nb := (n + B - 1) / B
	k := min(c.IdxTopK, nb)
	if nb <= c.IdxTopK {
		return nil
	}
	if c.msaFault == msaFaultOneShort {
		k--
	}
	own := (n - 1) / B
	ninf := float32(math.Inf(-1))
	for g := 0; g < c.IdxHeads; g++ {
		row := bias[g*stride : g*stride+n]
		qg := q[g*d : (g+1)*d]
		if c.msaFault == msaFaultSharedGroups {
			qg = q[:d]
		}
		s.msaScores(li, slot, qg, tmp[:n])
		kept := kept[:nb]
		clear(kept)
		have := 0
		if c.msaFault != msaFaultNoLocal {
			for l := 0; l < c.IdxLocal; l++ {
				if b := max(own-l, 0); !kept[b] {
					kept[b] = true
					have++
				}
			}
		}
		if c.msaFault == msaFaultFirstPosition {
			// Each block's first position stands for it: the scores of the
			// rest are pushed below every first one.
			for t := range tmp[:n] {
				if t%B != 0 {
					tmp[t] = ninf
				}
			}
		}
		ord.Begin(tmp[:n])
		for have < k {
			_, t, ok := ord.Next()
			if !ok {
				break
			}
			if b := int(t) / B; !kept[b] {
				kept[b] = true
				have++
			}
		}
		for b := 0; b < nb; b++ {
			lo, hi := b*B, min((b+1)*B, n)
			if kept[b] {
				clear(row[lo:hi])
				continue
			}
			for t := lo; t < hi; t++ {
				row[t] = ninf
			}
		}
	}
	return bias
}

// msaScores fills dst[t] = q . k_t for the first len(dst) positions of slot's
// history in layer li, reading each cached row's indexer head.
func (s *State) msaScores(li, slot int, q, dst []float32) {
	c := s.m.Cfg
	pg := &s.kv.layers[li]
	kvl := s.kvlAt(li)
	for pos, end := 0, len(dst); pos < end && pg.p > 0; {
		sp := pg.span(pos, end)
		if !pg.resident(sp.page) {
			return
		}
		s.attnM.AttnScores(dst[pos:], pg.kSpan(kvl, sp, slot, c.NKVHead), q, sp.n)
		pos += sp.n
	}
}

// msaRows computes the selection for each of n rows before a batched
// attention: row i's query is in bmsaQ, its history is slot(i)'s first
// win(i) positions of layer li, and its masks land in bmsaBias at
// i*IdxHeads*attStride (masks[i], nil for a dense row).
func (s *State) msaRows(li, n int, slot, win func(i int) int, masks [][]float32) {
	c := s.m.Cfg
	hq, hb := c.IdxHeads*c.IdxHeadDim, c.IdxHeads*s.attStride
	for i := 0; i < n; i++ {
		masks[i] = s.msaMask(li, slot(i), win(i), s.attStride, s.bmsaQ[i*hq:(i+1)*hq], s.msaTmp,
			s.bmsaBias[i*hb:(i+1)*hb], s.msaKept, &s.msaOrd)
	}
}

// msaRow assembles one row's cache vectors for an MSA block: its k (kvDim
// wide) then its indexer key in kr, its v then zeros in vr.
func msaRow(kr, vr, k, v, key []float32) {
	n := copy(kr, k)
	copy(kr[n:], key)
	n = copy(vr, v)
	clear(vr[n:])
}

// msaProjectRows fills every row's indexer key (bmsaK) and query (bmsaQ)
// from a chunk's n normed inputs in bh, each at its own row's rotary table.
func (s *State) msaProjectRows(l *layer, n int, cs, csSWA []float32, li int) error {
	c := s.m.Cfg
	d, hq := c.IdxHeadDim, c.IdxHeads*c.IdxHeadDim
	if err := s.mm(s.bmsaK, l.idxK, s.bh, n); err != nil {
		return err
	}
	if err := s.mm(s.bmsaQ, l.idxQ, s.bh, n); err != nil {
		return err
	}
	for i := 0; i < n; i++ {
		t := s.ropeTable(li, cs, csSWA, i)
		s.msaKeyRest(s.bmsaK[i*d:(i+1)*d], l, t)
		s.msaQueryRest(s.bmsaQ[i*hq:(i+1)*hq], l, t)
	}
	return nil
}

// msaRowOf is row i's cache vectors for an MSA block, assembled in kc and
// vc from its k and v and its indexer key in bmsaK.
func (s *State) msaRowOf(li, i int, k, v []float32) ([]float32, []float32) {
	c := s.m.Cfg
	d, kvr := c.IdxHeadDim, c.KVRowAt(li)
	msaRow(s.kc[:kvr], s.vc[:kvr], k, v, s.bmsaK[i*d:(i+1)*d])
	return s.kc[:kvr], s.vc[:kvr]
}
