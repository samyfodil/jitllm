package model

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
)

// DeepSeek Sparse Attention: the lightning indexer of DeepSeek V3.2
// (transformers' DeepseekV32Indexer, llama.cpp's deepseek32.cpp).
//
// For a query at p the indexer scores every cached t <= p as
// sum_h w_h * relu(q_h . k_t), and the attention reads only the IdxTopK best
// positions -- every position at or below IdxTopK, where the block is exactly
// deepseek2's -- through a -inf mask added after the scale and before the
// softmax. The key is cached in the MLA row itself, after the latent and the
// rotary key (Config.KVDim), so it pages, relocates and is prefix-cached with
// the latent: the MLA kernels read the row's first KVLoraRank+NRot elements
// and the indexer its last IdxHeadDim. Every step is generated code.
//
// The projections, norms and scales: docs/engineering-history/
// model-correctness.md, "engine/model/indexer.go".

// idxFault is a violation of one indexer piece, set only by a gate
// (Config.idxFault) to show the fixture can see it.
type idxFault uint8

const (
	idxFaultNone idxFault = iota
	// idxFaultPairRope rotates adjacent pairs (MLA's own layout) where the
	// indexer rotates NEOX halves.
	idxFaultPairRope
	// idxFaultNoReLU sums the heads' raw scores.
	idxFaultNoReLU
)

// idxEps is the indexer key norm's epsilon: nn.LayerNorm(eps=1e-6) in the
// reference, hardcoded in llama.cpp's builder; no GGUF key carries it.
const idxEps = 1e-6

// loadIndexer reads block bi's indexer tensors and checks their shapes.
func loadIndexer(l *layer, bi int32, cfg *Config,
	get func(jlm.Role, int32, int32) (tensor, error), vec func(jlm.Role, int32) ([]float32, error)) error {
	var err error
	if l.idxQB, err = get(jlm.RoleIdxQB, bi, -1); err != nil {
		return err
	}
	if l.idxK, err = get(jlm.RoleIdxK, bi, -1); err != nil {
		return err
	}
	if l.idxProj, err = get(jlm.RoleIdxProj, bi, -1); err != nil {
		return err
	}
	if l.idxKNorm, err = vec(jlm.RoleIdxKNorm, bi); err != nil {
		return err
	}
	if l.idxKNormB, err = vec(jlm.RoleIdxKNormB, bi); err != nil {
		return err
	}
	h, d := cfg.IdxHeads, cfg.IdxHeadDim
	if l.idxQB.k != cfg.QLoraRank || l.idxQB.rows != h*d || l.idxK.k != cfg.NEmbd || l.idxK.rows != d ||
		l.idxProj.k != cfg.NEmbd || l.idxProj.rows != h || len(l.idxKNorm) != d || len(l.idxKNormB) != d {
		return fmt.Errorf("model: block %d's indexer is q %dx%d, k %dx%d, weights %dx%d, norm %d/%d; "+
			"want %d heads of %d over a %d-wide query latent", bi, l.idxQB.k, l.idxQB.rows, l.idxK.k,
			l.idxK.rows, l.idxProj.k, l.idxProj.rows, len(l.idxKNorm), len(l.idxKNormB), h, d, cfg.QLoraRank)
	}
	return nil
}

// allocIndexer sizes the decode token's indexer buffers.
func (s *State) allocIndexer() {
	c := s.m.Cfg
	if !c.Indexer() {
		return
	}
	s.idxQ = make([]float32, c.IdxHeads*c.IdxHeadDim)
	s.idxW = make([]float32, c.IdxHeads)
	s.idxTmp = make([]float32, s.attStride)
	s.idxScore = make([]float32, s.attStride)
	s.idxBias = make([]float32, s.attStride)
	s.idxOrd.Reserve(s.maxSeq)
}

// growIndexerRows sizes a chunk's: each row's query, weights and mask.
func (s *State) growIndexerRows(n int) {
	c := s.m.Cfg
	if !c.Indexer() || len(s.bidxW) >= n*c.IdxHeads {
		return
	}
	s.bidxQ = make([]float32, n*c.IdxHeads*c.IdxHeadDim)
	s.bidxW = make([]float32, n*c.IdxHeads)
	s.bidxK = make([]float32, n*c.IdxHeadDim)
	s.bidxBias = make([]float32, n*s.attStride)
	s.idxMasks = make([][]float32, n)
}

// idxKey writes one position's indexer key into row's tail: the projection of
// the block input x, its LayerNorm, and the rotary at table cs.
func (s *State) idxKey(row []float32, l *layer, x, cs []float32) error {
	c := s.m.Cfg
	k := row[c.KVLoraRank+c.NRot : c.KVDim()]
	if err := s.mv(k, l.idxK, x); err != nil {
		return err
	}
	s.idxKeyRest(k, l, cs)
	return nil
}

// idxKeyRest is idxKey after the projection: the norm and the rotary.
func (s *State) idxKeyRest(k []float32, l *layer, cs []float32) {
	nn.LayerNorm32JIT(k, k, l.idxKNorm, l.idxKNormB, idxEps)
	if !s.m.Cfg.NoPosEnc {
		nn.RoPE32JIT(k, len(k), cs, s.m.Cfg.idxFault != idxFaultPairRope)
	}
}

// idxQuery fills q (IdxHeads*IdxHeadDim) and w (IdxHeads) for one position:
// the query from the normed query latent qa, rotated at cs, and the weights
// from the block input x, scaled.
func (s *State) idxQuery(q, w []float32, l *layer, qa, x, cs []float32) error {
	if err := s.mv(q, l.idxQB, qa); err != nil {
		return err
	}
	if err := s.mv(w, l.idxProj, x); err != nil {
		return err
	}
	s.idxQueryRest(q, w, cs)
	return nil
}

// idxQueryRest is idxQuery after the projections.
func (s *State) idxQueryRest(q, w, cs []float32) {
	c := s.m.Cfg
	if !c.NoPosEnc {
		nn.RoPE32JIT(q, c.IdxHeadDim, cs, c.idxFault != idxFaultPairRope)
	}
	s.scale(w, float32(1/math.Sqrt(float64(c.IdxHeadDim*c.IdxHeads))))
}

// idxMask fills bias[0:n) with the selection for a query whose window is the
// first n cached positions of sequence slot in layer li's history: 0 at the
// IdxTopK positions the indexer scores highest and -inf everywhere else. It
// returns nil when n is at most IdxTopK, where every position is kept and the
// attention is dense. The window's pages must be resident (kvEnsureWindow).
//
// tmp and score are n-long scratch; ord is the selection's state. A caller
// running rows in parallel hands each its own.
func (s *State) idxMask(li, slot, n int, q, w, tmp, score, bias []float32, ord *nn.SampleOrder) []float32 {
	c := s.m.Cfg
	if n <= c.IdxTopK {
		return nil
	}
	d := c.IdxHeadDim
	score, tmp, bias = score[:n], tmp[:n], bias[:n]
	clear(score)
	for h := 0; h < c.IdxHeads; h++ {
		s.idxScores(li, slot, q[h*d:(h+1)*d], tmp)
		if c.idxFault != idxFaultNoReLU {
			nn.Act32JIT(tmp, nn.ActReLU)
		}
		s.axpy(score, tmp, w[h])
	}
	ninf := float32(math.Inf(-1))
	for i := range bias {
		bias[i] = ninf
	}
	ord.Begin(score)
	for range c.IdxTopK {
		_, id, ok := ord.Next()
		if !ok {
			break
		}
		bias[id] = 0
	}
	return bias
}

// idxScores fills dst[t] = q . k_t for the first len(dst) positions of slot's
// history in layer li, reading each cached row's indexer tail.
func (s *State) idxScores(li, slot int, q, dst []float32) {
	c := s.m.Cfg
	pg := &s.kv.layers[li]
	off := s.kvl.slots(c.KVLoraRank + c.NRot)
	for pos, end := 0, len(dst); pos < end && pg.p > 0; {
		sp := pg.span(pos, end)
		if !pg.resident(sp.page) {
			return
		}
		s.idxAttn.AttnScores(dst[pos:], pg.kSpan(s.kvl, sp, slot, 0)[off:], q, sp.n)
		pos += sp.n
	}
}

// idxApply adds a row's selection mask to scaled scores af, whose window
// starts at w0; nil is no mask.
func (s *State) idxApply(af, bias []float32, w0 int) {
	if bias != nil {
		s.axpy(af, bias[w0:w0+len(af)], 1)
	}
}

// idxRows computes the selection for each of n rows before a batched
// attention: row i's query and weights are in bidxQ/bidxW, its history is
// slot(i)'s first win(i) positions of layer li, and its mask lands in
// bidxBias at i*attStride (masks[i], nil for a dense row).
func (s *State) idxRows(li, n int, slot, win func(i int) int, masks [][]float32) {
	c := s.m.Cfg
	hq := c.IdxHeads * c.IdxHeadDim
	for i := 0; i < n; i++ {
		b := s.bidxBias[i*s.attStride : (i+1)*s.attStride]
		masks[i] = s.idxMask(li, slot(i), win(i), s.bidxQ[i*hq:(i+1)*hq],
			s.bidxW[i*c.IdxHeads:(i+1)*c.IdxHeads], s.idxTmp, s.idxScore, b, &s.idxOrd)
	}
}
