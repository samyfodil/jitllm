package model

import (
	"fmt"
	"math"
	"runtime"

	"github.com/samyfodil/jitllm/engine/nn"
	"github.com/samyfodil/jitllm/format/jlm"
	"github.com/samyfodil/jitllm/format/quant"
	"github.com/samyfodil/jitllm/tok"
)

// The BERT-family encoder: the graph behind the bert and nomic-bert embedding
// models (all-minilm, mxbai-embed-large, bge-large, snowflake-arctic-embed,
// nomic-embed-text).
//
// A BERT block is the vision tower's ViT block in post-norm order:
// x = LN(x + attn(x)), then x = LN(x + mlp(x)), with bidirectional attention.
// It runs on the tower's generated kernels. nomic-bert swaps the position
// table for NEOX rotary and the GELU MLP for SwiGLU.
//
// It is not a decoder: no KV cache, no persistent position, no logits. A
// sequence goes in whole and one vector comes out, so it is reached through
// an Embedder rather than NewState.

// encBlock is one post-norm encoder block.
type encBlock struct {
	wq, wk, wv, wo tensor
	bq, bk, bv, bo []float32
	ln1W, ln1B     []float32 // LN over x + attention
	up, gate, down tensor    // gate is empty for BERT's ungated MLP
	bUp, bDown     []float32
	ln2W, ln2B     []float32 // LN over x + MLP
	pageIndex      int
}

// encoder is the part of a Model that only an encoder has.
type encoder struct {
	typ0     []float32 // token-type row 0 ("sentence A"), or nil
	pos      tensor    // learned absolute positions; empty for nomic-bert
	lnW, lnB []float32 // the LayerNorm over the summed embeddings
	blocks   []encBlock
	rope     *nn.Rope // nomic-bert's rotary, nil for BERT
	swiglu   bool
	mb       *modernBERT // ModernBERT's own graph; nil for BERT and nomic-bert
	// zeros is the bias a LayerNorm without one is handed on a device
	// (encZeros).
	zeros []float32
}

// buildEncoder loads an encoder container. It touches no block page: every
// norm and bias is Expanded (dense), and the matrices are bound by pageIn as
// Embed reaches them -- the same O(bytes read) Open the text model has.
func buildEncoder(c *jlm.File, m *Model) error {
	cfg := m.Cfg
	get := func(role jlm.Role, block int32) (tensor, error) {
		e, ok := c.Find(role, block, -1)
		if !ok {
			return tensor{}, fmt.Errorf("model: missing %v (block %d)", role, block)
		}
		if e.NDim != 2 {
			return tensor{}, fmt.Errorf("model: %v has %d dimensions, want 2", role, e.NDim)
		}
		typ, ok := jlm.SourceType(e.Type)
		if !ok || !quant.Dequantable(typ) {
			return tensor{}, fmt.Errorf("model: %v is %v, which has no dequantizer yet", role, e.Type)
		}
		qs, _, _ := c.Span(e)
		t := tensor{e: e, typ: typ, data: qs, rows: int(e.Dims[1]), k: int(e.Dims[0])}
		if jlm.Packed(e.Type) {
			t.packed = &nn.Packed{}
		}
		return t, nil
	}
	vec := func(role jlm.Role, block int32, want int) ([]float32, error) {
		e, ok := c.Find(role, block, -1)
		if !ok {
			return nil, fmt.Errorf("model: missing %v (block %d)", role, block)
		}
		typ, ok := jlm.SourceType(e.Type)
		if !ok {
			return nil, fmt.Errorf("model: %v is %v, which has no dequantizer", role, e.Type)
		}
		n := 1
		for d := 0; d < int(e.NDim); d++ {
			n *= int(e.Dims[d])
		}
		if want > 0 && n != want {
			return nil, fmt.Errorf("model: %v (block %d) has %d values, want %d", role, block, n, want)
		}
		qs, _, _ := c.Span(e)
		out := make([]float32, n)
		if err := quant.Dequant32(typ, qs, out); err != nil {
			return nil, fmt.Errorf("model: %v: %w", role, err)
		}
		return out, nil
	}
	// optVec: the presence of the tensor is the fact.
	optVec := func(role jlm.Role, block int32, want int) ([]float32, error) {
		if !c.Has(role, block, -1) {
			return nil, nil
		}
		return vec(role, block, want)
	}
	enc := &encoder{swiglu: cfg.Arch == jlm.ArchNomicBERT.String()}
	d := cfg.NEmbd
	var err error
	if m.embd, err = get(jlm.RoleTokenEmbd, jlm.DenseBlock); err != nil {
		return err
	}
	if cfg.Arch == jlm.ArchModernBERT.String() {
		if err := buildModernBERT(c, m, enc); err != nil {
			return err
		}
		m.enc = enc
		m.layers = make([]layer, m.encBlocks())
		return nil
	}
	if c.Has(jlm.RoleTokenTypes, jlm.DenseBlock, -1) {
		tt, err := vec(jlm.RoleTokenTypes, jlm.DenseBlock, 0)
		if err != nil {
			return err
		}
		if len(tt) < d {
			return fmt.Errorf("model: the token-type table holds %d values, want at least %d", len(tt), d)
		}
		enc.typ0 = tt[:d]
	}
	if enc.lnW, err = vec(jlm.RoleTokenEmbdNorm, jlm.DenseBlock, d); err != nil {
		return err
	}
	if enc.lnB, err = vec(jlm.RoleTokenEmbdNormBias, jlm.DenseBlock, d); err != nil {
		return err
	}
	if enc.swiglu {
		// No position table, and its absence is required: a nomic-bert that
		// also carried one would have two positional encodings.
		if c.Has(jlm.RolePosEmbd, jlm.DenseBlock, -1) {
			return fmt.Errorf("model: %s carries a position table and a rotary; one of them is not this graph", cfg.Arch)
		}
		enc.rope = &nn.Rope{NRot: cfg.NRot, Base: cfg.RopeBase, Neox: cfg.RopeNeox, Scale: 1}
	} else if enc.pos, err = get(jlm.RolePosEmbd, jlm.DenseBlock); err != nil {
		return err
	}
	enc.blocks = make([]encBlock, cfg.NLayer)
	for i := range enc.blocks {
		b, bi := &enc.blocks[i], int32(i)
		b.pageIndex = i
		for _, w := range []struct {
			role jlm.Role
			dst  *tensor
		}{{jlm.RoleAttnQ, &b.wq}, {jlm.RoleAttnK, &b.wk}, {jlm.RoleAttnV, &b.wv},
			{jlm.RoleAttnOut, &b.wo}, {jlm.RoleFFNUp, &b.up}, {jlm.RoleFFNDown, &b.down}} {
			if *w.dst, err = get(w.role, bi); err != nil {
				return err
			}
		}
		if enc.swiglu {
			if b.gate, err = get(jlm.RoleFFNGate, bi); err != nil {
				return err
			}
		} else if c.Has(jlm.RoleFFNGate, bi, -1) {
			return fmt.Errorf("model: block %d of a BERT carries a gate; its MLP is ungated", i)
		}
		for _, v := range []struct {
			role jlm.Role
			dst  *[]float32
			want int
			opt  bool
		}{
			{jlm.RoleAttnQBias, &b.bq, d, true}, {jlm.RoleAttnKBias, &b.bk, d, true},
			{jlm.RoleAttnVBias, &b.bv, d, true}, {jlm.RoleAttnOutBias, &b.bo, d, true},
			{jlm.RoleFFNUpBias, &b.bUp, cfg.NFFN, true}, {jlm.RoleFFNDownBias, &b.bDown, d, true},
			{jlm.RoleAttnOutNorm, &b.ln1W, d, false}, {jlm.RoleAttnOutNormBias, &b.ln1B, d, false},
			{jlm.RoleLayerOutNorm, &b.ln2W, d, false}, {jlm.RoleLayerOutNormBias, &b.ln2B, d, false},
		} {
			if v.opt {
				*v.dst, err = optVec(v.role, bi, v.want)
			} else {
				*v.dst, err = vec(v.role, bi, v.want)
			}
			if err != nil {
				return err
			}
		}
		// The shapes the graph below assumes, checked once here rather than
		// discovered as a short buffer inside a kernel.
		for _, w := range []struct {
			t       tensor
			rows, k int
		}{{b.wq, d, d}, {b.wk, d, d}, {b.wv, d, d}, {b.wo, d, d},
			{b.up, cfg.NFFN, d}, {b.down, d, cfg.NFFN}} {
			if w.t.rows != w.rows || w.t.k != w.k {
				return fmt.Errorf("model: %s is %dx%d, want %dx%d", w.t.name(), w.t.rows, w.t.k, w.rows, w.k)
			}
		}
	}
	if cfg.NHead*cfg.HeadDim != d {
		return fmt.Errorf("model: %d heads of %d do not make a %d-wide residual", cfg.NHead, cfg.HeadDim, d)
	}
	m.enc = enc
	// One zero entry a block: the encoder's blocks are encBlock's, and the
	// segment State a device places them through (encdev.go) indexes the
	// model's block list.
	m.layers = make([]layer, m.encBlocks())
	return nil
}

func (b *encBlock) weights() []*tensor {
	ws := []*tensor{&b.wq, &b.wk, &b.wv, &b.wo, &b.up, &b.down}
	if b.gate.e != nil {
		ws = append(ws, &b.gate)
	}
	return ws
}

// encPageIn makes block li's matrices readable and binds them. It is
// Model.pageIn for the encoder, and it does not short-circuit on Resident for
// the reason pageIn gives: a claimed frame is not a filled one.
func (m *Model) encPageIn(li int) error {
	c := m.container
	if c == nil || li >= int(c.H.NBlocks) {
		return nil
	}
	b := &m.enc.blocks[li]
	var rs []jlm.Range
	for _, w := range b.weights() {
		rs = appendSpans(rs, c, w.e)
	}
	if err := c.EnsureRanges(li, rs); err != nil {
		return err
	}
	m.bind.Lock()
	defer m.bind.Unlock()
	for _, w := range b.weights() {
		bindOne(c, w)
	}
	return nil
}

// encode runs the encoder over one sequence and leaves the last block's
// residual in e.x, n rows of NEmbd.
func (e *Embedder) encode(ids []int32) error {
	m, c, enc := e.m, e.m.Cfg, e.m.enc
	n, d := len(ids), c.NEmbd
	if !enc.swiglu && n > enc.pos.rows {
		return fmt.Errorf("model: %d tokens, and the position table has %d rows", n, enc.pos.rows)
	}
	e.grow(n)
	x := e.x[:n*d]
	// 1. Token + token-type + position, then the embedding LayerNorm. The type
	// row is "sentence A" for every token, which is what llama.cpp hardcodes and
	// what sentence-transformers passes for a single text.
	for i, id := range ids {
		r := x[i*d : (i+1)*d]
		if err := m.embedRow(r, int(id)); err != nil {
			return err
		}
		if enc.typ0 != nil {
			nn.Axpy32JIT(r, enc.typ0, 1)
		}
		if !enc.swiglu {
			nn.Row32JIT(e.t[:d], enc.pos.typ, enc.pos.data, i, d)
			nn.Axpy32JIT(r, e.t[:d], 1)
		}
	}
	e.rowsLN(x, x, enc.lnW, enc.lnB, n)
	// The rotary tables, one per position, built once for every block.
	if enc.rope != nil {
		for i := 0; i < n; i++ {
			e.jit.RopeTable(*enc.rope, e.cs[i*c.NRot:(i+1)*c.NRot], i)
		}
	}
	scale := float32(1 / math.Sqrt(float64(c.HeadDim)))
	if err := e.devPrepare(n, 0); err != nil {
		return err
	}
	var cs []float32
	if enc.rope != nil {
		cs = e.cs[:n*c.NRot]
	}
	for li := 0; li < len(enc.blocks); li++ {
		// A run of placed blocks is one device call over the sequence.
		if e.onDevice(li) {
			hi, err := e.devRun(li, len(enc.blocks), n, x, cs, nil)
			if err != nil {
				return err
			}
			li = hi - 1
			if e.afterBlock != nil {
				e.afterBlock(li, x)
			}
			continue
		}
		if err := m.encPageIn(li); err != nil {
			return err
		}
		b := &enc.blocks[li]
		q, k, v := e.q[:n*d], e.k[:n*d], e.v[:n*d]
		for _, p := range []struct {
			w   tensor
			b   []float32
			dst []float32
		}{{b.wq, b.bq, q}, {b.wk, b.bk, k}, {b.wv, b.bv, v}} {
			if err := e.mm(p.dst, p.w, x, n); err != nil {
				return err
			}
			e.addBiasRows(p.dst, p.b, n)
		}
		if enc.rope != nil {
			for i := 0; i < n; i++ {
				cs := e.cs[i*c.NRot : (i+1)*c.NRot]
				nn.RoPE32JIT(q[i*d:(i+1)*d], c.HeadDim, cs, enc.rope.Neox)
				nn.RoPE32JIT(k[i*d:(i+1)*d], c.HeadDim, cs, enc.rope.Neox)
			}
		}
		// 2. Bidirectional attention: every position attends to all n, so the
		// key count is n for every query rather than pos+1. On the pool over
		// heads, each with its own score row and accumulator.
		xb := e.xb[:n*d]
		e.jit.Parallel(c.NHead, 1, func(lo, hi int) {
			for h := lo; h < hi; h++ {
				off := h * c.HeadDim
				att := e.att[h*e.attStride : h*e.attStride+n]
				acc := e.acc[h*c.HeadDim : (h+1)*c.HeadDim]
				for i := 0; i < n; i++ {
					e.jit.AttnScores(att, k[off:], q[i*d+off:], n)
					nn.Scale32JIT(att, scale)
					nn.Softmax32JIT(att, n)
					e.jit.AttnAcc(acc, v[off:], att, n)
					copy(xb[i*d+off:i*d+off+c.HeadDim], acc)
				}
			}
		})
		// 3. x = LN(x + o(attn) + bo): post-norm, the whole difference from the
		// tower's block.
		t := e.t[:n*d]
		if err := e.mm(t, b.wo, xb, n); err != nil {
			return err
		}
		e.addBiasRows(t, b.bo, n)
		e.axpyRows(t, x)
		e.rowsLN(x, t, b.ln1W, b.ln1B, n)
		// 4. The MLP and x = LN(x + mlp(x)).
		ff := e.ff[:n*c.NFFN]
		if err := e.mm(ff, b.up, x, n); err != nil {
			return err
		}
		e.addBiasRows(ff, b.bUp, n)
		if enc.swiglu {
			g := e.gg[:n*c.NFFN]
			if err := e.mm(g, b.gate, x, n); err != nil {
				return err
			}
			// silu(gate) * up, written into g: fc11 is up and fc12 the gate
			// in nomic's own module, and llama.cpp's LLM_FFN_PAR agrees.
			e.parallelVec(len(g), func(lo, hi int) { nn.ActMul32JIT(g[lo:hi], ff[lo:hi], nn.ActSiLU) })
			ff = g
		} else {
			e.parallelVec(len(ff), func(lo, hi int) { nn.Act32JIT(ff[lo:hi], nn.ActGELU) })
		}
		if err := e.mm(t, b.down, ff, n); err != nil {
			return err
		}
		e.addBiasRows(t, b.bDown, n)
		e.axpyRows(t, x)
		e.rowsLN(x, t, b.ln2W, b.ln2B, n)
		if e.afterBlock != nil {
			e.afterBlock(li, x)
		}
	}
	return nil
}

// mm is out = W x for n rows: the packed batch when the weight is packed, and
// the generated float matvec per row when the container carries it verbatim
// (an F16 or F32 BERT, which is every one Ollama ships).
func (e *Embedder) mm(out []float32, w tensor, x []float32, n int) error {
	if w.packed != nil {
		if e.jit.MatMulPacked(out, w.typ, w.packed, x, w.rows, w.k, n) {
			return nil
		}
		for i := 0; i < n; i++ {
			e.jit.NewInput()
			if !e.jit.MatVecPacked(out[i*w.rows:(i+1)*w.rows], w.typ, w.packed, x[i*w.k:(i+1)*w.k], w.rows, w.k) {
				return fmt.Errorf("jitllm: no host kernel reads %s (%s) on %s", w.name(), w.typ, runtime.GOARCH)
			}
		}
		return nil
	}
	// A float matrix (F32, F16, BF16) reads each row once per tile of tokens
	// on the float GEMM; MatVec per token is the path for a type it declines.
	if n > 1 && e.jit.MatMulFloat(out, w.typ, w.data, x, w.rows, w.k, n) {
		return nil
	}
	for i := 0; i < n; i++ {
		if !e.jit.MatVec(out[i*w.rows:(i+1)*w.rows], w.typ, w.data, x[i*w.k:(i+1)*w.k], w.rows, w.k) {
			return fmt.Errorf("jitllm: no host kernel reads %s (%s, %dx%d) on %s",
				w.name(), w.typ, w.rows, w.k, runtime.GOARCH)
		}
	}
	return nil
}

// rowsLN is y[r] = LayerNorm(x[r]) for n rows, on the pool.
func (e *Embedder) rowsLN(y, x, w, b []float32, n int) {
	d, eps := e.m.Cfg.NEmbd, e.m.Cfg.RMSEps
	e.jit.Parallel(n, 1, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			nn.LayerNorm32JIT(y[r*d:(r+1)*d], x[r*d:(r+1)*d], w, b, eps)
		}
	})
}

// addBiasRows adds b to each of n rows of width len(b); nil is no bias.
func (e *Embedder) addBiasRows(out, b []float32, n int) {
	if b == nil {
		return
	}
	w := len(b)
	e.jit.Parallel(n, 1, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			nn.Axpy32JIT(out[r*w:(r+1)*w], b, 1)
		}
	})
}

// axpyRows is dst += src over the whole buffer.
func (e *Embedder) axpyRows(dst, src []float32) {
	e.parallelVec(len(dst), func(lo, hi int) { nn.Axpy32JIT(dst[lo:hi], src[lo:hi], 1) })
}

// parallelVec splits [0,n) on vector boundaries across the pool.
func (e *Embedder) parallelVec(n int, fn func(lo, hi int)) {
	lanes := nn.ElemLanes
	vecs := n / lanes
	w := e.jit.Workers()
	if vecs < 2*w || w < 2 {
		fn(0, n)
		return
	}
	e.jit.Parallel(vecs, elemChunk(vecs, w), func(lo, hi int) {
		b := hi * lanes
		if hi == vecs {
			b = n
		}
		fn(lo*lanes, b)
	})
}

// buildEncoderModel is build() for an encoder container: the config, the
// vocabulary, and the encoder graph. There is no output projection and no
// decoder layer, so m.layers stays empty and NewState is refused by name.
func buildEncoderModel(c *jlm.File, options ...tok.Option) (*Model, error) {
	cfg, err := configFrom(c.Config())
	if err != nil {
		return nil, err
	}
	if cfg.Pooling == jlm.PoolNone && c.Config().Decision == jlm.DecisionNone {
		return nil, fmt.Errorf("model: %s is an encoder with no pooling; the container "+
			"says nothing about how to read a vector out of it", cfg.Arch)
	}
	if !cfg.NonCausal {
		return nil, fmt.Errorf("model: %s is an encoder with a causal mask, which is not this graph", cfg.Arch)
	}
	m := &Model{Cfg: cfg}
	if m.Vocab, m.TokErr = tok.New(c.Vocab(), options...); m.TokErr != nil {
		m.Vocab = nil
	}
	if err := buildEncoder(c, m); err != nil {
		return nil, err
	}
	return m, nil
}

// loadDenseHeads binds an embedding model's projection heads, when the
// container carries them. Both or neither: one alone is a model this graph
// cannot express.
func loadDenseHeads(c *jlm.File, m *Model) error {
	has1 := c.Has(jlm.RoleEmbdDense1, jlm.DenseBlock, -1)
	has2 := c.Has(jlm.RoleEmbdDense2, jlm.DenseBlock, -1)
	if !has1 && !has2 {
		return nil
	}
	if has1 != has2 {
		return fmt.Errorf("model: the container carries one of two dense heads")
	}
	for _, h := range []struct {
		role jlm.Role
		dst  *tensor
	}{{jlm.RoleEmbdDense1, &m.dense1}, {jlm.RoleEmbdDense2, &m.dense2}} {
		e, _ := c.Find(h.role, jlm.DenseBlock, -1)
		typ, ok := jlm.SourceType(e.Type)
		if !ok || e.NDim != 2 {
			return fmt.Errorf("model: %v is %v in %d dimensions", h.role, e.Type, e.NDim)
		}
		qs, _, _ := c.Span(e)
		*h.dst = tensor{e: e, typ: typ, data: qs, rows: int(e.Dims[1]), k: int(e.Dims[0])}
		if jlm.Packed(e.Type) {
			h.dst.packed = &nn.Packed{}
		}
	}
	if m.dense1.k != m.Cfg.NEmbd || m.dense2.k != m.dense1.rows || m.dense2.rows != m.Cfg.NEmbd {
		return fmt.Errorf("model: dense heads %dx%d then %dx%d do not map %d back to itself",
			m.dense1.rows, m.dense1.k, m.dense2.rows, m.dense2.k, m.Cfg.NEmbd)
	}
	return nil
}
