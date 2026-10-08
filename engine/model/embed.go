package model

import (
	"fmt"
	"math"

	"github.com/samyfodil/jitllm/engine/nn"
	"github.com/samyfodil/jitllm/format/jlm"
	"github.com/samyfodil/jitllm/format/quant"
	"github.com/samyfodil/jitllm/jit/cpu"
)

// Embeddings: a sequence in, one L2-normalised vector out.
//
// One API serves two kinds of model. An encoder (BERT, nomic-bert) has no
// decoder State and runs encode(); a decoder embedding model (qwen3-embedding,
// EmbeddingGemma) is a language model whose container says how to pool its
// last residual and whether attention is bidirectional, and runs through
// State.embedPrefill. The same three lines serve both:
//
//	e, _ := m.NewEmbedder()
//	defer e.Close()
//	v, _ := e.EmbedText("some text")

// embedMaxSeq caps a decoder embedding session's context: the KV cache is
// committed as it grows, so this bounds the worst case and costs nothing below it.
const embedMaxSeq = 8192

// Embedder is the working memory for embedding one sequence at a time. It is
// not safe for concurrent use; make one per goroutine.
type Embedder struct {
	m   *Model
	jit *nn.JIT
	st  *State // a decoder embedding model's session; nil for an encoder

	// The encoder's scratch, grown to the longest sequence seen.
	cap                       int
	x, xb, q, k, v, t, ff, gg []float32
	att, acc, cs              []float32
	attStride                 int
	pooled, norm, ones, dense []float32

	// afterBlock, when set, sees the residual after every encoder block --
	// the per-layer bisection hook the correctness gates use.
	afterBlock func(li int, x []float32)
	// skipDense reads the vector out before any projection head, matching
	// llama.cpp on a GGUF converted without the dense modules. Tests only.
	skipDense bool
}

// Pooling is how this model reads an embedding out of its last residual, or
// jlm.PoolNone for a model that is not an embedding model.
func (m *Model) Pooling() jlm.Pooling { return m.Cfg.Pooling }

// IsEmbedding reports whether the container describes an embedding model.
func (m *Model) IsEmbedding() bool { return m.Cfg.Pooling != jlm.PoolNone }

// IsEncoder reports whether the model is an encoder (BERT, nomic-bert): it has
// no decoder State, so NewEmbedder is its only path and NewState refuses it.
func (m *Model) IsEncoder() bool { return m.enc != nil }

// NewEmbedder builds the working memory Embed needs. A model whose container
// carries no pooling is refused: its last residual has no readout the weights
// were trained for.
func (m *Model) NewEmbedder() (*Embedder, error) {
	if !m.IsEmbedding() {
		return nil, fmt.Errorf("model: %s is not an embedding model (its container sets no pooling)", m.Cfg.Arch)
	}
	e := &Embedder{m: m}
	c := m.Cfg
	if m.enc == nil {
		st := m.NewState(min(c.NCtx, embedMaxSeq))
		e.st, e.jit = st, st.jit
	} else {
		types := []quant.Type{quant.F32, m.embd.typ}
		for i := range m.enc.blocks {
			for _, w := range m.enc.blocks[i].weights() {
				types = append(types, w.typ)
			}
		}
		e.jit = nn.NewJIT(max(c.NEmbd, c.NFFN), max(c.NEmbd, c.NFFN), types, m.jit...)
		if e.jit == nil {
			return nil, fmt.Errorf("model: no code generator for the encoder on this host")
		}
		// f32 k and v at the residual's stride: the encoder's k/v are its
		// projections, rows of NEmbd, the layout the tower's attention reads.
		e.jit.AddAttn(c.HeadDim, c.NEmbd, cpu.KVF32)
	}
	e.ones = make([]float32, c.NEmbd)
	for i := range e.ones {
		e.ones[i] = 1
	}
	e.pooled = make([]float32, c.NEmbd)
	e.norm = make([]float32, c.NEmbd)
	return e, nil
}

// Close releases the embedder's worker pool or session.
func (e *Embedder) Close() {
	if e.st != nil {
		e.st.Close()
		return
	}
	if e.jit != nil {
		e.jit.Close()
	}
}

// MaxTokens is the longest sequence Embed takes: a decoder's session context,
// an encoder's position table, or for a rotary encoder (nomic-bert), which has
// no table to run out of, the context it was trained at.
func (e *Embedder) MaxTokens() int {
	switch {
	case e.st != nil:
		return e.st.maxSeq
	case !e.m.enc.swiglu:
		return e.m.enc.pos.rows
	}
	return e.m.Cfg.NCtx
}

// grow sizes the encoder scratch for an n-token sequence.
func (e *Embedder) grow(n int) {
	if n <= e.cap {
		return
	}
	c := e.m.Cfg
	d := c.NEmbd
	e.cap = n
	e.x, e.xb = make([]float32, n*d), make([]float32, n*d)
	e.q, e.k, e.v, e.t = make([]float32, n*d), make([]float32, n*d), make([]float32, n*d), make([]float32, n*d)
	e.ff = make([]float32, n*c.NFFN, ffnPad(n*c.NFFN))
	e.gg = make([]float32, n*c.NFFN, ffnPad(n*c.NFFN))
	e.attStride = nn.SoftmaxPad(n)
	e.att = make([]float32, c.NHead*e.attStride)
	e.acc = make([]float32, c.NHead*c.HeadDim)
	e.cs = make([]float32, n*max(c.NRot, 2))
}

// EmbedText tokenizes text the way the model's own tokenizer does -- the
// special tokens it adds, and for a last-token pooled decoder the EOS it pools
// -- and embeds it.
func (e *Embedder) EmbedText(text string) ([]float32, error) {
	v := e.m.Vocab
	if v == nil {
		return nil, fmt.Errorf("model: no tokenizer: %v", e.m.TokErr)
	}
	return e.Embed(e.m.EmbedIDs(text))
}

// EmbedIDs is the token sequence EmbedText embeds.
//
// For a last-token pooled decoder the EOS is part of the input: qwen3-embedding
// pools at its <|endoftext|> (add_eos_token). tok.Encode does not add it,
// because a generation prompt must never carry one, so this is the one caller
// that acts on the flag. WordPiece's [SEP] is appended by Encode itself.
func (m *Model) EmbedIDs(text string) []int32 {
	v := m.Vocab
	ids := v.Encode(text, true)
	if m.enc == nil && v.AddEOS && v.EOS >= 0 && (len(ids) == 0 || ids[len(ids)-1] != v.EOS) {
		ids = append(ids, v.EOS)
	}
	return ids
}

// Embed runs one token sequence and returns its pooled, L2-normalised
// embedding. The slice is the embedder's own and is overwritten by the next
// call; copy it to keep it.
func (e *Embedder) Embed(ids []int32) ([]float32, error) {
	if len(ids) == 0 {
		return nil, fmt.Errorf("model: Embed called with no tokens")
	}
	if n := e.MaxTokens(); n > 0 && len(ids) > n {
		return nil, fmt.Errorf("model: %d tokens, and this model embeds at most %d", len(ids), n)
	}
	c := e.m.Cfg
	d := c.NEmbd
	var rows []float32 // n rows of the final hidden state, or the one pooled
	switch {
	case e.m.enc != nil:
		if err := e.encode(ids); err != nil {
			return nil, err
		}
		rows = e.x[:len(ids)*d]
	default:
		var err error
		if rows, err = e.st.embedPrefill(ids, e.pooled); err != nil {
			return nil, err
		}
	}
	out := e.pooled
	if e.m.enc != nil {
		e.pool(out, rows, len(ids))
	}
	if !e.skipDense {
		if err := e.m.denseHeads(e, out); err != nil {
			return nil, err
		}
	}
	e.normalize(out)
	return out, nil
}

// pool reads one vector out of n rows of the last residual.
func (e *Embedder) pool(out, rows []float32, n int) {
	d := e.m.Cfg.NEmbd
	switch e.m.Cfg.Pooling {
	case jlm.PoolCLS:
		copy(out, rows[:d])
	case jlm.PoolLast:
		copy(out, rows[(n-1)*d:n*d])
	default: // mean
		for i := range out {
			out[i] = 0
		}
		inv := 1 / float32(n)
		for r := 0; r < n; r++ {
			nn.Axpy32JIT(out, rows[r*d:(r+1)*d], inv)
		}
	}
}

// normalize scales v to unit L2 norm, on generated code: an RMSNorm with a
// weight of ones is v/sqrt(mean(v^2)), and 1/sqrt(len) turns the mean into the
// sum. The eps is sentence-transformers' 1e-12 floor on the norm, squared and
// taken per element so the two agree on everything but a zero vector.
func (e *Embedder) normalize(v []float32) {
	n := len(v)
	nn.RMSNorm32JIT(e.norm[:n], v, e.ones[:n], 1e-24/float64(n))
	copy(v, e.norm[:n])
	nn.Scale32JIT(v, float32(1/math.Sqrt(float64(n))))
}
