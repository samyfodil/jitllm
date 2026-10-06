package model

import (
	"math"

	"github.com/samyfodil/jitllm/engine/nn"
)

// The gated delta net: the other half of a hybrid's block loop.
//
// The graph is transcribed from llama.cpp's running graph; six of its steps
// are ones a from-paper reading gets wrong. `llama-eval-callback` on
// Qwen3-Next-80B-A3B-Instruct prints layer 0 as:
//
//	conv_states       GET_ROWS(cache_r_l0{24576})   -> {3, 8192}
//	qkv_mixed         MUL_MAT(attn_qkv.weight{2048,8192}, attn_norm)
//	conv_input        CONCAT(conv_states, TRANSPOSE(qkv_mixed))   -> {4, 8192}
//	conv_state_last   VIEW(conv_input)[1:]          -> CPY into cache_r_l0
//	conv_output_raw   SSM_CONV(conv_input, ssm_conv1d.weight{4,8192})
//	conv_output_silu  SILU(conv_output_raw)                        <- (1)
//	q_conv/k_conv     VIEW{128,16} -> L2_NORM -> REPEAT to {128,32} <- (2),(3)
//	v_conv            VIEW{128,32}
//	mixed_ba          MUL_MAT(ssm_ba.weight{2048,64}, attn_norm) -> {4,16}
//	a                 VIEW{2,16} -> +ssm_dt.bias -> SOFTPLUS -> *ssm_a  <- (4)
//	b                 VIEW{2,16} -> SIGMOID                             <- (5)
//	node_53           GATED_DELTA_NET(q, k, v, gate, beta, state)
//	attn_output       VIEW -> RMS_NORM -> *ssm_norm.weight
//	z                 MUL_MAT(attn_gate.weight{2048,4096}, attn_norm)
//	final_output      node_59 * SILU(z)                                 <- (6)
//	linear_attn_out   MUL_MAT(ssm_out.weight{4096,2048}, final_output)
//
//	(1) the convolution's output is passed through SiLU before it is split.
//	(2) q and k are L2-normalised, not RMS-normalised: there is no weight, and
//	    the divisor is the vector's own length rather than its root mean square.
//	(3) there are 16 key heads and 32 value heads, and the repeat is
//	    interleaved: value head h reads key head h/2, not h%16.
//	(4) the decay is softplus(a + dt_bias) * ssm_a, and ssm_a is already
//	    -exp(A_log), so the product is negative and exp() of it is in (0,1).
//	(5) beta is a plain sigmoid, no bias.
//	(6) the output gate is SiLU(z), while the full-attention layers of the same
//	    model gate with sigmoid.
//
// The state is not a KV cache: it is the same size at every position, so
// residency, paging and placement cannot treat it as KV.

// kdaL2Eps is the epsilon Kimi Delta Attention's q/k L2 normalisation adds to
// the sum of squares. It is the reference helper's own default, not the
// model's rms_norm_eps (1e-5 on the checkpoints shipped so far).
const kdaL2Eps = 1e-6

// deltaGeom is the recurrent block's derived geometry. Every field is a
// function of Config.SSM; they are computed once per call rather than stored so
// that a Config stays a description of the file.
type deltaGeom struct {
	conv    int // convolution width, in positions
	chans   int // convolved channels: q + k + v
	kHeads  int // key/query heads
	vHeads  int // value heads, and the number of decay rates
	kDim    int // key/query head width, which is also the state's key extent
	vDim    int // value head width, which is also the state's value extent
	rep     int // value heads per key head
	qkBytes int // q and k each occupy kHeads*kDim of the convolved channels
	// chanDecay is Kimi Delta Attention rather than qwen3next's gated delta
	// rule: the decay is a vector per value head instead of one rate, and both
	// gates come off a rank-kDim bottleneck (f_a_proj and g_a_proj project to
	// linear_head_dim, which is kDim).
	chanDecay bool
	rank      int
	// tiled pairs value head vh with key head vh % kHeads instead of vh / rep:
	// the order llama.cpp's qwen35 converter writes the value heads in. See
	// keyHead.
	tiled bool
	// ssd is Mamba-2's selective update in place of the delta rule (see
	// ssdAttn): the same state, k is B, q is C and v is x, and the convolved
	// channels arrive in that order because the converter put them there.
	ssd bool
	// scale is 1/sqrt(kDim), which the L2 norm's rescale and the delta rule's
	// query scale both want.
	scale float32
	// mamba1 is Mamba-1's selective scan (mamba1Attn): every channel a value
	// head of one row (vHeads is Inner, vDim one), B and C one key head
	// shared by all, and only x convolved -- B and C come off it afterwards,
	// so chans is Inner. rank is its dt bottleneck.
	mamba1 bool
}

func (c *Config) delta() deltaGeom {
	g := deltaGeom{
		conv:   int(c.SSM.ConvKernel),
		kHeads: int(c.SSM.Groups),
		vHeads: int(c.SSM.NHeadV),
		kDim:   int(c.SSM.StateSize),
	}
	g.qkBytes = g.kHeads * g.kDim
	g.chans = int(c.SSM.Inner) + 2*g.qkBytes
	if g.vHeads > 0 {
		g.vDim = int(c.SSM.Inner) / g.vHeads
	}
	if g.kHeads > 0 {
		g.rep = g.vHeads / g.kHeads
	}
	if g.kDim > 0 {
		g.scale = 1 / float32(math.Sqrt(float64(g.kDim)))
	}
	g.chanDecay, g.rank = c.ChanDecay(), g.kDim
	g.tiled = c.DeltaKeyTiled
	g.ssd = c.SSD()
	if c.Mamba1() {
		g.mamba1, g.chans, g.qkBytes, g.rank = true, int(c.SSM.Inner), 0, c.dtRank
	}
	return g
}

// keyHead is the key/query head value head vh reads.
//
// Two pairings ship, and both are fluent when wrong. qwen3next's GGUF keeps
// HuggingFace's grouped order (value heads kh*rep .. kh*rep+rep-1 share key
// head kh); llama.cpp's qwen35 converter reorders to tiled order (value head
// vh shares key head vh % kHeads). They differ only when rep > 1.
func (g deltaGeom) keyHead(vh int) int {
	if g.tiled {
		return vh % g.kHeads
	}
	return vh / g.rep
}

// convStateLen and deltaStateLen are one sequence's recurrent state for one
// linear layer. Both are constant in the context length.
func (g deltaGeom) convStateLen() int  { return (g.conv - 1) * g.chans }
func (g deltaGeom) deltaStateLen() int { return g.vHeads * g.vDim * g.kDim }

// allocRecurrent sizes the recurrent state and the scratch the delta rule
// needs. Both are nil for every architecture that is not a hybrid.
//
// It is per sequence, like the KV cache: a batched session's rows share the
// weights and nothing else.
func (s *State) allocRecurrent(nseq int) {
	c := s.c
	if !c.Hybrid() || !s.linearIn() {
		return
	}
	g := c.delta()
	s.dg = g
	s.rconv = make([][]float32, c.NLayer)
	s.rstate = make([][]float32, c.NLayer)
	for i := 0; i < c.NLayer; i++ {
		if !c.LayerKind(i).Recurrent() {
			continue
		}
		s.rconv[i] = make([]float32, nseq*g.convStateLen())
		s.rstate[i] = make([]float32, nseq*g.deltaStateLen())
	}
	s.dMixed = make([]float32, g.chans)
	if g.mamba1 {
		s.dLowRank = make([]float32, g.rank)
		s.dSB, s.dSC = make([]float32, g.kDim), make([]float32, g.kDim)
		s.dZeroA = make([]float32, g.vHeads)
	}
	if c.ShortConv() {
		// B and C, the short convolution's two gates.
		s.dZ = make([]float32, 2*g.chans)
	} else {
		s.dZ = make([]float32, g.vHeads*g.vDim)
	}
	// KDA's decay is per channel, so its gate buffers are vHeads*kDim wide;
	// qwen3next's are vHeads.
	nGate := g.vHeads
	if g.chanDecay {
		nGate = g.vHeads * g.kDim
	}
	s.dBA = make([]float32, 2*g.vHeads)
	s.dDecay = make([]float32, nGate)
	s.dBeta = make([]float32, g.vHeads)
	s.dAlpha = make([]float32, nGate)
	s.dBRaw = make([]float32, g.vHeads)
	if g.chanDecay {
		s.dLowRank = make([]float32, g.rank)
		// The gate kernel writes both outputs, and KDA wants them at two
		// different lengths: the decay over vHeads*kDim and beta over vHeads.
		// dGateWaste receives whichever half the call in question is not for.
		s.dGateWaste = make([]float32, nGate)
	}
	s.dOut = make([]float32, g.vHeads*g.vDim)
	s.dOnes = make([]float32, g.kDim)
	for i := range s.dOnes {
		s.dOnes[i] = 1
	}
}

// linearIn reports whether any block this State runs is a linear one. A draft
// State over a prediction block runs attention alone, so it keeps no summary
// and recurrent() is false for it.
func (s *State) linearIn() bool {
	for li := s.lo; li < s.hi; li++ {
		if s.c.LayerKind(li).Recurrent() {
			return true
		}
	}
	return false
}

// ResetRecurrent clears one row's recurrent state, which is what retiring a
// sequence means for a linear layer: there is no per-position cache to
// invalidate, only a running summary to forget.
func (s *State) ResetRecurrent(row int) {
	for li := range s.rconv {
		if s.rconv[li] == nil {
			continue
		}
		n := s.dg.convStateLen()
		clear(s.rconv[li][row*n : (row+1)*n])
		n = s.dg.deltaStateLen()
		clear(s.rstate[li][row*n : (row+1)*n])
	}
}

// linearAttn runs one token of one sequence through layer li's gated delta net.
// h is the normed residual; out receives the block's output, n_embd wide, and
// is written rather than added to -- the caller folds it into the residual, the
// same contract the attention path has.
func (s *State) linearAttn(li int, l *layer, row int, h, out []float32) error {
	c, g := s.c, s.dg
	if g.ssd {
		return s.ssdAttn(li, l, row, h, out)
	}
	if g.mamba1 {
		return s.mamba1Attn(li, l, row, h, out)
	}
	if c.ShortConv() {
		return s.shortConv(li, l, row, h, out)
	}

	if err := s.mv(s.dMixed, l.wq, h); err != nil {
		return err
	}
	if g.chanDecay {
		// KDA's two gates are low-rank: two matvecs each. Folding f_b @ f_a
		// into one matrix at conversion would halve the dispatches and
		// multiply the bytes, the wrong trade for bandwidth-bound decode.
		if l.ssmGA.rows == 0 {
			// Kimi-K3's full-rank output gate: one matrix, the z projection's
			// place.
			if err := s.mv(s.dZ, l.ssmGate, h); err != nil {
				return err
			}
		} else {
			if err := s.mv(s.dLowRank, l.ssmGA, h); err != nil {
				return err
			}
			if err := s.mv(s.dZ, l.ssmGB, s.dLowRank); err != nil {
				return err
			}
		}
		if err := s.mv(s.dLowRank, l.ssmFA, h); err != nil {
			return err
		}
		if err := s.mv(s.dAlpha, l.ssmFB, s.dLowRank); err != nil {
			return err
		}
		// ssmBA is b_proj alone here: beta's logits, one per value head. There
		// is no alpha stream in it to deinterleave -- the forget gate above is
		// where KDA's alpha comes from.
		if err := s.mv(s.dBRaw, l.ssmBA, h); err != nil {
			return err
		}
	} else {
		if err := s.mv(s.dZ, l.ssmGate, h); err != nil {
			return err
		}
		if err := s.mv(s.dBA, l.ssmBA, h); err != nil {
			return err
		}
	}

	s.m.trace(li, "qkv_mixed", s.dMixed)
	s.conv1d(li, row, l.ssmConv1d)
	s.m.trace(li, "conv_output_raw", s.dMixed)
	nn.Act32JIT(s.dMixed, nn.ActSiLU)
	s.jit.NewInput()
	s.m.trace(li, "conv_output_silu", s.dMixed)

	// The gates are deinterleaved first, then computed in one generated call.
	// ba reshapes to {2*rep, n_head_k}, beta first and alpha second within
	// each key head's group; reading it as two contiguous halves pairs every
	// value head with the wrong decay.
	if g.chanDecay {
		// Two calls because the gates are different lengths here: the decay
		// runs over vHeads*kDim channels and beta over vHeads heads. Each call
		// keeps the half it wants and sends the other to dGateWaste. The
		// arithmetic is qwen3next's; ssm_a already holds -exp(A_log),
		// broadcast per channel at conversion.
		if lb := c.KDALowerBound; lb != 0 {
			// Kimi-K3's gate_lower_bound: lb*sigma(exp(A_log)*(f + dt)), where
			// Kimi-Linear softplusses.
			nn.DeltaDecayBound32JIT(s.dDecay, s.dAlpha, l.ssmDtBias, l.ssmA, lb)
		} else {
			nn.DeltaGate32JIT(s.dDecay, s.dGateWaste, s.dAlpha, l.ssmDtBias, s.dAlpha, l.ssmA)
		}
		n := g.vHeads
		nn.DeltaGate32JIT(s.dGateWaste[:n], s.dBeta, s.dAlpha[:n], l.ssmDtBias[:n], s.dBRaw, l.ssmA[:n])
	} else {
		for kh := 0; kh < g.kHeads; kh++ {
			base := kh * 2 * g.rep
			for r := 0; r < g.rep; r++ {
				vh := kh*g.rep + r
				s.dBRaw[vh] = s.dBA[base+r]
				s.dAlpha[vh] = s.dBA[base+g.rep+r]
			}
		}
		nn.DeltaGate32JIT(s.dDecay, s.dBeta, s.dAlpha, l.ssmDtBias, s.dBRaw, l.ssmA)
	}

	// q and k are L2-normalised per key head, which is rmsnorm with a unit
	// weight, eps/n, and a 1/sqrt(n) rescale -- the identity llama.cpp builds
	// the op from. Dividing eps by kDim makes it the constant added to the
	// sum of squares; KDA hardcodes 1e-6 there instead of rms_norm_eps.
	eps := c.RMSEps / float64(g.kDim)
	if g.chanDecay {
		eps = kdaL2Eps / float64(g.kDim)
	}
	for kh := 0; kh < 2*g.kHeads; kh++ {
		v := s.dMixed[kh*g.kDim : (kh+1)*g.kDim]
		s.rmsnorm(v, v, s.dOnes, eps)
		s.scale(v, g.scale)
	}
	// The query carries the attention scale, exactly once, here.
	s.scale(s.dMixed[:g.qkBytes], g.scale)

	s.m.trace(li, "q_conv_predelta", s.dMixed[:g.qkBytes])
	s.m.trace(li, "k_conv_predelta", s.dMixed[g.qkBytes:2*g.qkBytes])
	s.m.trace(li, "v_conv_predelta", s.dMixed[2*g.qkBytes:])
	s.m.trace(li, "gate", s.dDecay)
	s.m.trace(li, "beta", s.dBeta)
	s.delta(li, row)
	s.m.trace(li, "attn_output", s.dOut)

	// The gated output norm: RMSNorm per value head against ssm_norm, times
	// SiLU of the z projection.
	for vh := 0; vh < g.vHeads; vh++ {
		o := s.dOut[vh*g.vDim : (vh+1)*g.vDim]
		s.rmsnorm(o, o, l.ssmNorm, c.RMSEps)
	}
	if g.chanDecay && c.k3Fault == k3FaultNoKDAGate {
		copy(s.dZ, s.dOut)
	} else if g.chanDecay {
		// KDA's gate is out*sigma(z) (KimiLinearRMSNormGated), where
		// qwen3next's is SiLU(z)*out: they differ by a factor of z.
		nn.SigmoidMul32JIT(s.dZ, s.dOut)
	} else {
		nn.ActMul32JIT(s.dZ, s.dOut, nn.ActSiLU)
	}
	s.jit.NewInput()
	s.m.trace(li, "final_output", s.dZ)
	if err := s.mv(out, l.ssmOut, s.dZ); err != nil {
		return err
	}
	s.m.trace(li, "linear_attn_out", out)
	return nil
}

// conv1d runs the causal convolution over dMixed in place and advances the
// convolution's state.
//
// The state is plane-major and the shift is part of the kernel, so the
// reduction and the shift walk the state once (llama.cpp's CONCAT+VIEW reads
// it twice).
func (s *State) conv1d(li, row int, w []float32) {
	g := s.dg
	n := g.conv - 1
	st := s.rconv[li][row*g.convStateLen():][:n*g.chans]
	x := s.dMixed
	s.jit.Conv1d(x, st, w, x, g.conv, g.chans)
}

// delta applies one token of the delta rule to layer li's recurrent state and
// writes the per-head output into dOut.
//
// One generated call per value head. Per value head, with S[j][i] the state,
// k and q the convolved key and query of the corresponding key head, v the
// value, g the decay and b the gate, it is a rank-one update:
//
//	S       *= exp(g)
//	sk[j]    = sum_i S[j][i]*k[i]
//	d[j]     = (v[j] - sk[j]) * b
//	S[j][i] += k[i]*d[j]
//	o[j]     = sum_i S[j][i]*q[i]
//
// The two sums read the same row the two writes touch, so each row of the state
// is brought in twice rather than four times: decay and the key sum share one
// pass, the update and the query sum share the other.
func (s *State) delta(li, row int) {
	g := s.dg
	j := &s.rg.delta
	j.d = s.m.layers[li].ssmD
	j.st = s.rstate[li][row*g.deltaStateLen():]
	j.q = s.dMixed[:g.qkBytes]
	j.k = s.dMixed[g.qkBytes : 2*g.qkBytes]
	j.v = s.dMixed[2*g.qkBytes:]
	if j.fn == nil {
		j.fn = s.deltaHeads
	}
	s.jit.Parallel(g.vHeads, 1, j.fn)
	j.st, j.q, j.k, j.v, j.d = nil, nil, nil, nil, nil
}

// ssdNorm is a Mamba-2 block's grouped RMSNorm, in place on y: each group of
// Inner/SSMNormGroups channels against its own slice of w.
func (s *State) ssdNorm(y, w []float32) {
	c := s.m.Cfg
	gw := len(y) / c.SSMNormGroups
	for gi := 0; gi < c.SSMNormGroups; gi++ {
		v := y[gi*gw : (gi+1)*gw]
		s.rmsnorm(v, v, w[gi*gw:(gi+1)*gw], c.RMSEps)
	}
	s.jit.NewInput()
}

// ssdAttn runs one token of one sequence through layer li's Mamba-2 mixer;
// h, out and the aliasing contract are linearAttn's.
//
// llama.cpp's build_mamba2_layer, in the delta rule's terms (convert/ssm.go
// put the convolved channels in q | k | v = C | B | x order):
//
//	zxBCdt  MUL_MAT(ssm_in) -> z, xBC, dt       three projections here
//	xBC     SSM_CONV(conv state ++ xBC) + conv bias -> SILU
//	y       SSM_SCAN(x, dt + dt_bias, A, B, C)  softplus inside the scan
//	y      += x * D
//	y       = SWIGLU_SPLIT(z, y)                y * silu(z)
//	y       RMS_NORM over Inner/Groups, * ssm_norm   the GATE goes first
//	out     MUL_MAT(ssm_out, y)
func (s *State) ssdAttn(li int, l *layer, row int, h, out []float32) error {
	c := s.m.Cfg
	if err := s.mv(s.dMixed, l.mixIn(), h); err != nil {
		return err
	}
	if err := s.mv(s.dZ, l.ssmGate, h); err != nil {
		return err
	}
	if err := s.mv(s.dBRaw, l.ssmBA, h); err != nil {
		return err
	}
	s.m.trace(li, "xBC", s.dMixed)
	s.conv1d(li, row, l.ssmConv1d)
	s.addBias(s.dMixed, l.ssmConvB)
	nn.Act32JIT(s.dMixed, nn.ActSiLU)
	s.jit.NewInput()
	s.m.trace(li, "conv_output_silu", s.dMixed)

	// dt = softplus(dt + dt_bias) into dBeta and exp(A*dt) into dDecay: the
	// delta rule's two per-head scalars, with dt where beta was.
	nn.SSDGate32JIT(s.dDecay, s.dBeta, s.dBRaw, l.ssmDtBias, l.ssmA)
	s.m.trace(li, "dt", s.dBeta)
	s.delta(li, row)
	s.m.trace(li, "ssm_y", s.dOut)

	// The gate before the norm (Mamba's order, the reverse of qwen3next's),
	// and the norm over each group of Inner/Groups channels with its own
	// slice of the weight. A block with no norm is y*silu(z) alone.
	if l.ssmNorm != nil && c.SSMNormBeforeGate {
		s.ssdNorm(s.dOut, l.ssmNorm)
	}
	nn.ActMul32JIT(s.dZ, s.dOut, nn.ActSiLU)
	s.jit.NewInput()
	if l.ssmNorm != nil && !c.SSMNormBeforeGate {
		s.ssdNorm(s.dZ, l.ssmNorm)
	}
	s.m.trace(li, "ssm_norm", s.dZ)
	if err := s.mv(out, l.ssmOut, s.dZ); err != nil {
		return err
	}
	s.m.trace(li, "ssm_out", out)
	return nil
}

// splitQGate deinterleaves n rows of a double-width query projection. A
// hybrid's full-attention layers fold a sigmoid output gate into attn_q
// ({n_embd, 2*n_head*head_dim}), interleaved per head: head h owns
// [h*2*hd, h*2*hd+hd) as its query and the next head_dim as its gate.
func (s *State) splitQGate(q, gate, src []float32, n int) {
	c := s.c
	hd, qd := c.HeadDim, c.NHead*c.HeadDim
	for r := 0; r < n; r++ {
		in := src[r*2*qd:]
		for hh := 0; hh < c.NHead; hh++ {
			copy(q[r*qd+hh*hd:][:hd], in[hh*2*hd:][:hd])
			copy(gate[r*qd+hh*hd:][:hd], in[hh*2*hd+hd:][:hd])
		}
	}
}

// applyOutGate turns gate into sigmoid(gate)*att, in place, for n rows.
func (s *State) applyOutGate(gate, att []float32, n int) {
	qd := s.c.NHead * s.c.HeadDim
	nn.SigmoidMul32JIT(gate[:n*qd], att[:n*qd])
}

// projectQ runs the query projection for n rows, splitting off the output gate
// when the architecture folds one in. It returns the slice the attention should
// read as its query.
func (s *State) projectQ(l *layer, dst, gate, h []float32, n int, mm bool) ([]float32, error) {
	c := s.c
	if !c.AttnOutGate {
		var err error
		if mm {
			err = s.mm(dst, l.wq, h, n)
		} else {
			err = s.mv(dst, l.wq, h)
		}
		if err != nil {
			return nil, err
		}
		s.addBiasRows(dst, l.bq, n)
		return dst, nil
	}
	qg := s.qgate[:n*2*c.NHead*c.HeadDim]
	var err error
	if mm {
		err = s.mm(qg, l.wq, h, n)
	} else {
		err = s.mv(qg, l.wq, h)
	}
	if err != nil {
		return nil, err
	}
	s.addBiasRows(qg, l.bq, n)
	s.splitQGate(dst, gate, qg, n)
	return dst, nil
}

// mixIn is a recurrent block's mixed projection: wq, unless the block attends
// too and wq is the attention's query (layer.ssmIn).
func (l *layer) mixIn() tensor {
	if l.ssmIn.rows != 0 {
		return l.ssmIn
	}
	return l.wq
}

// shortConv runs one token of LFM2's gated short convolution: B*x through
// the depthwise convolution (its window is the layer's only state), C times
// that, then out_proj. Both products are the generated gated multiply with
// the identity as its activation, in place.
func (s *State) shortConv(li int, l *layer, row int, h, out []float32) error {
	n := s.dg.chans
	b, cg := s.dZ[:n], s.dZ[n:2*n]
	if err := s.mv(s.dMixed, l.wq, h); err != nil {
		return err
	}
	if err := s.mv(b, l.ssmGate, h); err != nil {
		return err
	}
	if err := s.mv(cg, l.ssmBA, h); err != nil {
		return err
	}
	nn.ActMul32JIT(s.dMixed, b, nn.ActIdentity)
	s.m.trace(li, "bx", s.dMixed)
	s.conv1d(li, row, l.ssmConv1d)
	s.m.trace(li, "conv", s.dMixed)
	nn.ActMul32JIT(s.dMixed, cg, nn.ActIdentity)
	s.jit.NewInput()
	if err := s.mv(out, l.ssmOut, s.dMixed); err != nil {
		return err
	}
	s.m.trace(li, "shortconv_out", out)
	return nil
}

// mamba1Attn runs one token of one sequence through layer li's Mamba-1
// mixer; h, out and the aliasing contract are linearAttn's.
//
// llama.cpp's build_mamba_layer:
//
//	xz      MUL_MAT(ssm_in) -> x, z           two projections here
//	x       SSM_CONV(conv state ++ x) + conv bias -> SILU
//	x_db    MUL_MAT(ssm_x, x) -> dt, B, C      three projections here
//	        RMS_NORM each, * its weight         Jamba, FalconMamba (ones)
//	dt      MUL_MAT(ssm_dt, dt) + ssm_dt.bias
//	y       SSM_SCAN(x, dt, A, B, C)            softplus inside the scan
//	y      += x * D
//	y       = SWIGLU_SPLIT(z, y)                y * silu(z)
//	out     MUL_MAT(ssm_out, y)
func (s *State) mamba1Attn(li int, l *layer, row int, h, out []float32) error {
	c := s.m.Cfg
	if err := s.mv(s.dMixed, l.wq, h); err != nil {
		return err
	}
	if err := s.mv(s.dZ, l.ssmGate, h); err != nil {
		return err
	}
	s.m.trace(li, "x", s.dMixed)
	s.conv1d(li, row, l.ssmConv1d)
	s.addBias(s.dMixed, l.ssmConvB)
	nn.Act32JIT(s.dMixed, nn.ActSiLU)
	s.jit.NewInput()
	s.m.trace(li, "conv_output_silu", s.dMixed)
	for _, p := range []struct {
		dst []float32
		w   tensor
	}{{s.dLowRank, l.ssmXDt}, {s.dSB, l.ssmXB}, {s.dSC, l.ssmXC}} {
		if err := s.mv(p.dst, p.w, s.dMixed); err != nil {
			return err
		}
	}
	if l.ssmDtNorm != nil {
		s.rmsnorm(s.dLowRank, s.dLowRank, l.ssmDtNorm, c.RMSEps)
		s.rmsnorm(s.dSB, s.dSB, l.ssmBNorm, c.RMSEps)
		s.rmsnorm(s.dSC, s.dSC, l.ssmCNorm, c.RMSEps)
	}
	s.jit.NewInput()
	if err := s.mv(s.dBRaw, l.ssmDtProj, s.dLowRank); err != nil {
		return err
	}
	// dt = softplus(dt + dt_bias) into dBeta; the gate's decay output, exp(0)
	// against zero rates, is not read -- the scan takes the decay per state
	// element from A itself.
	nn.SSDGate32JIT(s.dDecay, s.dBeta, s.dBRaw, l.ssmDtBias, s.dZeroA)
	s.m.trace(li, "dt", s.dBeta)
	s.mamba1Scan(li, row)
	s.m.trace(li, "ssm_y", s.dOut)
	nn.ActMul32JIT(s.dZ, s.dOut, nn.ActSiLU)
	s.jit.NewInput()
	if err := s.mv(out, l.ssmOut, s.dZ); err != nil {
		return err
	}
	s.m.trace(li, "ssm_out", out)
	return nil
}

// mamba1Scan runs the selective scan over every channel of layer li, fanned
// out over the pool; dOut receives y.
func (s *State) mamba1Scan(li, row int) {
	g, l := s.dg, &s.m.layers[li]
	j := &s.rg.mamba1
	j.st = s.rstate[li][row*g.deltaStateLen():][:g.deltaStateLen()]
	j.a, j.d = l.ssmA, l.ssmD
	if j.fn == nil {
		j.fn = s.mamba1Channels
	}
	s.jit.Parallel(g.vHeads, 32, j.fn)
	j.st, j.a, j.d = nil, nil, nil
}
