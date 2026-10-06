package cpu

// DeepSeek V4's two small kernels, the hyper-connection mixer and the
// compressor's pool. This file holds what every tier shares: the constant
// block and the ABI; hc.go (AVX2), sse_hc.go and hc_a64.go are the bodies.
//
// The hyper-connection mixer (EmitHCMix) turns one row's (2+H)*H mixes, H = 4
// (HCStreams), into the collapse weights, the placements and the stream mixer:
//
//	pre[i]  = sigma(m[i]*s[i] + b[i]) + eps                  i < 4
//	post[i] = 2*sigma(m[4+i]*s[4+i] + b[4+i])
//	comb    = softmax_rows(m[8:]*s[8:] + b[8:]) + eps        4 x 4, row-major
//	comb    = comb / (column sums + eps), then iters-1 times:
//	          comb / (row sums + eps), comb / (column sums + eps)
//
// iters 0 stops after the softmax (a gate's violation). A head kernel computes
// pre alone, from eight-float buffers whose last four are padding.
//
//	Out     the 24 outputs (8 for a head)
//	Q32     the mixes
//	AScale  each mix's scale (s0 on pre, s1 on post, s2 on comb)
//	Q2      each mix's base
//	Scr     HCMixConsts(eps)
//
// Every tier keeps the 4 x 4 mixer one row per 128-bit lane (AVX2: two rows a
// YMM; SSE and NEON: one row a register), so a row's sum and maximum are
// within-lane shuffles and a column's are adds of whole registers.
//
// The pool (EmitColPool) is the compressor's softmax over positions, per
// channel:
//
//	out[c] = sum_s e[s,c] * kv[s*W+c] / sum_s e[s,c],  e = exp(gate[s*W+c] - max_s gate[s*W+c])
//
//	Out     out, W wide
//	Q32     kv, S rows of W
//	Q2      gate, S rows of W
//	Scr     ActConsts()
//	K, Rows the whole vectors and tail elements of W (ElemLanes)
//	Cols    S, at least one
//	RowStr  4*W, the slot stride in bytes

// HCStreams is the stream count the mixer is built for: DeepSeek V4's.
const HCStreams = 4

// HCMixConsts is the mixer's constant block: exp's block, padded to 64 bytes,
// then pre/post's multiplier (1 x4, 2 x4) at 64, its offset (eps x4, 0 x4) at
// 96 and eps at 128.
func HCMixConsts(eps float32) []float32 {
	c := make([]float32, 16, 33)
	copy(c, expConsts())
	c = append(c, 1, 1, 1, 1, 2, 2, 2, 2)
	c = append(c, eps, eps, eps, eps, 0, 0, 0, 0)
	return append(c, eps)
}

const (
	hcMulOff = 64
	hcAddOff = 96
	hcEpsOff = 128
)
