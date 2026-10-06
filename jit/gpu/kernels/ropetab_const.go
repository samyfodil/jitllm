package kernels

import "math"

// The rotary table's constant block and its layout: the {cos, sin} pair per
// rotary pair, formed by a kernel rather than math.Cos/math.Sin in Go.
//
// One copy of the layout for four tiers (AVX2, SSE, NEON and the device): the
// layout is the ABI between the emitters and nn. It lives here rather than in
// jit/cpu because jit/cpu already imports this package, so a device kernel
// importing jit/cpu would close a cycle; jit/cpu/ropetab_const.go re-exports
// every name.
//
// The precision lives in the constants. The angle is pos*g radians with pos up
// to 131072 -- thousands of radians, where an f32 ulp is 5e-4. So the position
// is decomposed into four 7-bit digits and each digit's contribution is folded
// mod 4 quarter-turns at setup, in float64:
//
//	pos = d0 + 128*d1 + 128^2*d2 + 128^3*d3          (d_i in [0,128))
//	v   = pos*g*2/pi = sum_i d_i * U_i   (mod 4),  U_i = fmod(g*128^i*2/pi, 4)
//
// and the kernel needs only frac(v) and floor(v) mod 4, which survive because
// the reduction mod 4 distributes over the sum.
//
// Each U_i is split so its digit product is exact. H_i is U_i rounded to a
// multiple of 2^-12, so it needs at most 15 significant bits; a digit needs 7;
// 7+15 = 22 < 24, so d_i*H_i is exact in float32. The four products sum to at
// most 2032 (8323072 units of 2^-12, under 2^24), so the sum and its mod-4
// fold are exact too. The only rounded quantity is the residual plane
// L_i = U_i - H_i, whose digit product is under 0.0156, giving ~1e-9
// quarter-turns of error (~1e-7 radians at every position; see
// engine/nn/ropetable.go).
//
// The per-model block is PLANE-MAJOR -- all of H0, then all of H1, ... -- so a
// kernel reads each plane as one contiguous vector and the four digit
// broadcasts stay in registers for the whole table:
//
//	word 0*S .. 1*S-1   H0      the 2^-12 head of U_0
//	word 1*S .. 2*S-1   H1
//	word 2*S .. 3*S-1   H2
//	word 3*S .. 4*S-1   H3
//	word 4*S .. 5*S-1   L0      U_0 - H_0, the residual
//	word 5*S .. 6*S-1   L1
//	word 6*S .. 7*S-1   L2
//	word 7*S .. 8*S-1   L3
//	word 8*S            mscale  (rope.scaling.attn_factor, or 1)
//
// where S = RopeTabStride(npairs).

// RopeTabDigits is how many base-128 digits of the position the kernel folds.
// Four cover positions below 2^28 (three would stop at 2^21) for two more
// instructions, and the exactness argument still holds.
const RopeTabDigits = 4

// RopeTabDigitBits is the width of one of those digits. Seven, because the
// digit's bits plus the head plane's must not exceed float32's 24.
const RopeTabDigitBits = 7

// RopeTabMaxPos is the first position this decomposition does not cover.
const RopeTabMaxPos = 1 << (RopeTabDigits * RopeTabDigitBits)

// RopeTabGrid is the head plane's quantization grid, as its reciprocal's
// denominator: the planes hold multiples of 1/RopeTabGrid. It is the largest
// grid for which the digit product and the four-term sum stay exact in
// float32. Exported because the device kernel folds mod four in integers and
// needs the (power-of-two, so exact) scale.
const RopeTabGrid = 1 << 12

const ropeTabGrid = RopeTabGrid

// ropeTabLanes is the pad every per-model block is sized to: eight whatever the
// tier, since the AVX2 kernel reads eight pairs at a time and the SSE and NEON
// ones four. A device thread reads one pair and needs no pad.
const ropeTabLanes = 8

// RopeTabStride is the word stride between two planes of a per-model block.
func RopeTabStride(npairs int) int {
	return (npairs + ropeTabLanes - 1) &^ (ropeTabLanes - 1)
}

// RopeTabBlock is how many float32 words a per-model block for npairs rotary
// pairs occupies: eight planes and the scale.
func RopeTabBlock(npairs int) int { return 8*RopeTabStride(npairs) + 1 }

// RopeTabPlaneOff is the word offset of plane p (0..3 the heads H, 4..7 the
// residuals L) inside a per-model block.
func RopeTabPlaneOff(p, npairs int) int { return p * RopeTabStride(npairs) }

// RopeTabScaleOff is the word offset of mscale inside a per-model block.
func RopeTabScaleOff(npairs int) int { return 8 * RopeTabStride(npairs) }

// RopeTabHeadSplit splits one folded quarter-turn constant into the head that
// multiplies a digit exactly and the residual that carries the rest.
//
// u is taken in float64 and must already be folded into [0,4]; the caller
// (nn.Rope) does the fold because it owns the frequency.
func RopeTabHeadSplit(u float64) (h, l float32) {
	h = float32(math.Round(u*ropeTabGrid) / ropeTabGrid)
	l = float32(u - float64(h))
	return h, l
}

// RopeTabConsts is the shared constant block every rotary-table kernel reads --
// through Args.Scr on the host tiers and as a device buffer on the other.
//
// The magic is a rounding, not a mask: x - ((x+M)-M) with M = 2^25 is x minus
// the nearest multiple of four, because at 2^25 an f32 ulp is exactly 4. The
// remainder is in [-2,2], congruent to x mod 4, and exact for every x the
// digit products can produce (at most 2032).
//
// pi/2 is split in two so one rounded f32 pi/2 does not put a 5e-8 relative
// error on every angle (the same remedy exp.go uses for ln2).
//
// The polynomials are degree 7 in sin and 8 in cos: on |r| <= pi/4 these
// minimax fits are accurate to about 1e-8, an order under the reduction's
// ~1e-7. One degree less would make sin's truncation (4.5e-7) the largest
// error term.
//
// The last two words are the device's alone. A SPIR-V driver (NVIDIA's
// Vulkan path) reassociates (a+b)-b to a, so the magic returns zero; the
// device therefore folds mod four in integers: s is a sum of exact multiples
// of 2^-12, so s*2^12 is an exact integer under 2^24, masking it with
// 4*2^12-1 is the remainder, and scaling back is exact. RopeTabMod4MaskWord is
// that mask (and the device's violation surface for the fold).
//
// The half: the IR's only float-to-integer op, ir.OpCvtI32, truncates, where
// the host rounds to nearest. The integer fold leaves the remainder in [0,4),
// so s+c is never below -1/16 and truncating s+c+0.5 is the floor. The two
// roundings differ only at an exact tie, where the quadrant swap and sign flip
// give bit-identical results because the sine fit is odd and the cosine even.
// A remainder differing from the host's by a multiple of four changes n by the
// same multiple, leaving f = rem-n and bits 0 and 1 of n unchanged.
func RopeTabConsts() []float32 {
	c := make([]float32, RopeTabConstWords)
	c[RopeTabMagicWord] = float32(1 << 25) // the round-to-a-multiple-of-four magic
	// pi/2, split so the head is exact in f32 and the tail carries the rest.
	hi := float32(math.Pi / 2)
	c[RopeTabPio2HiWord] = hi
	c[RopeTabPio2LoWord] = float32(math.Pi/2 - float64(hi))
	// sin(r) = r + r^3*(c3 + r^2*(c5 + r^2*c7)) on |r| <= pi/4.
	c[RopeTabSinC7Word] = -1.9515295891e-4
	c[RopeTabSinC5Word] = 8.3321608736e-3
	c[RopeTabSinC3Word] = -1.6666654611e-1
	// cos(r) = 1 + r^2*(-1/2 + r^2*(c4 + r^2*(c6 + r^2*c8))) on |r| <= pi/4.
	c[RopeTabCosC8Word] = 2.443315711809948e-5
	c[RopeTabCosC6Word] = -1.388731625493765e-3
	c[RopeTabCosC4Word] = 4.166664568298827e-2
	c[RopeTabNegHalfWord] = -0.5
	c[RopeTabOneWord] = 1
	// the digit mask, as bits
	c[RopeTabDigitWord] = math.Float32frombits(uint32(1<<RopeTabDigitBits - 1))
	// Two replicated vectors rather than scalars, because the host kernels apply
	// them with a whole-vector memory operand (VPAND/VPADDD have no broadcast
	// form on that tier). A device thread reads word zero of each.
	for i := 0; i < 8; i++ {
		c[RopeTabSignVecWord+i] = math.Float32frombits(0x80000000) // the sign bit
		c[RopeTabOneVecWord+i] = math.Float32frombits(1)           // the integer one
		c[RopeTabTwoVecWord+i] = math.Float32frombits(2)           // the integer two
	}
	// The device's two: the mod-four mask in the 2^-12 grid, as bits, and the
	// half that turns a truncating convert into a rounding one.
	c[RopeTabMod4MaskWord] = math.Float32frombits(4*ropeTabGrid - 1)
	c[RopeTabHalfWord] = 0.5
	return c
}

// The word indices into that block, named so an emitter cannot transpose two.
//
// Some are also the violation surface: several structural steps read a word
// only that step reads, so zeroing it removes exactly that step on every tier
// (engine/nn/ropetabviolation_test.go for the host, backend's device gate for the
// device). The sign step needs two words because amd64 and the device mask
// bit 1 of the quadrant into a sign bit and XOR, while NEON selects between a
// value and its FNEG under a mask built by comparing n&2 against two.
const (
	RopeTabMagicWord   = 0
	RopeTabPio2HiWord  = 1
	RopeTabPio2LoWord  = 2
	RopeTabSinC7Word   = 3
	RopeTabSinC5Word   = 4
	RopeTabSinC3Word   = 5
	RopeTabCosC8Word   = 6
	RopeTabCosC6Word   = 7
	RopeTabCosC4Word   = 8
	RopeTabNegHalfWord = 9
	RopeTabOneWord     = 10
	RopeTabDigitWord   = 11
	RopeTabSignVecWord = 12
	RopeTabOneVecWord  = 20
	RopeTabTwoVecWord  = 28
	// The device's own two, standing in for instructions the IR does not have
	// (see RopeTabConsts).
	RopeTabMod4MaskWord = 36
	RopeTabHalfWord     = 37
	// RopeTabConstWords is how many float32 the block holds, so an upload is
	// sized from the layout rather than from a literal.
	RopeTabConstWords = RopeTabHalfWord + 1
)
