//go:build amd64

package cpu

import (
	"fmt"

	"github.com/jitllm/jitllm/format/quant"

	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// EmitPackedMatVecFused runs every word of a sub-block in one pass over the
// rows, so the integer accumulator never leaves a register. In the wide
// kernel's separate pass per word, two thirds of the memory operations were
// the accumulator's load and store (see AGENTS.md's top-down counters and
// docs/engineering-history/cpu-kernels.md).
//
// Holding every word's activation broadcast at once costs 2*words registers.
// They are found by precomputing the super-block scale and its minimum as f32
// into scratch once per super-block: the f16 gather costs four instructions
// and two shuffle-mask registers, a precomputed f32 one load and none.
//
//	per 16 rows of a Q4_K sub-block   wide   fused
//	  instructions                     120      94
//	  memory operations                 34      18
//
// Args:
//
//	Out      [rows] f32, ACCUMULATED into; the caller zeroes it
//	W/PD/PSC the three spans, at this call's first row
//	Rows     how many PackedFusedGroup-row groups this call serves
//	RowStr   the layout's row stride in bytes
//	Scratch  [rows] f32, the precomputed super-block scale
//	Q32      [rows] f32, its minimum -- only for a format that has one
//
// EmitPackedMatVecFusedWin has no amd64 form: VPDPBUSD sums unscaled bytes,
// so an integer fold would need a VPMULLD per sub-block (packedgemm.go).
func EmitPackedMatVecFusedWin(t quant.Type, _ int) ([]byte, error) {
	return nil, fmt.Errorf("jit: EmitPackedMatVecFusedWin: %s: no amd64 integer form", t)
}

func EmitPackedMatVecFused(t quant.Type, dk DotKind) ([]byte, error) {
	return EmitPackedMatVecFusedAhead(t, dk, 0)
}

// FusedAheadWords selects EmitPackedMatVecFusedAhead's word-ahead prefetch:
// eight payload words ahead in the same row group, rather than a distance in
// row groups.
const FusedAheadWords = 64

// EmitPackedMatVecFusedAhead is the fused kernel with a PREFETCHT0 per payload
// word, `ahead` row groups in front of the one being read; 0 emits none. The
// right distance depends on the host's prefetcher, so nn measures it on real
// tokens (engine/nn/tune.go's prefetch duel).
func EmitPackedMatVecFusedAhead(t quant.Type, dk DotKind, ahead int) ([]byte, error) {
	// FusedAheadWords is a different prefetch, not a longer one. A distance in
	// row groups runs off the end of the call's row range at the last groups of
	// each sub-block, so the next sub-block's first lines are demand misses.
	// Eight words ahead in the same row group is exactly the set of lines this
	// call reads next.
	wordAhead := ahead == FusedAheadWords
	q, ok := kernels.QuantOf(t)
	if !ok {
		return nil, fmt.Errorf("jit: EmitPackedMatVecFused: %s has no device layout", t)
	}
	// The saturation refusal, before anything is emitted; see PreVNNIPacked.
	if dk == DotVEX {
		if err := PreVNNIPacked(t); err != nil {
			return nil, err
		}
	}
	sub, bits, biasK, biasArray := kernels.Layout(q)
	perSuper, scOff := kernels.ScaleLayout(q)
	signed := kernels.SignedPayload(q)
	minD := kernels.MinInD(q)
	hi := kernels.HiPlane(q)
	codes, err := hasCodes(q)
	if err != nil {
		return nil, err
	}
	// The payload is pw primary words then hw secondary ones. The hi code
	// reaches the same accumulator as the primary because its mask is
	// pre-shifted by 4 (see PackedScratch).
	pw, hw := sub*bits/32, sub*hi/32
	words := pw + hw
	halfSub := sub / 2
	const vec = 32
	group := PackedFusedGroupOf(t)
	grp := group / 8

	// The register budget is checked: a spilling tile has measured as a large
	// loss, and the activation set grows with the payload width.
	nact := pw
	if bits == 4 {
		nact = 2 * pw
	}
	need := 3 + grp + 3 + nact // mask, da, bscale + accumulators + pay/tmp/t2
	if scOff != 0 {
		need++
	}
	if hi != 0 {
		need++ // the pre-shifted secondary mask
	}
	if codes {
		need++ // the code table
	}
	if need > 16 {
		return nil, fmt.Errorf("jit: EmitPackedMatVecFused: %s needs %d vector registers, have 16",
			t, need)
	}

	var a Buf
	next := Y0
	take := func() Reg { r := next; next++; return r }
	mask, da, bscale := take(), take(), take()
	iac := make([]Reg, grp)
	for i := range iac {
		iac[i] = take()
	}
	act := make([]Reg, nact)
	for i := range act {
		act[i] = take()
	}
	pay, tmp, t2 := take(), take(), take()
	// A straddling SC field reads the next word with no register to hold its
	// address (R14 is g): the cursor steps one plane stride and back around the
	// load (scstream.go).
	stream := kernels.ScStream(q)
	nextSC := func(off int32) func(Reg) {
		return func(r Reg) {
			a.ADDQ(R9, R13)
			a.VMOVDQULoad(r, Idx(R9, R15, 1, off))
			a.SUBQ(R9, R13)
		}
	}
	// The i16 ones VPMADDWD needs costs no register; the budget above lands at
	// exactly 16 for Q4_0, Q8_0, Q4_K and Q5_K.
	//   bits == 8: the pre-VNNI path loads neither the nibble mask nor the 0x80
	//     sign flip (VPSIGNB replaces the centring), so `mask` holds the ones
	//     vector for the whole kernel.
	//   bits == 4: `mask` is live, but t2 is an epilogue temporary untouched by
	//     the payload loop, so ones is rebuilt into it once per row group.
	dot := dotEmitter{kind: dk, flip: mask}
	if dk == DotVEX {
		dot.ones = mask
		if bits == 4 {
			dot.ones = t2
		}
	}
	scOffV := Y0
	if scOff != 0 {
		scOffV = take()
	}
	hiMask := Y0
	if hi != 0 {
		hiMask = take()
	}
	lut := Y0
	if codes {
		lut = take()
	}

	// R10 is AHalf or the minimum scratch, never both: a format with a minimum
	// array has 32-element sub-blocks and no per-16 sums, and a 16-wide format is
	// the reverse. That is what closes the GPR budget at thirteen with R14 unused.
	a.MOVLoad(RCX, At(RDI, 0))  // Out
	a.MOVLoad(R11, At(RDI, 16)) // A
	a.MOVLoad(R12, At(RDI, 24)) // AScale
	a.MOVLoad(R13, At(RDI, 48)) // RowStr
	a.MOVLoad(RBX, At(RDI, 56)) // Scr, for the constants below
	if biasArray {
		a.MOVLoad(R10, At(RDI, 112)) // Q32 -> the minimum scratch
	} else {
		a.MOVLoad(R10, At(RDI, 64)) // AHalf
	}
	switch {
	case bits == 4:
		a.VMOVDQULoad(mask, At(RBX, 0))
	case dk == DotVEX:
		// The ones vector, built rather than loaded; see ones16.
		ones16(&a, mask)
	case signed:
		a.VMOVDQULoad(mask, At(RBX, 192))
	}
	if scOff != 0 {
		a.VMOVDQULoad(scOffV, At(RBX, 224+int32(scOffSlot(scOff))))
	}
	if hi != 0 {
		a.VMOVDQULoad(hiMask, At(RBX, hiMaskSlot(hi)))
	}
	if codes {
		a.VMOVDQULoad(lut, At(RBX, codesSlot))
	}
	// The two shuffle masks are needed only by the precompute pass, which runs
	// while the activation registers are still free.
	shufLo, shufHi := act[0], act[1]

	rowLoop := func(body func()) {
		a.MOVLoad(RBX, At(RDI, 32))
		a.SUBQ(R15, R15)
		l := a.Label()
		a.Bind(l)
		body()
		a.ADDimm(R15, int32(group*4))
		a.DEC(RBX)
		a.JNZ(l)
	}

	a.MOVLoad(RDX, At(RDI, 8)) // QS
	// R9 is the SC cursor, and a narrow format has no SC plane, so there it
	// carries the d plane's own row offset; no other register is free.
	narrow := NarrowD(t)
	e8 := E8M0D(t)
	drow := int32(DRowBytes(t))
	if !narrow {
		a.MOVLoad(R9, At(RDI, 152)) // SC
	}
	a.MOVLoad(RAX, At(RDI, 40)) // K, in super-blocks
	a.MOVLoad(R8, At(RDI, 144)) // D, held for the whole kernel

	outer := a.Label()
	a.Bind(outer)

	// --- precompute this super-block's scales, once ------------------------
	// RBX is reloaded because rowLoop owns it as the row counter: after the
	// first loop it is zero. R8 is not reloaded: it is the d cursor for the
	// whole kernel and advances one super-block per outer iteration.
	a.MOVLoad(RBX, At(RDI, 56))
	a.MOVLoad(RSI, At(RDI, 72)) // Scratch
	a.VMOVDQULoad(shufLo, At(RBX, 64))
	a.VMOVDQULoad(shufHi, At(RBX, 96))
	if narrow {
		a.SUBQ(R9, R9)
	}
	rowLoop(func() {
		for g := 0; g < grp; g++ {
			off := int32(g * vec)
			if e8 {
				// MXFP4's super-scale is a biased E8M0 exponent, so the byte
				// shifted left 23 is the f32, exactly (exp.go's 2^x trick).
				a.VPMOVZXBD(tmp, Idx(R8, R9, 1, off/4))
				a.VPSLLD(tmp, tmp, 23)
			} else if narrow {
				// Eight f16 are sixteen bytes and VCVTPH2PS widens them in one
				// instruction, where the wide plane needs a load, a VPSHUFB and
				// a VPERMQ to gather eight halves out of eight words.
				a.VCVTPH2PS(tmp, Idx(R8, R9, 1, off/2))
			} else {
				a.VMOVDQULoad(pay, Idx(R8, R15, 1, off))
				a.VPSHUFB(tmp, pay, shufLo)
				a.VPERMQ(tmp, tmp, 0xD8)
				a.VCVTPH2PSReg(tmp, tmp)
			}
			a.VMOVDQUStore(Idx(RSI, R15, 1, off), tmp)
			if biasArray {
				a.VPSHUFB(t2, pay, shufHi)
				a.VPERMQ(t2, t2, 0xD8)
				a.VCVTPH2PSReg(t2, t2)
				a.VMOVDQUStore(Idx(R10, R15, 1, off), t2)
			}
		}
		if narrow {
			a.ADDimm(R9, int32(group)*drow)
		}
	})
	if narrow {
		a.ADDQmem(R8, At(RDI, 160)) // half the payload's stride; see DSuperBytes
	} else {
		a.ADDQ(R8, R13) // D advances one super-block
	}

	// pl is the payload word at off in the current row group, and prefetch
	// issues the software prefetch this kernel was emitted with; pfd is set
	// per row group below.
	var pfd int32
	pl := func(off int32) Mem {
		if wordAhead {
			return At(RSI, off)
		}
		return Idx(RSI, R15, 1, off)
	}
	prefetch := func() {
		switch {
		case wordAhead:
			a.PREFETCHT0(Idx(RSI, R13, 8, 0))
		case pfd > 0:
			a.PREFETCHT0(Idx(RSI, R15, 1, pfd))
		}
	}

	// --- the sub-blocks ----------------------------------------------------
	for sn := 0; sn < perSuper; sn++ {
		pairOff := int32(sub * sn / Q8Block * 8)
		sh := byte((sn % perWordOf(biasArray)) * strideOf(biasArray))
		for w := 0; w < pw; w++ {
			a.VPBROADCASTD(act[w], At(R11, int32(sub*sn+4*w)))
			if bits == 4 {
				a.VPBROADCASTD(act[pw+w], At(R11, int32(sub*sn+halfSub+4*w)))
			}
		}
		a.VBROADCASTSS(da, At(R12, pairOff))
		a.MOVLoad(RBX, At(RDI, 56)) // rowLoop clobbered it
		switch {
		case biasArray:
			a.VBROADCASTSS(bscale, At(R12, pairOff+4))
			a.VMOVDQULoad(tmp, At(RBX, 128)) // -8
			a.VMULPS(bscale, bscale, tmp)
			a.VMULPS(bscale, bscale, da)
		case sub == Q8Block/2:
			a.VBROADCASTSS(bscale, At(R10, int32(4*sn)))
			a.VMOVDQULoad(tmp, At(RBX, 288)) // +4
			a.VMULPS(bscale, bscale, tmp)
		case biasK != 0 || signed:
			a.VBROADCASTSS(bscale, At(R12, pairOff+4))
			a.VMOVDQULoad(tmp, At(RBX, 160)) // +8
			a.VMULPS(bscale, bscale, tmp)
		}
		rowLoop(func() {
			// RSI is the payload cursor and then the scratch base: the cursor
			// dies when the last word has issued, and the scratch is only read
			// in the epilogue.
			a.MOVQ(RSI, RDX) // reset every row group
			if wordAhead {
				a.ADDQ(RSI, R15) // the cursor carries the row offset; see below
			}
			for g := 0; g < grp; g++ {
				a.VPXOR(iac[g], iac[g], iac[g])
			}
			if dk == DotVEX && bits == 4 {
				ones16(&a, t2) // dead until the epilogue; see the note above
			}
			// A word's lines are contiguous across row groups, so the line
			// `ahead` groups on is a constant displacement. The hardware
			// prefetcher sees each word as a short run and re-trains every
			// sub-block, which matters on high-latency hosts (two-socket
			// Broadwell); see docs/engineering-history/cpu-kernels.md.
			pfd = 0
			if !wordAhead {
				pfd = int32(ahead * grp * vec)
			}
			for w := 0; w < pw; w++ {
				prefetch()
				for g := 0; g < grp; g++ {
					off := int32(g * vec)
					a.VMOVDQULoad(pay, pl(off))
					if bits == 4 {
						a.VPAND(tmp, pay, mask)
						if codes {
							a.VPSHUFB(tmp, lut, tmp)
						}
						dot.unsigned(&a, iac[g], tmp, act[w])
						a.VPSRLD(tmp, pay, 4)
						a.VPAND(tmp, tmp, mask)
						if codes {
							a.VPSHUFB(tmp, lut, tmp)
						}
						dot.unsigned(&a, iac[g], tmp, act[pw+w])
					} else if signed {
						dot.signed(&a, iac[g], pay, act[w], tmp)
					} else {
						dot.unsigned(&a, iac[g], pay, act[w])
					}
				}
				a.ADDQ(RSI, R13)
			}
			// The secondary plane, after the cursor has walked the primary
			// one, so RSI already points at hi word 0. packSub's pairing makes
			// each extraction a single shift: byte j of a hi word carries every
			// element with l%4 == j, ordered by l/4.
			lanes := 0
			if hi != 0 {
				lanes = 8 / hi
			}
			for hwi := 0; hwi < hw; hwi++ {
				prefetch()
				for g := 0; g < grp; g++ {
					off := int32(g * vec)
					a.VMOVDQULoad(pay, pl(off))
					for half := 0; half < 2; half++ {
						for w := 0; w < pw; w++ {
							gi := w + half*pw
							if gi/lanes != hwi {
								continue
							}
							// The mask already carries the <<4, so the shift is
							// relative to bit 4 and can go either way.
							switch sh := hi * (gi % lanes); {
							case sh < 4:
								a.VPSLLD(tmp, pay, byte(4-sh))
								a.VPAND(tmp, tmp, hiMask)
							case sh == 4:
								a.VPAND(tmp, pay, hiMask)
							default:
								a.VPSRLD(tmp, pay, byte(sh-4))
								a.VPAND(tmp, tmp, hiMask)
							}
							dot.unsigned(&a, iac[g], tmp, act[half*pw+w])
						}
					}
				}
				a.ADDQ(RSI, R13)
			}
			// The pre-VNNI signed payload accumulated dot(w, a); put it back on
			// VPDPBUSD's dot(w+128, a) so the epilogue below -- and its rounding
			// -- is byte for byte the kernel a VNNI host runs.
			if signed {
				dot.signedFix(&a, iac, At(R12, pairOff+4), t2)
			}
			a.MOVLoad(RSI, At(RDI, 72)) // the scratch base, for the epilogue
			for g := 0; g < grp; g++ {
				off := int32(g * vec)
				if stream {
					// Container v27's stream: the scale into t2 before d takes
					// tmp, which a straddling field borrows.
					a.VMOVDQULoad(pay, Idx(R9, R15, 1, off))
					scField(avxSC(&a), t2, pay, tmp, sn, false, nextSC(off))
					if scOff != 0 {
						a.VPSUBD(t2, t2, scOffV)
					}
					a.VMOVDQULoad(tmp, Idx(RSI, R15, 1, off)) // d, precomputed f32
					a.VCVTDQ2PS(t2, t2)
					a.VMULPS(tmp, tmp, t2)
				} else if perSuper > 1 {
					a.VMOVDQULoad(tmp, Idx(RSI, R15, 1, off)) // d, precomputed f32
					a.VMOVDQULoad(pay, Idx(R9, R15, 1, off))  // the scale word
					if 24-sh > 0 {
						a.VPSLLD(t2, pay, 24-sh)
					} else {
						a.VMOVAPSReg(t2, pay)
					}
					a.VPSRLD(t2, t2, 24)
					if scOff != 0 {
						a.VPSUBD(t2, t2, scOffV)
					}
					a.VCVTDQ2PS(t2, t2)
					a.VMULPS(tmp, tmp, t2)
				} else {
					a.VMOVDQULoad(tmp, Idx(RSI, R15, 1, off)) // d, precomputed f32
				}
				a.VMULPS(tmp, tmp, da) // scale * d_a
				a.VCVTDQ2PS(iac[g], iac[g])
				a.VMOVDQULoad(t2, Idx(RCX, R15, 1, off))
				a.VFMADD231PS(t2, iac[g], tmp)
				if stream {
					// The minimum into pay first: a straddling one borrows tmp.
					scField(avxSC(&a), pay, pay, tmp, sn, true, nextSC(off))
					a.VCVTDQ2PS(pay, pay)
					a.VMOVDQULoad(tmp, Idx(R10, R15, 1, off)) // dmin, precomputed
					a.VMULPS(tmp, tmp, pay)
					a.VMULPS(tmp, tmp, bscale)
					a.VSUBPS(t2, t2, tmp)
				} else if biasArray {
					a.VMOVDQULoad(tmp, Idx(R10, R15, 1, off)) // dmin, precomputed
					if !minD {                                // Q5_1's m is one: no sc word to read it from
						a.VPSLLD(pay, pay, 16-sh)
						a.VPSRLD(pay, pay, 24)
						a.VCVTDQ2PS(pay, pay)
						a.VMULPS(tmp, tmp, pay)
					}
					a.VMULPS(tmp, tmp, bscale)
					a.VSUBPS(t2, t2, tmp)
				} else {
					a.VMULPS(tmp, tmp, bscale)
					a.VADDPS(t2, t2, tmp)
				}
				a.VMOVDQUStore(Idx(RCX, R15, 1, off), t2)
			}
		})
		for w := 0; w < words; w++ {
			a.ADDQ(RDX, R13)
		}
		if stream && scStep(sn) || !stream && perSuper > 1 && sn%perWordOf(biasArray) == perWordOf(biasArray)-1 {
			a.ADDQ(R9, R13)
		}
	}
	a.ADDimm(R11, int32(perSuper*sub))
	a.ADDimm(R12, int32(perSuper*sub/Q8Block*8))
	if sub == Q8Block/2 {
		a.ADDimm(R10, int32(perSuper*4))
	}
	if perSuper == 1 {
		a.ADDQ(R9, R13)
	}
	a.DEC(RAX)
	a.JNZ(outer)

	a.VZEROUPPER()
	a.RET()
	code := a.Bytes()
	if len(code) > PackedWideBudget {
		return nil, fmt.Errorf("jit: EmitPackedMatVecFused: %s is %d bytes, over the %d budget",
			t, len(code), PackedWideBudget)
	}
	return code, nil
}

func perWordOf(biasArray bool) int {
	if biasArray {
		return 2
	}
	return 4
}

func strideOf(biasArray bool) int {
	if biasArray {
		return 16
	}
	return 8
}
