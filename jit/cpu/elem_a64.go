//go:build arm64

package cpu

// The arm64 twins of the elementwise kernels: RMSNorm, exp, softmax and the
// activations. With 32 vector registers, exp's constants sit in v16..v26 for
// the whole body, and a horizontal maximum is one FMAXV where amd64 needs a
// shuffle ladder.

// a64ExpConsts is the constant block, byte-for-byte the same layout the amd64
// side uses so one caller can fill one buffer for both.
func a64ExpConsts() []float32 { return expConstsShared() }

// loadA64ExpConsts puts exp's constants in v16..v24, from the block at x5.
func loadA64ExpConsts(a *A64) {
	for i, v := range []struct {
		reg VReg
		off int32
	}{
		{16, 0}, {17, 4}, {18, 8}, {19, 12}, {20, 16},
		{21, 20}, {22, 24}, {23, 28}, {24, 40},
	} {
		_ = i
		a.LDRs(v.reg, X5, v.off)
		a.DUPs4(v.reg, v.reg)
	}
	// v25 is the exponent bias as an integer, v26 the low clamp.
	a.LDRs(25, X5, 36)
	a.DUPs4(25, 25)
	a.LDRs(26, X5, 32)
	a.DUPs4(26, 26)
}

// emitA64Exp computes exp(d) into d, clobbering t0 and t1.
//
// Same algorithm as the amd64 emitExpPS, including both clamps: the upper one
// exists because GELU feeds this 2z, which can exceed the f32 exponent field.
func emitA64Exp(a *A64, d, t0, t1 VReg) {
	a.FMAX4s(d, d, 26) // clamp low
	a.FMIN4s(d, d, 24) // clamp high
	a.FMUL4s(t0, d, 16)
	a.FCVTNS4s(t1, t0) // n, integer, ROUND TO NEAREST
	a.SCVTF4s(t0, t1)  // n, float
	a.FMLA4s(d, t0, 17)
	a.FMLA4s(d, t0, 18)
	// Horner, degree 5. FMLA is dst += n*m, so each step is a multiply into a
	// scratch and an add (arm64 has no 213-shaped FMA).
	a.MOVvec(t0, 19)
	a.FMUL4s(t0, t0, d)
	a.FADD4s(t0, t0, 20)
	a.FMUL4s(t0, t0, d)
	a.FADD4s(t0, t0, 21)
	a.FMUL4s(t0, t0, d)
	a.FADD4s(t0, t0, 22)
	a.FMUL4s(t0, t0, d)
	a.FADD4s(t0, t0, 23)
	a.FMUL4s(t0, t0, d)
	a.FADD4s(t0, t0, 23)
	// 2^n in the exponent field.
	a.ADD4s(t1, t1, 25)
	a.SHL4s(t1, t1, 23)
	a.FMUL4s(d, t0, t1)
}

// EmitA64RMSNorm is the NEON twin of EmitRMSNorm: y = x/sqrt(mean(x^2)+eps) * w.
//
// x0 args, x1 out, x2 x, x3 w, x5 consts, x4 counter.
func EmitA64RMSNorm(n int) []byte {
	var a A64
	a.LDRx(X1, X0, 0)  // Out
	a.LDRx(X2, X0, 16) // A -> x
	a.LDRx(X3, X0, 24) // AScale -> w
	a.LDRx(X5, X0, 56) // Scr -> [1/n, eps, 1]

	// pass 1: sum of squares, four chains.
	a.MOVIzero(0)
	a.MOVIzero(1)
	a.MOVIzero(2)
	a.MOVIzero(3)
	a.MOVreg(X6, X2)
	// Guarded, because CBNZ makes this a do-while: a zero trip count would run
	// the body once and then loop 2^64 times. amd64's twin has the same guard.
	if blocks := n / 16; blocks > 0 {
		a.MOVimm(X4, int64(blocks))
		lp := a.Label()
		a.Bind(lp)
		for i := 0; i < 4; i++ {
			a.LDRq(VReg(4+i), X6, int32(16*i))
			a.FMLA4s(VReg(i), VReg(4+i), VReg(4+i))
		}
		a.ADDimm(X6, X6, 64)
		a.SUBimm(X4, X4, 1)
		a.CBNZ(X4, lp)
	}
	for off := (n / 16) * 16; off+4 <= n; off += 4 {
		a.LDRq(4, X6, 0)
		a.FMLA4s(0, 4, 4)
		a.ADDimm(X6, X6, 16)
	}
	// The last n%4 one at a time; LDR s zeroes lanes 1..3, which add nothing.
	for i := 0; i < n%4; i++ {
		a.LDRs(4, X6, int32(4*i))
		a.FMLA4s(0, 4, 4)
	}
	a.FADD4s(0, 0, 1)
	a.FADD4s(2, 2, 3)
	a.FADD4s(0, 0, 2)
	a.FADDP4s(0, 0, 0)
	a.FADDPs(0, 0)

	// scale = 1/sqrt(ss/n + eps)
	a.LDRs(5, X5, 0) // 1/n
	a.LDRs(6, X5, 4) // eps
	a.FMULs(0, 0, 5)
	a.FADD4s(0, 0, 6)
	a.FSQRT4s(0, 0)
	a.LDRs(7, X5, 8) // 1.0
	a.FDIV4s(0, 7, 0)
	a.DUPs4(0, 0)

	// pass 2: y = x * scale * w
	a.MOVreg(X6, X2)
	a.MOVreg(X7, X1)
	if blocks := n / 16; blocks > 0 {
		a.MOVimm(X4, int64(blocks))
		lp2 := a.Label()
		a.Bind(lp2)
		for i := 0; i < 4; i++ {
			a.LDRq(VReg(4+i), X6, int32(16*i))
			a.LDRq(VReg(8+i), X3, int32(16*i))
			a.FMUL4s(VReg(4+i), VReg(4+i), 0)
			a.FMUL4s(VReg(4+i), VReg(4+i), VReg(8+i))
			a.STRq(VReg(4+i), X7, int32(16*i))
		}
		a.ADDimm(X6, X6, 64)
		a.ADDimm(X3, X3, 64)
		a.ADDimm(X7, X7, 64)
		a.SUBimm(X4, X4, 1)
		a.CBNZ(X4, lp2)
	}
	for off := (n / 16) * 16; off+4 <= n; off += 4 {
		a.LDRq(4, X6, 0)
		a.LDRq(8, X3, 0)
		a.FMUL4s(4, 4, 0)
		a.FMUL4s(4, 4, 8)
		a.STRq(4, X7, 0)
		a.ADDimm(X6, X6, 16)
		a.ADDimm(X3, X3, 16)
		a.ADDimm(X7, X7, 16)
	}
	for i := 0; i < n%4; i++ {
		d := int32(4 * i)
		a.LDRs(4, X6, d)
		a.LDRs(8, X3, d)
		a.FMUL4s(4, 4, 0)
		a.FMUL4s(4, 4, 8)
		a.STRs(4, X7, d)
	}
	a.RET()
	return a.Bytes()
}

// EmitA64Softmax is the NEON twin of EmitSoftmax: Args.K whole vectors of four
// and Args.Rows single elements after them. The tails are EmitSoftmax's: the
// maximum broadcasts its element, the exp pass keeps lane 0 alone before the
// sum.
func EmitA64Softmax() []byte {
	var a A64
	a.LDRx(X1, X0, 0)   // Out -> the row, in place
	a.LDRx(X5, X0, 56)  // Scr -> consts
	a.LDRx(X9, X0, 40)  // K -> whole vectors
	a.LDRx(X10, X0, 32) // Rows -> tail elements
	loadA64ExpConsts(&a)
	pass := func(vec, one func()) {
		a.MOVreg(X6, X1)
		a.MOVreg(X4, X9)
		a.MOVreg(X7, X10)
		a64ElemLoops(&a, X4, X7, []XReg{X6}, vec, one)
	}

	// pass 1: maximum, seeded from the ROW -- see EmitSoftmax for why not from
	// the clamp constant.
	a.LDRs(3, X1, 0)
	a.DUPs4(3, 3)
	pass(func() {
		a.LDRq(0, X6, 0)
		a.FMAX4s(3, 3, 0)
	}, func() {
		a.LDRs(0, X6, 0)
		a.DUPs4(0, 0)
		a.FMAX4s(3, 3, 0)
	})
	a.FMAXV(3, 3) // one instruction; amd64 needs four
	a.DUPs4(3, 3)

	// pass 2: exp(x - max), summed.
	a.MOVIzero(4)
	pass(func() {
		a.LDRq(0, X6, 0)
		a.FSUB4s(0, 0, 3)
		emitA64Exp(&a, 0, 1, 2)
		a.STRq(0, X6, 0)
		a.FADD4s(4, 4, 0)
	}, func() {
		a.LDRs(0, X6, 0)
		a.FSUB4s(0, 0, 3)
		emitA64Exp(&a, 0, 1, 2)
		a.MOVIzero(7) // lanes 1..3 held exp(-max): keep lane 0 alone
		a.INSs(7, 0, 0, 0)
		a.STRs(7, X6, 0)
		a.FADD4s(4, 4, 7)
	})

	// pass 3: divide by the sum.
	a.FADDP4s(4, 4, 4)
	a.FADDPs(4, 4)
	a.LDRs(5, X5, 28) // 1.0
	a.FDIV4s(4, 5, 4)
	a.DUPs4(4, 4)
	pass(func() {
		a.LDRq(0, X6, 0)
		a.FMUL4s(0, 0, 4)
		a.STRq(0, X6, 0)
	}, func() {
		a.LDRs(0, X6, 0)
		a.FMUL4s(0, 0, 4)
		a.STRs(0, X6, 0)
	})
	a.RET()
	return a.Bytes()
}

// EmitA64ActMul is the NEON twin of EmitActMul. The activation is baked: it is
// constant from model.Open, so it is a codegen input.
func EmitA64ActMul(k ActKind) []byte { return emitA64Act(k, true) }

// EmitA64SigmoidMul is the NEON twin of EmitSigmoidMul: dst[i] = sigma(dst[i])
// * up[i], the gate in dst and what it gates in up. See EmitSigmoidMul for why
// this is not SiLU.
func EmitA64SigmoidMul() []byte { return emitA64Act(actSigmoid, true) }

// EmitA64Act is the ungated twin, for the vision tower's un-gated FFN.
// It takes a KIND: see EmitAct for why the bool had to go.
func EmitA64Act(k ActKind) []byte { return emitA64Act(k, false) }

func emitA64Act(act ActKind, mul bool) []byte {
	var a A64
	a.LDRx(X1, X0, 0)  // Out -> gate, in place
	a.LDRx(X3, X0, 24) // AScale -> up
	a.LDRx(X5, X0, 56) // Scr -> consts
	a.LDRx(X4, X0, 40) // K -> whole vectors
	a.LDRx(X6, X0, 32) // Rows -> tail elements
	loadA64ExpConsts(&a)
	cursors := []XReg{X1}
	if mul {
		cursors = append(cursors, X3)
	}
	a64ElemLoop(&a, X4, X6, cursors, func(ld func(VReg, XReg), st func(XReg, VReg)) {
		emitA64ActBody(&a, act, mul, ld, st)
	})
	a.RET()
	return a.Bytes()
}

func emitA64ActBody(a *A64, act ActKind, mul bool, ld func(VReg, XReg), st func(XReg, VReg)) {
	ld(0, X1)
	a.MOVvec(3, 0) // keep x
	switch act {
	case ActIdentity:
		// x itself: the multiply below is the whole kernel.
	case ActSiLU:
		a.FNEG4s(0, 0)
		emitA64Exp(a, 0, 1, 2)
		a.FADD4s(0, 0, 23) // + 1
		a.FDIV4s(0, 3, 0)  // x / (1+exp(-x))
	case ActQuickGELU:
		// quick-GELU: x/(1+exp(-1.702x)). SiLU with the exponent scaled.
		a.LDRs(5, X5, 60) // 1.702
		a.DUPs4(5, 5)
		a.FMUL4s(0, 0, 5) // 1.702x
		a.FNEG4s(0, 0)
		emitA64Exp(a, 0, 1, 2)
		a.FADD4s(0, 0, 23) // + 1
		a.FDIV4s(0, 3, 0)  // x / that
	case ActReLU:
		a.MOVIzero(4)
		a.FMAX4s(0, 0, 4)
	case ActReLU2:
		// max(x, 0)^2.
		a.MOVIzero(4)
		a.FMAX4s(0, 0, 4)
		a.FMUL4s(0, 0, 0)
	case actSigmoid:
		// sigma(x) = 1/(1+exp(-x)): SiLU with the numerator 1 instead of x.
		a.FNEG4s(0, 0)
		emitA64Exp(a, 0, 1, 2)
		a.FADD4s(0, 0, 23) // + 1
		a.FDIV4s(0, 23, 0) // 1 / (1+exp(-x))
	case ActSwiGLUOAI:
		// See the amd64 twin: min(gate, 7), quick-GELU on it, times
		// clamp(up, -7, 7) + 1.
		a.LDRs(5, X5, 64) // 7
		a.DUPs4(5, 5)
		a.FMIN4s(0, 0, 5)
		a.MOVvec(3, 0)
		a.LDRs(4, X5, 60) // 1.702
		a.DUPs4(4, 4)
		a.FMUL4s(0, 0, 4)
		a.FNEG4s(0, 0)
		emitA64Exp(a, 0, 1, 2)
		a.FADD4s(0, 0, 23) // + 1
		a.FDIV4s(0, 3, 0)  // x*sigma(1.702x)
		ld(6, X3)
		a.FMIN4s(6, 6, 5)
		a.LDRs(4, X5, 68) // -7
		a.DUPs4(4, 4)
		a.FMAX4s(6, 6, 4)
		a.FADD4s(6, 6, 23) // y + 1
		a.FMUL4s(0, 0, 6)
	case ActSwiGLUClamp:
		// See the amd64 twin: min(gate, 10), SiLU on it, times
		// clamp(up, -10, 10).
		a.LDRs(5, X5, 72) // 10
		a.DUPs4(5, 5)
		a.FMIN4s(0, 0, 5)
		a.MOVvec(3, 0)
		a.FNEG4s(0, 0)
		emitA64Exp(a, 0, 1, 2)
		a.FADD4s(0, 0, 23) // + 1
		a.FDIV4s(0, 3, 0)  // x*sigma(x)
		ld(6, X3)
		a.FMIN4s(6, 6, 5)
		a.LDRs(4, X5, 76) // -10
		a.DUPs4(4, 4)
		a.FMAX4s(6, 6, 4)
		a.FMUL4s(0, 0, 6)
	case ActSqrtSoftplus:
		// See the amd64 twin: max(x,0) + log(1+exp(-|x|)), the log the atanh
		// series EmitA64DeltaGate runs, then the square root.
		a.MOVIzero(4)
		a.FMAX4s(4, 0, 4) // p = max(x, 0)
		a.FABS4s(0, 0)
		a.FNEG4s(0, 0) // -|x|
		emitA64Exp(a, 0, 1, 2)
		a.LDRs(5, X5, 80) // 2
		a.DUPs4(5, 5)
		a.FADD4s(1, 0, 5) // 2 + u
		a.FDIV4s(2, 0, 1) // s
		a.FMUL4s(0, 2, 2) // s^2
		a.LDRs(1, X5, 84) // 1/9
		a.DUPs4(1, 1)
		for _, off := range []int32{88, 92, 96} {
			a.LDRs(6, X5, off)
			a.DUPs4(6, 6)
			a.FMUL4s(1, 1, 0)
			a.FADD4s(1, 1, 6)
		}
		a.FMUL4s(1, 1, 0)
		a.FADD4s(1, 1, 23) // ... + 1
		a.FMUL4s(1, 1, 2)
		a.FMUL4s(1, 1, 5) // 2*s*h
		a.FADD4s(0, 1, 4) // softplus
		a.FSQRT4s(0, 0)
	case ActSitu:
		// See the amd64 twin: c*tanh(x/c) with the rational tanh at both
		// bounds, and sigma(g) dividing the gate's.
		cst := func(v VReg, off int32) {
			a.LDRs(v, X5, off)
			a.DUPs4(v, v)
		}
		bound := func(x VReg, inv, c int32) {
			cst(5, inv)
			a.FMUL4s(0, x, 5) // x/c
			cst(5, 120)
			a.FMIN4s(0, 0, 5)
			cst(5, 124)
			a.FMAX4s(0, 0, 5) // clamped
			a.FMUL4s(2, 0, 0) // x^2
			cst(1, 128)
			for off := int32(132); off <= 152; off += 4 {
				cst(5, off)
				a.FMUL4s(1, 1, 2)
				a.FADD4s(1, 1, 5)
			}
			a.FMUL4s(1, 1, 0) // x*P(x^2)
			cst(7, 156)
			for off := int32(160); off <= 168; off += 4 {
				cst(5, off)
				a.FMUL4s(7, 7, 2)
				a.FADD4s(7, 7, 5)
			}
			a.FDIV4s(0, 1, 7) // tanh
			cst(5, c)
			a.FMUL4s(0, 0, 5) // c*tanh
		}
		bound(3, 104, 108)
		a.MOVvec(4, 0)
		a.FNEG4s(0, 3)
		emitA64Exp(a, 0, 1, 2)
		a.FADD4s(0, 0, 23) // 1 + exp(-g)
		a.FDIV4s(4, 4, 0)  // 4*tanh(g/4)*sigma(g)
		ld(6, X3)
		bound(6, 112, 116)
		a.FMUL4s(0, 0, 4)
	default:
		a.LDRs(5, X5, 44) // 0.044715
		a.DUPs4(5, 5)
		a.FMUL4s(4, 3, 3)
		a.FMUL4s(4, 4, 5)
		a.FADD4s(4, 4, 23)
		a.FMUL4s(4, 4, 3)
		a.LDRs(5, X5, 48) // sqrt(2/pi)
		a.DUPs4(5, 5)
		a.FMUL4s(0, 4, 5)
		a.LDRs(5, X5, 52) // 2.0
		a.DUPs4(5, 5)
		a.FMUL4s(0, 0, 5)
		emitA64Exp(a, 0, 1, 2)
		a.FADD4s(0, 0, 23)
		a.FDIV4s(0, 5, 0) // 2/(exp(2z)+1)
		a.FSUB4s(0, 5, 0) // 1 + tanh = 2 - that
		a.LDRs(5, X5, 56) // 0.5
		a.DUPs4(5, 5)
		a.FMUL4s(0, 0, 5)
		a.FMUL4s(0, 0, 3)
	}
	if mul && act != ActSwiGLUOAI && act != ActSwiGLUClamp && act != ActSitu {
		ld(6, X3)
		a.FMUL4s(0, 0, 6)
	}
	st(X1, 0)
}

// a64ElemLoop is elemLoop's NEON twin: body over K whole vectors of four, then
// over R single elements, K and R in registers and either possibly zero. LDR s
// zeroes the rest of the vector, so the tail runs the vector body on one lane
// with nothing read or written past the end. See elemLoop.
func a64ElemLoop(a *A64, k, r XReg, cursors []XReg, body func(ld func(VReg, XReg), st func(XReg, VReg))) {
	vecLd := func(v VReg, p XReg) { a.LDRq(v, p, 0) }
	vecSt := func(p XReg, v VReg) { a.STRq(v, p, 0) }
	oneLd := func(v VReg, p XReg) { a.LDRs(v, p, 0) }
	oneSt := func(p XReg, v VReg) { a.STRs(v, p, 0) }
	a64ElemLoops(a, k, r, cursors, func() { body(vecLd, vecSt) }, func() { body(oneLd, oneSt) })
}

// a64ElemLoops is elemLoops' NEON twin: the two bodies written separately.
func a64ElemLoops(a *A64, k, r XReg, cursors []XReg, vec, one func()) {
	tail, done := a.Label(), a.Label()
	a.CBZ(k, tail)
	lp := a.Label()
	a.Bind(lp)
	vec()
	for _, c := range cursors {
		a.ADDimm(c, c, 16)
	}
	a.SUBimm(k, k, 1)
	a.CBNZ(k, lp)

	a.Bind(tail)
	a.CBZ(r, done)
	lt := a.Label()
	a.Bind(lt)
	one()
	for _, c := range cursors {
		a.ADDimm(c, c, 4)
	}
	a.SUBimm(r, r, 1)
	a.CBNZ(r, lt)
	a.Bind(done)
}

// EmitRMSNorm is the architecture-neutral name nn/ calls; on arm64 it is the
// NEON twin. Keeping one name means the caller has no build tag and no branch.
func EmitRMSNorm(n int) []byte    { return EmitA64RMSNorm(n) }
func EmitSoftmax() []byte         { return EmitA64Softmax() }
func EmitActMul(k ActKind) []byte { return EmitA64ActMul(k) }
func EmitAct(k ActKind) []byte    { return EmitA64Act(k) }
func EmitSigmoidMul() []byte      { return EmitA64SigmoidMul() }

func EmitLayerNorm(n int, bias bool) []byte { return EmitA64LayerNorm(n, bias) }
func EmitGaussTopK(n int) []byte            { return emitA64RowStat(n, false, RowGauss) }
func EmitMagMatch(n int) []byte             { return emitA64RowStat(n, false, RowMag) }

func EmitAxpy() []byte    { return EmitA64Axpy() }
func EmitScale() []byte   { return EmitA64Scale() }
func EmitSoftcap() []byte { return EmitA64Softcap() }
func EmitClamp() []byte   { return EmitA64Clamp() }

// elemLanes: a NEON vector is 128 bits.
const elemLanes = 4

// EmitA64LayerNorm is the NEON twin of EmitLayerNorm:
// y = (x - mean)/sqrt(var + eps) * w + b.
//
// Three passes for the amd64 kernel's reason: the one-pass E[x^2]-E[x]^2
// identity cancels in f32 on a residual whose mean dwarfs its variance.
//
// x1 out, x2 x, x3 w, x8 b, x5 consts, x4 counter. v30 holds the mean and v31
// the inverse deviation across the whole body.
func EmitA64LayerNorm(n int, bias bool) []byte { return emitA64RowStat(n, bias, RowLayerNorm) }

// emitA64RowStat is the layer norm, Gemma 3n's gaussian top-k or AltUp's
// magnitude match, as the amd64 emitRowStat; v29 holds the match's target.
func emitA64RowStat(n int, bias bool, mode RowMode) []byte {
	var a A64
	a.LDRx(X1, X0, 0)  // Out -> y
	a.LDRx(X2, X0, 16) // A -> x
	a.LDRx(X3, X0, 24) // AScale -> w
	a.LDRx(X8, X0, 8)  // W -> b
	a.LDRx(X5, X0, 56) // Scr -> [1/n, eps, 1]

	// sum walks x with four chains, leaving the total in lane 0 of v0. sub
	// shapes each loaded vector -- identity for the mean, (x-mean)^2 for the
	// variance.
	//
	// The last n%4 are one at a time and must be neutral in lanes 1..3: LDR s
	// zeroes them, which the mean can take and the variance cannot, so tail
	// keeps lane 0 alone (through v28) before the add.
	sum := func(base XReg, sub func(r VReg), tail bool) {
		for i := 0; i < 4; i++ {
			a.MOVIzero(VReg(i))
		}
		a.MOVreg(X6, base)
		if blocks := n / 16; blocks > 0 {
			a.MOVimm(X4, int64(blocks))
			lp := a.Label()
			a.Bind(lp)
			for i := 0; i < 4; i++ {
				a.LDRq(VReg(4+i), X6, int32(16*i))
				sub(VReg(4 + i))
				a.FADD4s(VReg(i), VReg(i), VReg(4+i))
			}
			a.ADDimm(X6, X6, 64)
			a.SUBimm(X4, X4, 1)
			a.CBNZ(X4, lp)
		}
		for off := (n / 16) * 16; off+4 <= n; off += 4 {
			a.LDRq(4, X6, 0)
			sub(4)
			a.FADD4s(0, 0, 4)
			a.ADDimm(X6, X6, 16)
		}
		for i := 0; i < n%4; i++ {
			a.LDRs(4, X6, int32(4*i))
			sub(4)
			if tail {
				a.MOVIzero(28)
				a.INSs(28, 0, 4, 0)
				a.MOVvec(4, 28)
			}
			a.FADD4s(0, 0, 4)
		}
		a.FADD4s(0, 0, 1)
		a.FADD4s(2, 2, 3)
		a.FADD4s(0, 0, 2)
		a.FADDP4s(0, 0, 0)
		a.FADDPs(0, 0)
	}

	// pass 1: the mean, or the match's reference magnitude (v29) with a zero
	// mean.
	if mode == RowMag {
		sum(X3, func(r VReg) { a.FMUL4s(r, r, r) }, false)
		a.LDRs(5, X5, 0) // 1/n
		a.FMULs(0, 0, 5)
		a.FSQRT4s(0, 0)
		a.DUPs4(29, 0)
		a.MOVIzero(30)
	} else {
		sum(X2, func(r VReg) {}, false)
		a.LDRs(5, X5, 0) // 1/n
		a.FMULs(0, 0, 5)
		a.DUPs4(30, 0)
	}

	// pass 2: the variance.
	sum(X2, func(r VReg) {
		a.FSUB4s(r, r, 30)
		a.FMUL4s(r, r, r)
	}, true)
	a.LDRs(5, X5, 0) // 1/n
	a.LDRs(6, X5, 4) // eps
	a.FMULs(0, 0, 5)
	if mode == RowMag {
		a.FMAX4s(0, 0, 6)
	} else {
		a.FADD4s(0, 0, 6)
	}
	a.FSQRT4s(0, 0)
	a.LDRs(7, X5, 8) // 1.0, or the gaussian's c
	switch mode {
	case RowGauss:
		a.FMULs(0, 0, 7)
		a.FADD4s(0, 0, 30) // the cutoff, mean + c*sd
	case RowMag:
		a.FDIV4s(0, 29, 0) // target / magnitude
	default:
		a.FDIV4s(0, 7, 0)
	}
	a.DUPs4(31, 0)
	if mode == RowGauss {
		a.MOVIzero(28)
	}

	// pass 3: y = (x - mean) * inv * w + b.
	// apply is pass 3 on a loaded vector; ld reads w or b, a vector or one
	// element.
	apply := func(r VReg, off int32, ld func(VReg, XReg, int32)) {
		switch mode {
		case RowGauss:
			a.FSUB4s(r, r, 31)
			a.FMAX4s(r, r, 28)
		case RowMag:
			a.FMUL4s(r, r, 31)
		default:
			a.FSUB4s(r, r, 30)
			a.FMUL4s(r, r, 31)
			ld(12, X3, off)
			a.FMUL4s(r, r, 12)
			if bias {
				ld(13, X8, off)
				a.FADD4s(r, r, 13)
			}
		}
	}
	one := func(r VReg, off int32) {
		a.LDRq(r, X6, off)
		apply(r, off, func(v VReg, b XReg, o int32) { a.LDRq(v, b, o) })
		a.STRq(r, X7, off)
	}
	a.MOVreg(X6, X2)
	a.MOVreg(X7, X1)
	if blocks := n / 16; blocks > 0 {
		a.MOVimm(X4, int64(blocks))
		lp := a.Label()
		a.Bind(lp)
		for i := 0; i < 4; i++ {
			one(VReg(4+i), int32(16*i))
		}
		a.ADDimm(X6, X6, 64)
		a.ADDimm(X3, X3, 64)
		a.ADDimm(X8, X8, 64)
		a.ADDimm(X7, X7, 64)
		a.SUBimm(X4, X4, 1)
		a.CBNZ(X4, lp)
	}
	for off := (n / 16) * 16; off+4 <= n; off += 4 {
		one(4, 0)
		a.ADDimm(X6, X6, 16)
		a.ADDimm(X3, X3, 16)
		a.ADDimm(X8, X8, 16)
		a.ADDimm(X7, X7, 16)
	}
	for i := 0; i < n%4; i++ {
		d := int32(4 * i)
		a.LDRs(4, X6, d)
		apply(4, d, func(v VReg, b XReg, o int32) { a.LDRs(v, b, o) })
		a.STRs(4, X7, d)
	}
	a.RET()
	return a.Bytes()
}

// EmitA64Axpy is the NEON twin of EmitAxpy: dst[i] += alpha * src[i], with the
// length in Args.K whole vectors plus Args.Rows single elements and alpha
// read from Scr.
func EmitA64Axpy() []byte {
	var a A64
	a.LDRx(X1, X0, 0)  // Out -> dst, in place
	a.LDRx(X3, X0, 24) // AScale -> src
	a.LDRx(X5, X0, 56) // Scr -> &alpha
	a.LDRx(X4, X0, 40) // K -> whole vectors
	a.LDRx(X6, X0, 32) // Rows -> tail elements
	a.LDRs(0, X5, 0)
	a.DUPs4(0, 0)

	a64ElemLoop(&a, X4, X6, []XReg{X1, X3}, func(ld func(VReg, XReg), st func(XReg, VReg)) {
		ld(1, X1)
		ld(2, X3)
		a.FMLA4s(1, 2, 0)
		st(X1, 1)
	})
	a.RET()
	return a.Bytes()
}

// EmitA64Scale is the NEON twin of EmitScale: dst[i] *= alpha, length in Args.K
// whole vectors plus Args.Rows single elements, alpha from Scr. See EmitScale
// for why it exists.
func EmitA64Scale() []byte {
	var a A64
	a.LDRx(X1, X0, 0)  // Out -> dst, in place
	a.LDRx(X5, X0, 56) // Scr -> &alpha
	a.LDRx(X4, X0, 40) // K -> whole vectors
	a.LDRx(X6, X0, 32) // Rows -> tail elements
	a.LDRs(0, X5, 0)
	a.DUPs4(0, 0)

	a64ElemLoop(&a, X4, X6, []XReg{X1}, func(ld func(VReg, XReg), st func(XReg, VReg)) {
		ld(1, X1)
		a.FMUL4s(1, 1, 0)
		st(X1, 1)
	})
	a.RET()
	return a.Bytes()
}

// EmitA64Clamp is the NEON twin of EmitClamp: x = min(max(x, lo), hi), Scr
// pointing at {lo, hi}.
func EmitA64Clamp() []byte {
	var a A64
	a.LDRx(X1, X0, 0)  // Out -> x, in place
	a.LDRx(X5, X0, 56) // Scr -> {lo, hi}
	a.LDRx(X4, X0, 40) // K -> whole vectors
	a.LDRx(X6, X0, 32) // Rows -> tail elements
	a.LDRs(0, X5, 0)
	a.DUPs4(0, 0)
	a.LDRs(2, X5, 4)
	a.DUPs4(2, 2)

	a64ElemLoop(&a, X4, X6, []XReg{X1}, func(ld func(VReg, XReg), st func(XReg, VReg)) {
		ld(1, X1)
		a.FMAX4s(1, 1, 0)
		a.FMIN4s(1, 1, 2)
		st(X1, 1)
	})
	a.RET()
	return a.Bytes()
}

// EmitA64Softcap is the NEON twin of EmitSoftcap: x = c*tanh(x/c), with W
// pointing at {2/c, c}. See EmitSoftcap for the identity.
func EmitA64Softcap() []byte {
	var a A64
	a.LDRx(X1, X0, 0)  // Out -> x, in place
	a.LDRx(X3, X0, 8)  // W -> {2/c, c}
	a.LDRx(X5, X0, 56) // Scr -> consts
	a.LDRx(X4, X0, 40) // K -> whole vectors
	a.LDRx(X6, X0, 32) // Rows -> tail elements
	loadA64ExpConsts(&a)
	a.LDRs(4, X5, 52) // 2.0
	a.DUPs4(4, 4)
	a.LDRs(5, X3, 0) // 2/c
	a.DUPs4(5, 5)
	a.LDRs(6, X3, 4) // c
	a.DUPs4(6, 6)

	a64ElemLoop(&a, X4, X6, []XReg{X1}, func(ld func(VReg, XReg), st func(XReg, VReg)) {
		ld(0, X1)
		a.FMUL4s(0, 0, 5)
		emitA64Exp(&a, 0, 1, 2)
		a.FADD4s(0, 0, 23) // exp(2x/c) + 1
		a.FDIV4s(0, 4, 0)  // 2/(that)
		a.FSUB4s(0, 23, 0) // tanh(x/c)
		a.FMUL4s(0, 0, 6)
		st(X1, 0)
	})
	a.RET()
	return a.Bytes()
}
