package ptx

import (
	"fmt"

	"github.com/samyfodil/jitllm/jit/gpu/ir"
)

// Lower translates a kernel into PTX text.
//
// It is one forward walk with no analysis: the IR is already in the order the
// target wants, control flow is structured, and ptxas owns everything a real
// compiler back end would do. The only lookahead is at a loop head, where the
// loop-carried values have to be initialised before the label they branch back
// to -- which is the one thing a linear walk cannot discover after the fact.
func Lower(k *ir.Kernel, target string) (string, error) {
	if err := k.Validate(); err != nil {
		return "", err
	}
	for i, o := range k.Ops {
		if o.Kind == ir.OpPhi && o.Tile != nil {
			// Named here and not at the phi, whose register is allocated at the
			// loop head before the walk reaches any tile op's refusal below.
			return "", fmt.Errorf("ptx: op %d: a loop-carried collective tile has no lowering: "+
				"this backend takes the fragment-shaped mma (ir.OpMMA)", i)
		}
	}
	var b Buf
	reg := make([]Reg, len(k.Ops)+1) // value id -> register, 1-based
	// The one op with several results. OpMMA writes its D fragment into these
	// and OpMMAGet aliases them, so nothing else in the walk needs to know.
	mma := make([][]Reg, len(k.Ops)+1)
	// Which values are shared arrays, so a load or store on one is emitted in
	// the shared address space rather than global.
	shared := make([]bool, len(k.Ops)+1)
	// A vector access needs its whole width aligned, and a shared array is
	// otherwise declared at its element's; 16 covers every LoadV/StoreV. Only
	// a kernel that has one gets it, so no other kernel's text changes.
	sharedAlign := 0
	for _, o := range k.Ops {
		if o.Kind == ir.OpLoadV || o.Kind == ir.OpStoreV {
			sharedAlign = 16
		}
	}

	class := func(t ir.Type) Class {
		switch t {
		case ir.F32:
			return F32
		case ir.Pred:
			return Pred
		case ir.Ptr:
			return B64
		default:
			return B32
		}
	}
	sfx := func(t ir.Type) string {
		switch t {
		case ir.F32:
			return "f32"
		case ir.I32:
			return "s32"
		default:
			return "u32"
		}
	}

	params := make([]Param, len(k.Params))
	for i, p := range k.Params {
		params[i] = Param{Name: p.Name, Type: "u64"}
	}

	type loopFrame struct {
		label   string
		merge   string // non-empty only for a runtime count, which needs an entry guard
		counter Reg
		phis    []ir.Value
	}
	var stack []loopFrame

	for i := 0; i < len(k.Ops); i++ {
		o := k.Ops[i]
		v := ir.Value(i + 1)
		a0, a1, a2 := o.Args[0], o.Args[1], o.Args[2]

		switch o.Kind {
		case ir.OpShared:
			shared[v] = true
			// The array is a kernel-scope `.shared` declaration (ir.Validate
			// requires OpShared before any loop) and the value is its address,
			// kept in the shared space: loads and stores on it are
			// space-qualified (see space), since a generic or global access
			// to it assembles cleanly and reads the wrong memory.
			nm := fmt.Sprintf("jitllm_sh%d", v)
			b.Decl(".shared .align %d .b8 %s[%d];", max(elemSize(o.Type), sharedAlign), nm, int(o.Imm)*elemSize(o.Type))
			r := b.Reg(B64)
			b.Op("mov.u64 %v,%s;", r, nm)
			reg[v] = r

		case ir.OpBarrier:
			b.Op("bar.sync 0;")

		case ir.OpParam:
			r := b.Reg(class(o.Type))
			b.Op("ld.param.u64 %v,[%s];", r, k.Params[o.Imm].Name)
			b.Op("cvta.to.global.u64 %v,%v;", r, r)
			reg[v] = r

		case ir.OpConst:
			r := b.Reg(class(o.Type))
			if o.Type == ir.F32 {
				b.Op("mov.f32 %v,0f%08X;", r, uint32(o.Imm))
			} else {
				b.Op("mov.u32 %v,%d;", r, o.Imm)
			}
			reg[v] = r

		case ir.OpTID, ir.OpCTAID, ir.OpNTID:
			r := b.Reg(B32)
			name := map[ir.Kind]string{ir.OpTID: "%tid.x", ir.OpCTAID: "%ctaid.x", ir.OpNTID: "%ntid.x"}[o.Kind]
			b.Op("mov.u32 %v,%s;", r, name)
			reg[v] = r

		case ir.OpAdd, ir.OpSub, ir.OpMul, ir.OpDiv, ir.OpRem, ir.OpMin, ir.OpMax:
			r := b.Reg(class(o.Type))
			op := map[ir.Kind]string{
				ir.OpAdd: "add", ir.OpSub: "sub", ir.OpMul: "mul",
				ir.OpDiv: "div", ir.OpRem: "rem", ir.OpMin: "min", ir.OpMax: "max",
			}[o.Kind]
			if o.Kind == ir.OpMul && o.Type != ir.F32 {
				op = "mul.lo" // PTX defaults integer mul to the WIDE form
			}
			if o.Kind == ir.OpDiv && o.Type == ir.F32 {
				// ptxas refuses a bare div.f32. .rn gives the IEEE result; the
				// divides here are reciprocals off the hot path.
				op = "div.rn"
			}
			b.Op("%s.%s %v,%v,%v;", op, sfx(o.Type), r, reg[a0], reg[a1])
			reg[v] = r

		case ir.OpShl, ir.OpShr, ir.OpAnd, ir.OpXor:
			r := b.Reg(B32)
			// shr.u32, not shr.s32: OpShr is logical (see the IR comment).
			op := map[ir.Kind]string{ir.OpShl: "shl.b32", ir.OpShr: "shr.u32", ir.OpAnd: "and.b32", ir.OpXor: "xor.b32"}[o.Kind]
			b.Op("%s %v,%v,%v;", op, r, reg[a0], reg[a1])
			reg[v] = r

		case ir.OpFunnel:
			r := b.Reg(B32)
			b.Op("shf.r.clamp.b32 %v,%v,%v,%v;", r, reg[a0], reg[a1], reg[a2])
			reg[v] = r

		case ir.OpLt:
			r := b.Reg(Pred)
			b.Op("setp.lt.%s %v,%v,%v;", sfx(k.Ops[a0-1].Type), r, reg[a0], reg[a1])
			reg[v] = r

		case ir.OpSelect:
			// selp takes the predicate LAST, and the type suffix is the
			// selected values' type, not the predicate's.
			r := b.Reg(class(o.Type))
			b.Op("selp.%s %v,%v,%v,%v;", sfx(o.Type), r, reg[a1], reg[a2], reg[a0])
			reg[v] = r

		case ir.OpCvtF32:
			r := b.Reg(F32)
			b.Op("cvt.rn.f32.s32 %v,%v;", r, reg[a0])
			reg[v] = r

		case ir.OpExp:
			// exp(x) = 2^(x*log2e); ex2.approx is the only exponential PTX has.
			t := b.Reg(F32)
			r := b.Reg(F32)
			b.Op("mul.f32 %v,%v,0f3FB8AA3B;", t, reg[a0]) // log2(e)
			b.Op("ex2.approx.f32 %v,%v;", r, t)
			reg[v] = r

		case ir.OpSqrt:
			r := b.Reg(F32)
			b.Op("sqrt.rn.f32 %v,%v;", r, reg[a0])
			reg[v] = r

		case ir.OpCvtI32:
			r := b.Reg(B32)
			// Round-toward-zero, matching Go's float-to-int conversion.
			b.Op("cvt.rzi.s32.f32 %v,%v;", r, reg[a0])
			reg[v] = r

		case ir.OpBitcast:
			r := b.Reg(class(o.Type))
			b.Op("mov.b32 %v,%v;", r, reg[a0])
			reg[v] = r

		case ir.OpPackF16:
			// PTX has no pack primitive here: narrow each operand to .b16,
			// widen both back to .b32 and splice. `cvt.rn` is the round-to-
			// nearest-even the other two backends' PackHalf2x16 also uses, so
			// all three agree bit for bit.
			h0, h1 := b.Reg(B16), b.Reg(B16)
			r0, r1 := b.Reg(B32), b.Reg(B32)
			r := b.Reg(B32)
			b.Op("cvt.rn.f16.f32 %v,%v;", h0, reg[a0])
			b.Op("cvt.rn.f16.f32 %v,%v;", h1, reg[a1])
			b.Op("cvt.u32.u16 %v,%v;", r0, h0)
			b.Op("cvt.u32.u16 %v,%v;", r1, h1)
			b.Op("shl.b32 %v,%v,16;", r1, r1)
			b.Op("or.b32 %v,%v,%v;", r, r0, r1)
			reg[v] = r

		case ir.OpCvtF16H:
			// Two instructions: narrow to a 16-bit register, then convert. The
			// B16 class already existed for exactly this shape of operand.
			h := b.Reg(B16)
			r := b.Reg(F32)
			b.Op("cvt.u16.u32 %v,%v;", h, reg[a0])
			b.Op("cvt.f32.f16 %v,%v;", r, h)
			reg[v] = r

		case ir.OpLoad:
			// One mad.wide.u32 turns an element index into an address: the
			// multiply by the element size and the add to the base in a single
			// instruction, which is what makes indexed access affordable here.
			ad := b.Reg(B64)
			r := b.Reg(class(o.Type))
			b.Op("mad.wide.u32 %v,%v,%d,%v;", ad, reg[a1], elemSize(o.Type), reg[a0])
			b.Op("ld.%s.%s %v,[%v+%d];", space(shared[a0]), ldSuffix(o.Type), r, ad, o.Imm*int64(elemSize(o.Type)))
			reg[v] = r

		case ir.OpStore:
			et := k.Ops[a2-1].Type
			ad := b.Reg(B64)
			b.Op("mad.wide.u32 %v,%v,%d,%v;", ad, reg[a1], elemSize(et), reg[a0])
			b.Op("st.%s.%s [%v+%d],%v;", space(shared[a0]), ldSuffix(et), ad, o.Imm*int64(elemSize(et)), reg[a2])

		case ir.OpDot4:
			r := b.Reg(B32)
			// The accumulator is read and written, and PTX has no two-operand
			// form, so the incoming accumulator is copied first. ptxas coalesces
			// this away when the value is dead afterwards, which it always is in
			// a Phi chain.
			b.Op("mov.u32 %v,%v;", r, reg[a2])
			b.Op("dp4a.s32.s32 %v,%v,%v,%v;", r, reg[a0], reg[a1], r)
			reg[v] = r

		case ir.OpMMA:
			// .row.col is not symmetric: A is row-major MxK (the weights, k
			// contiguous) and B column-major KxN (activations, one token per
			// column). Swapping them assembles and transposes the result.
			sh := ir.MMAShapeOf(o)
			na, nb, nc := sh.Frags()
			d := make([]Reg, nc)
			for i := range d {
				d[i] = b.Reg(B32)
			}
			// c is copied into d first: PTX allows d and c to alias, but the
			// incoming accumulators are Phi values that other ops may still
			// read, so aliasing them here would clobber a live value. ptxas
			// coalesces the movs away when they are dead, as in dp4a above.
			for i := 0; i < nc; i++ {
				// b32 either way: mma's accumulator registers are untyped bits,
				// and a float32 moved through mov.u32 is unchanged.
				b.Op("mov.b32 %v,%v;", d[i], reg[o.Vargs[na+nb+i]])
			}
			list := func(vs []ir.Value) string {
				out := "{"
				for i, x := range vs {
					if i > 0 {
						out += ","
					}
					out += reg[x].String()
				}
				return out + "}"
			}
			dl := "{"
			for i, r := range d {
				if i > 0 {
					dl += ","
				}
				dl += r.String()
			}
			dl += "}"
			// The accumulator type comes FIRST in the mnemonic and the operand
			// type second: .s32.s8.s8.s32 is an int8 multiply into int32,
			// .f32.f16.f16.f32 a binary16 multiply into float32. Both keep the
			// accumulator wider than the operands.
			types := "s32.s8.s8.s32"
			if sh.Kind == ir.MMAF16 {
				types = "f32.f16.f16.f32"
			}
			b.Op("mma.sync.aligned.m%dn%dk%d.row.col.%s %s,%s,%s,%s;",
				sh.M, sh.N, sh.K, types, dl, list(o.Vargs[:na]), list(o.Vargs[na:na+nb]), dl)
			mma[v] = d
			reg[v] = d[0]

		case ir.OpSubF16x2:
			r := b.Reg(B32)
			b.Op("sub.rn.f16x2 %v,%v,%v;", r, reg[a0], reg[a1])
			reg[v] = r

		case ir.OpFmaF16x2:
			r := b.Reg(B32)
			b.Op("fma.rn.f16x2 %v,%v,%v,%v;", r, reg[a0], reg[a1], reg[a2])
			reg[v] = r

		case ir.OpMMAGet:
			reg[v] = mma[a0][o.Imm]

		case ir.OpLoadV:
			// One address for the whole vector, as OpLoad computes it, and the
			// components into consecutive registers OpLoadVGet then aliases.
			// .b32 because the lanes are untyped bits to a vector access.
			off, n := ir.VecOf(o)
			ad := b.Reg(B64)
			b.Op("mad.wide.u32 %v,%v,%d,%v;", ad, reg[a1], elemSize(o.Type), reg[a0])
			rs := make([]Reg, n)
			list := "{"
			for i := range rs {
				rs[i] = b.Reg(class(o.Type))
				if i > 0 {
					list += ","
				}
				list += rs[i].String()
			}
			b.Op("ld.%s.v%d.b32 %s},[%v+%d];", space(shared[a0]), n, list, ad, off*int64(elemSize(o.Type)))
			mma[v] = rs
			reg[v] = rs[0]

		case ir.OpLoadVGet:
			reg[v] = mma[a0][o.Imm]

		case ir.OpStoreV:
			off, n := ir.VecOf(o)
			et := k.Ops[o.Vargs[0]-1].Type
			ad := b.Reg(B64)
			b.Op("mad.wide.u32 %v,%v,%d,%v;", ad, reg[a1], elemSize(et), reg[a0])
			list := "{"
			for i := 0; i < n; i++ {
				if i > 0 {
					list += ","
				}
				list += reg[o.Vargs[i]].String()
			}
			b.Op("st.%s.v%d.b32 [%v+%d],%s};", space(shared[a0]), n, ad, off*int64(elemSize(et)), list)

		case ir.OpFma:
			r := b.Reg(F32)
			b.Op("fma.rn.f32 %v,%v,%v,%v;", r, reg[a0], reg[a1], reg[a2])
			reg[v] = r

		case ir.OpShuffleXor:
			// shfl.sync is a warp-wide barrier as well as a data move: it waits
			// until every thread in membermask has reached a shfl.sync with the
			// same qualifiers. That is what makes it correct after the per-lane
			// loops above it, which run a different number of iterations per
			// lane and therefore arrive here diverged.
			//
			// The literal 31 is the clamp/segmask operand: segmask 0 and
			// maxLane 31 means the whole warp is one segment, so lane^mask is
			// always a valid source. -1 is the member mask, which is what nvcc
			// emits for __shfl_xor_sync(0xffffffff, ...).
			//
			// The value moves through a .b32 register rather than being
			// shuffled in place: shfl is a bit-type instruction, and the two
			// movs cost nothing after ptxas coalesces them.
			src := b.Reg(B32)
			dst := b.Reg(B32)
			b.Op("mov.b32 %v,%v;", src, reg[a0])
			b.Op("shfl.sync.bfly.b32 %v,%v,%d,31,-1;", dst, src, o.Imm)
			r := b.Reg(class(o.Type))
			b.Op("mov.b32 %v,%v;", r, dst)
			reg[v] = r

		case ir.OpLoop:
			// Loop-carried values are initialised BEFORE the label, which needs
			// a scan forward to the matching EndLoop.
			f := loopFrame{label: b.Label(), counter: b.Reg(B32)}
			depth := 0
			for j := i + 1; j < len(k.Ops); j++ {
				switch k.Ops[j].Kind {
				case ir.OpLoop:
					depth++
				case ir.OpEndLoop:
					if depth == 0 {
						j = len(k.Ops)
						continue
					}
					depth--
				case ir.OpPhi:
					if depth == 0 {
						f.phis = append(f.phis, ir.Value(j+1))
					}
				}
			}
			for _, p := range f.phis {
				po := k.Ops[p-1]
				r := b.Reg(class(po.Type))
				reg[p] = r
				b.Op("mov.%s %v,%v;", movSuffix(po.Type), r, reg[po.Args[0]])
			}
			if o.Imm < 0 {
				// A runtime count. The loop below is a countdown do-while, so a
				// count of zero would wrap to 2^32 iterations; guard the entry
				// and skip to a merge label instead. SPIR-V and MSL test before
				// the body and need no such guard.
				f.merge = b.Label()
				b.Op("mov.u32 %v,%v;", f.counter, reg[o.Args[0]])
				pr := b.Reg(Pred)
				b.Op("setp.eq.u32 %v,%v,0;", pr, f.counter)
				b.Op("@%v bra %s;", pr, f.merge)
			} else {
				b.Op("mov.u32 %v,%d;", f.counter, o.Imm)
			}
			b.Bind(f.label)
			stack = append(stack, f)

		case ir.OpPhi:
			// The register was allocated at the loop head; nothing to emit.

		case ir.OpEndLoop:
			f := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			for _, p := range f.phis {
				po := k.Ops[p-1]
				if reg[po.Args[1]] != reg[p] {
					b.Op("mov.%s %v,%v;", movSuffix(po.Type), reg[p], reg[po.Args[1]])
				}
			}
			pr := b.Reg(Pred)
			b.Op("sub.u32 %v,%v,1;", f.counter, f.counter)
			b.Op("setp.ne.u32 %v,%v,0;", pr, f.counter)
			b.Op("@%v bra %s;", pr, f.label)
			if f.merge != "" {
				b.Bind(f.merge)
			}

		default:
			if ir.IsTileOp(o.Kind) {
				// A deliberate refusal: PTX's mma.sync takes per-lane fragments
				// with a documented layout (ir.OpMMA); a collective tile is the
				// opposite contract, with no lane layout to map.
				return "", fmt.Errorf("ptx: no lowering for %s: this backend takes the "+
					"fragment-shaped mma (ir.OpMMA), and a collective tile has no lane "+
					"layout to map onto one", o.Kind)
			}
			return "", fmt.Errorf("ptx: no lowering for %s", o.Kind)
		}
	}
	b.Op("ret;")
	return b.Text(target, k.Name, params), nil
}

func elemSize(t ir.Type) int {
	return 4 // every element type in this IR is 32 bits wide
}

func ldSuffix(t ir.Type) string {
	switch t {
	case ir.F32:
		return "f32"
	case ir.I32:
		return "s32"
	default:
		return "u32"
	}
}

func movSuffix(t ir.Type) string {
	switch t {
	case ir.F32:
		return "f32"
	case ir.Ptr:
		return "u64"
	}
	return "u32"
}

// space names the PTX address space for a load or store. A shared array's
// address is never cvta'd to generic, so its accesses must say so: `ld.global`
// on a shared address assembles cleanly and reads the wrong memory.
func space(isShared bool) string {
	if isShared {
		return "shared"
	}
	return "global"
}
