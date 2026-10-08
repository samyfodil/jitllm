package cpu

import (
	"fmt"
	"sync/atomic"
)

// The arm64 kernels without FEAT_DotProd. SDOT is ARMv8.2-A; a Cortex-A53,
// A57, A72 or A73 (a Raspberry Pi 3 or 4) has only ARMv8.0 NEON. RULE 8 makes
// a missing instruction a basic generated kernel, never a refusal, so on such
// a chip every SDOT and SDOTelem an emitter asks for is widened in place:
//
//	smull   t0.8h, vn.8b,  vm.8b      the low eight int8 x int8 products
//	smull2  t1.8h, vn.16b, vm.16b     the high eight
//	saddlp  t0.4s, t0.8h              adjacent pairs, widened to int32
//	saddlp  t1.4s, t1.8h
//	addp    t0.4s, t0.4s, t1.4s       pairs of pairs: lane i is bytes 4i..4i+3
//	add     vd.4s, vd.4s, t0.4s
//
// An int8 x int8 product fits int16 (|-128 * -128| = 16384), and every sum is
// taken at int32 from there on, so the six instructions compute SDOT's four
// lanes exactly: an emulated kernel gives the SAME BITS as the SDOT one, and the
// gates hold it to that. The by-element form first broadcasts vm.s[idx] into
// t1 (DUP), which SMULL2 may then overwrite since it reads before it writes.
//
// The activation quantization and the packed layouts are unchanged: the
// emulation is below the emitters, which only name two scratch vectors
// (A64.DotScratch). The cost is six instructions where one was; this is the
// basic kernel RULE 8 asks for, and a faster sequence (SMLAL chains over a
// layout regrouped for them) is owed, not designed here.

// dotEmulated counts the SDOTs widened since the process started, so a gate can
// assert the no-dotprod path was actually emitted and not merely requested.
var dotEmulated atomic.Int64

// DotEmulated reports how many SDOT/SDOTelem instructions have been widened to
// the ARMv8.0 sequence in this process.
func DotEmulated() int64 { return dotEmulated.Load() }

// DotScratch names the two vectors an emulated SDOT may clobber and decides,
// from the probe, whether this kernel's SDOTs are widened. An emitter calls it
// once, before its first SDOT, with registers it never holds live across one;
// on a chip with FEAT_DotProd they go unused.
func (a *A64) DotScratch(t0, t1 VReg) {
	if t0 == t1 {
		panic(fmt.Sprintf("jit: DotScratch: v%d twice", t0))
	}
	a.dotT = [2]VReg{t0, t1}
	a.dotEmu = a64EmulateDot()
}

func (a *A64) sdotWide(vd, vn, vm VReg, elem bool, idx uint8) {
	t0, t1 := a.dotT[0], a.dotT[1]
	for _, r := range []VReg{vd, vn, vm} {
		if r == t0 || r == t1 {
			panic(fmt.Sprintf("jit: emulated SDOT v%d, v%d, v%d overlaps its scratch v%d, v%d", vd, vn, vm, t0, t1))
		}
	}
	dotEmulated.Add(1)
	if elem {
		a.DUPs4lane(t1, vm, idx)
		vm = t1
	}
	a.SMULL8b(t0, vn, vm)
	a.SMULL16b(t1, vn, vm)
	a.SADDLP8h(t0, t0)
	a.SADDLP8h(t1, t1)
	a.ADDP4s(t0, t0, t1)
	a.ADD4s(vd, vd, t0)
}

// SMULL8b is smull vd.8h, vn.8b, vm.8b: the low eight signed byte products.
func (a *A64) SMULL8b(vd, vn, vm VReg) {
	a.w(0x0E20C000 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

// SMULL16b is smull2 vd.8h, vn.16b, vm.16b: the high eight.
func (a *A64) SMULL16b(vd, vn, vm VReg) {
	a.w(0x4E20C000 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

// SADDLP8h is saddlp vd.4s, vn.8h: adjacent int16 pairs summed into int32.
func (a *A64) SADDLP8h(vd, vn VReg) {
	a.w(0x4E602800 | uint32(vn)<<5 | uint32(vd))
}

// ADDP4s is addp vd.4s, vn.4s, vm.4s: integer pairwise add, vn's pairs in the
// low two lanes and vm's in the high two.
func (a *A64) ADDP4s(vd, vn, vm VReg) {
	a.w(0x4EA0BC00 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}
