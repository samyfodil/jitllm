package nn

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/samyfodil/jitllm/format/quant"
	"github.com/samyfodil/jitllm/jit/cpu"
	"github.com/samyfodil/jitllm/jit/gpu/kernels"
)

// The embedding lookup: one row of a table into float32, generated.

type rowKernel struct {
	code  *cpu.Code
	konst []byte
}

// rowKey keys a packed-row kernel by format AND tier (see elemSet for why the
// tier is in every package-level key).
type rowKey struct {
	t cpu.Tier
	q kernels.Quant
}

var (
	rowKernels sync.Map // rowKey -> *rowKernel
	widenK     [cpu.NumTiers][2]*cpu.Code
	widenOnce  tierOnce
)

// RowPacked32JIT writes row r of a packed tensor of nrows rows and k columns
// into dst.
func RowPacked32JIT(dst []float32, t quant.Type, p *Packed, r, nrows, k int) {
	q, ok := kernels.QuantOf(t)
	if !ok {
		panic(fmt.Sprintf("jit: %s has no packed layout", t))
	}
	if len(dst) < k || r < 0 || r >= nrows {
		panic(fmt.Sprintf("jit: row %d of %d into %d floats, want %d", r, nrows, len(dst), k))
	}
	rkey := rowKey{cpu.HostTier(), q}
	v, ok := rowKernels.Load(rkey)
	if !ok {
		c := mustEmit("packed_row_" + t.String())(cpu.EmittersFor(rkey.t).PackedRow(q))
		v, _ = rowKernels.LoadOrStore(rkey, &rowKernel{c, cpu.PackedRowConsts(q)})
	}
	rk := v.(*rowKernel)
	stride, row := p.Stride, p.Row+r
	if stride == 0 {
		stride = nrows
	}
	// The word, then the slot within it: two rows share a word on a narrow f16
	// scale and FOUR on MXFP4's E8M0 byte, so the offset is the slot times the
	// format's row width rather than a fixed 2.
	di, slot := kernels.DIndex(q, 0, stride, row)
	pd := 4*di + slot*cpu.DRowBytes(t)
	sub, _, _, _ := kernels.Layout(q)
	perSuper, _ := kernels.ScaleLayout(q)
	// The kernel's gather scratch is on the stack: Code.Call does not let args
	// escape, and a sync.Pool here allocated again after every GC cycle.
	var scr [cpu.PackedRowScratch]byte
	args := cpu.Args{
		Out: &dst[0], W: &p.QS[4*row], RowStr: int64(4 * stride),
		PD: &p.D[pd], DStr: int64(cpu.DSuperBytes(t, stride)),
		K:   int64(k / (sub * perSuper)),
		Scr: &rk.konst[0], Scratch: &scr[0],
	}
	if len(p.SC) > 0 {
		args.PSC = &p.SC[4*row]
	}
	rk.code.Call(&args)
}

// Row32JIT writes row r of a row-major F32, F16 or BF16 table of k columns
// into dst: a copy for F32, a generated widen for the two halves.
func Row32JIT(dst []float32, t quant.Type, w []byte, r, k int) {
	es := int(t.BlockBytes())
	if len(dst) < k || len(w) < (r+1)*k*es {
		panic(fmt.Sprintf("jit: row %d of a %s table of %d bytes, k=%d", r, t, len(w), k))
	}
	switch t {
	case quant.F32:
		copy(dst[:k], unsafe.Slice((*float32)(unsafe.Pointer(&w[r*k*4])), k))
		return
	case quant.F16, quant.BF16:
	default:
		panic(fmt.Sprintf("jit: a row-major %s table has no reader; a container packs it", t))
	}
	tier := cpu.HostTier()
	widenOnce.do(tier, func() {
		em := cpu.EmittersFor(tier)
		widenK[tier][0] = mustEmit("widen_f16")(em.Widen(false))
		widenK[tier][1] = mustEmit("widen_bf16")(em.Widen(true))
	})
	c := widenK[tier][0]
	if t == quant.BF16 {
		c = widenK[tier][1]
	}
	c.Call(&cpu.Args{Out: &dst[0], W: &w[r*k*2], K: int64(k / cpu.ElemLanes), Rows: int64(k % cpu.ElemLanes)})
}
