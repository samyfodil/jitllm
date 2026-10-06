//go:build amd64 || arm64

package nn

import (
	"github.com/samyfodil/jitllm/format/quant"

	"github.com/samyfodil/jitllm/jit/cpu"
	"github.com/samyfodil/jitllm/jit/gpu/kernels"
)

// MatVecPackedMulti computes several packed matvecs that share one activation
// (a mixture's experts' gate and up), in one pool region instead of one region
// each. The arithmetic is the same fused kernel on the same bytes; the pool
// splits one group space spanning every matrix, so a worker that finishes one
// expert's rows walks into the next instead of waiting at a barrier.
//
// Every matrix must share the type, shape and the fused kernel's row-group
// alignment -- a mixture's bank does by construction. Anything else returns
// false and the caller keeps its loop, which is what makes this additive.
func (f *JIT) MatVecPackedMulti(outs [][]float32, t quant.Type, ps []*Packed, x []float32, nrows, k int) bool {
	m := len(ps)
	if f == nil || m == 0 || len(outs) != m || len(x) < k || nrows <= 0 {
		return false
	}
	x = x[:k]
	// One matrix is not a batch; MatVecPacked serves it.
	if m == 1 {
		return false
	}
	fused := f.packedFused[t]
	step := cpu.PackedOuterElems(t)
	if !f.MixtureBatches(t) || step == 0 || k%step != 0 || k > len(f.q) ||
		nrows%grpOf(t) != 0 {
		return false
	}
	nb := k / cpu.Q8Block
	if 2*nb > len(f.pairs) {
		return false
	}
	for i, p := range ps {
		if p == nil || len(p.QS) == 0 || len(outs[i]) < nrows {
			return false
		}
	}
	f.prepAct(t, x, k, nb)

	groups := nrows / grpOf(t)
	total := m * groups
	per := f.fusedChunk(total, t, k)
	rowsPer := per * grpOf(t)
	f.growPackedScratch(f.pool.Max() * rowsPer)
	q, _ := kernels.QuantOf(t)
	j := &f.hot.multi
	j.fc = f.newFusedCall(fused, t, q, k, min(rowsPer, nrows))
	j.ps, j.outs, j.groups, j.nrows, j.rowsPer = ps, outs, groups, nrows, rowsPer
	for _, o := range outs {
		clear(o[:nrows])
	}
	f.pool.DoLabeled(regionLabel(t, "/multi"), total, per, f.hot.multiFn(f))
	j.ps, j.outs = nil, nil
	return true
}

// MatVecPackedGather computes m packed matvecs where each matrix has its own
// activation (a mixture's down projections), in two pool regions: one that
// quantizes every activation and one that does all the arithmetic. Activation
// j is x[j*xstride : j*xstride+k].
//
// It shares MatMulPacked's activation scratch (growPackedMM) and leaves f.q and
// f.pairs alone, so MatVecPacked's cached activation survives the call.
func (f *JIT) MatVecPackedGather(outs [][]float32, t quant.Type, ps []*Packed, x []float32, xstride, nrows, k int) bool {
	m := len(ps)
	if f == nil || m < 2 || len(outs) != m || nrows <= 0 || k <= 0 ||
		xstride < k || len(x) < (m-1)*xstride+k {
		return false
	}
	fused := f.packedFused[t]
	step := cpu.PackedOuterElems(t)
	if !f.MixtureBatches(t) || step == 0 || k%step != 0 || nrows%grpOf(t) != 0 {
		return false
	}
	nb := k / cpu.Q8Block
	if nb == 0 {
		return false
	}
	for i, p := range ps {
		if p == nil || len(p.QS) == 0 || len(outs[i]) < nrows {
			return false
		}
	}
	half := 0
	if cpu.NeedsHalfSums(t) {
		half = k / 16
	}
	f.growPackedMM(m, k, 2*nb, half)

	tq := f.quantStart()
	// One region for every activation's quantize, where there were m.
	gq := &f.hot.gatherQuant
	gq.qk, gq.t, gq.x = f.quantKernel(t), t, x
	gq.xstride, gq.k, gq.nb, gq.half, gq.win = xstride, k, nb, half, f.actWindow
	f.pool.Do(m, max(1, m/(4*f.pool.N())), f.hot.gatherQuantFn(f))
	gq.x = nil
	f.quantSince(tq)

	groups := nrows / grpOf(t)
	total := m * groups
	per := f.fusedChunk(total, t, k)
	rowsPer := per * grpOf(t)
	f.growPackedScratch(f.pool.Max() * rowsPer)
	q, _ := kernels.QuantOf(t)
	j := &f.hot.gather
	j.fc = f.newFusedCall(fused, t, q, k, min(rowsPer, nrows))
	j.ps, j.outs, j.groups, j.nrows, j.rowsPer = ps, outs, groups, nrows, rowsPer
	j.half, j.nb, j.k = half, nb, k
	// The kernel accumulates into Out, so the caller owns the zero.
	for _, o := range outs {
		clear(o[:nrows])
	}
	f.pool.DoLabeled(regionLabel(t, "/gather"), total, per, f.hot.gatherFn(f))
	j.ps, j.outs = nil, nil
	return true
}

// MixtureBatches reports whether MatVecPackedMulti and MatVecPackedGather
// serve t here, rather than declining to the caller's per-expert loop.
//
// Not on arm64 by default: there the fused kernel loses to the tiled one at a
// mixture's shapes, so the per-expert loop goes through
// MatVecPacked's per-shape choice instead. See
// docs/engineering-history/cpu-kernels.md.
func (f *JIT) MixtureBatches(t quant.Type) bool {
	if f == nil || f.packedFused[t] == nil || f.cfg.MixtureBatch < 0 {
		return false
	}
	return f.cfg.MixtureBatch > 0 || f.tier != cpu.TierNEON
}

// grpOf is the fused kernel's row group for t. It varies by format (Q5_K's is
// eight), and Args.Rows counts groups, so a hardcoded group reads past the
// payload.
func grpOf(t quant.Type) int { return cpu.PackedFusedGroupOf(t) }

// multiRegion is MatVecPackedMulti's or MatVecPackedGather's arithmetic region,
// kept on the JIT with its arguments in fields (see hotRegions): a mixture runs
// them every layer of every token, and a closure over the call's locals was two
// heap objects each time.
type multiRegion struct {
	fc                     fusedCall
	ps                     []*Packed
	outs                   [][]float32
	groups, nrows, rowsPer int
	half, nb, k            int // gather only: each matrix's own activation
	fn                     func(worker, lo, hi int)
}

func (h *hotRegions) multiFn(f *JIT) func(worker, lo, hi int) {
	if h.multi.fn == nil {
		h.multi.fn = func(worker, lo, hi int) {
			j := &h.multi
			// A worker's range may span matrices, so the call is per matrix and
			// the loop walks the boundaries.
			for lo < hi {
				mi := lo / j.groups
				g0 := lo - mi*j.groups
				g1 := min(hi-mi*j.groups, j.groups)
				p, out := j.ps[mi], j.outs[mi]
				// The stride is the buffer's row count, not this weight's, so
				// both come from the Packed being read, never from the batch.
				stride, row := p.Stride, p.Row
				if stride == 0 {
					stride = j.nrows
				}
				j.fc.run(p, out, row, stride, g0, g1, f.q, f.pairs, f.half, worker*j.rowsPer)
				lo = mi*j.groups + g1
			}
		}
	}
	return h.multi.fn
}

func (h *hotRegions) gatherFn(f *JIT) func(worker, lo, hi int) {
	if h.gather.fn == nil {
		h.gather.fn = func(worker, lo, hi int) {
			j := &h.gather
			// A worker's range may span matrices, which is the point.
			for lo < hi {
				mi := lo / j.groups
				g0 := lo - mi*j.groups
				g1 := min(hi-mi*j.groups, j.groups)
				p, out := j.ps[mi], j.outs[mi]
				stride, row := p.Stride, p.Row
				if stride == 0 {
					stride = j.nrows
				}
				var hs []float32
				if j.half > 0 {
					hs = f.mhalf[mi*j.half : (mi+1)*j.half]
				}
				k, nb := j.k, j.nb
				j.fc.run(p, out, row, stride, g0, g1, f.mq[mi*k:(mi+1)*k], f.mpairs[mi*2*nb:(mi+1)*2*nb], hs, worker*j.rowsPer)
				lo = mi*j.groups + g1
			}
		}
	}
	return h.gather.fn
}

func (h *hotRegions) gatherQuantFn(f *JIT) func(worker, lo, hi int) {
	if h.gatherQuant.fn == nil {
		h.gatherQuant.fn = func(wk, lo, hi int) {
			j := &h.gatherQuant
			k, nb := j.k, j.nb
			for i := lo; i < hi; i++ {
				var hs []float32
				if j.half > 0 {
					hs = f.mhalf[i*j.half : (i+1)*j.half]
				}
				j.qk.Run(j.t, f.mq[i*k:(i+1)*k], f.mpairs[i*2*nb:(i+1)*2*nb], hs, f.quantScratch(wk),
					j.x[i*j.xstride:i*j.xstride+k], k, 0, nb, j.win)
			}
		}
	}
	return h.gatherQuant.fn
}
