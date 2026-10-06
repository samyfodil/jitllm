package model

import (
	"unsafe"

	"github.com/samyfodil/jitllm/engine/nn"
	"github.com/samyfodil/jitllm/format/quant"
)

// kvLayout is where a (sequence slot, position, kv head) lives in the host KV
// cache, so the layout is one decision shared by every attention path, the
// cache write and the migrations. The attention kernels advance by Stride and
// read HeadDim elements per step; head-major makes that walk contiguous (see
// docs/engineering-history/cpu-kernels.md for the measurements). Both layouts
// occupy the same bytes and differ only in order.
//
//	row-major   [slot][position][kv head][dim]   Stride = kvDim
//	head-major  [slot][kv head][position][dim]   Stride = headDim
type kvLayout struct {
	headMajor bool
	maxSeq    int
	nKV       int
	headDim   int
	// elem is the bytes one cached element occupies: 4 for f32, 2 for binary16.
	// The cache stays a []float32 and holds packed halves at elem == 2; every
	// offset handed out is 4-byte aligned because headDim is a multiple of 8.
	elem int
}

// slots is how many float32 slots a cache of n elements needs.
func (l kvLayout) slots(n int) int { return n * l.elem / 4 }

func (l kvLayout) kvDim() int { return l.nKV * l.headDim }

// Stride is the element distance between one head's consecutive positions, and
// is exactly what the generated kernels bake as kvStride.
func (l kvLayout) Stride() int {
	if l.headMajor {
		return l.headDim
	}
	return l.kvDim()
}

// Base is the element offset of (slot, kvHead) at position zero. A kernel is
// handed cache[Base(...):] and advances by Stride, which is how both layouts
// reach the same kernel unchanged.
func (l kvLayout) Base(slot, kvHead int) int {
	s := slot * l.maxSeq * l.kvDim()
	if l.headMajor {
		s += kvHead * l.maxSeq * l.headDim
	} else {
		s += kvHead * l.headDim
	}
	return l.slots(s)
}

// At is one (slot, position, kvHead) vector, for the scalar fallbacks.
func (l kvLayout) At(slot, pos, kvHead int) int {
	return l.Base(slot, kvHead) + l.slots(pos*l.Stride())
}

// Write stores one position's whole kvDim-wide vector, narrowing to float32.
//
// Row-major writes a position's heads in one copy; head-major writes nKV
// separate copies, the layout's standing cost.
func (l kvLayout) Write(dst []float32, slot, pos int, src []float32) {
	if !l.headMajor {
		l.store(dst[l.slots(slot*l.maxSeq*l.kvDim()+pos*l.kvDim()):], src)
		return
	}
	for h := 0; h < l.nKV; h++ {
		l.store(dst[l.At(slot, pos, h):], src[h*l.headDim:(h+1)*l.headDim])
	}
}

// store copies one run of elements at the cache's own width.
//
// At elem == 2 this is the only place f32 becomes f16 in the engine, so an f16
// cache must match an f32 cache rounded through binary16 bit for bit.
// TestKVF16IsSelectedAndRuns holds the f16 cache end to end: that it is
// selected, and that its logits stay in the band rounding alone costs.
func (l kvLayout) store(dst []float32, src []float32) {
	if l.elem == 4 {
		narrow(dst, src)
		return
	}
	h := unsafe.Slice((*uint16)(unsafe.Pointer(&dst[0])), len(src))
	for i, v := range src {
		h[i] = quant.EncodeHalf(v)
	}
}

// kvlAt is layer li's layout: the State's own at the layer's geometry, which
// differs from kvl only where the sliding layers attend at their own head
// width and kv heads (Config.GeomSplit). MLA's one cached row is the same on
// every layer.
func (s *State) kvlAt(li int) kvLayout {
	c := s.c
	if c.MSAAt(li) {
		return s.kvlMSA()
	}
	if c.DSV4() {
		// One row a position, as wide as the block's (ds4.go).
		l := s.kvl
		l.nKV, l.headDim = 1, c.KVRowAt(li)
		return l
	}
	if !c.GeomSplit() || c.MLA() || !c.SWA(li) {
		return s.kvl
	}
	return s.kvlSWA()
}

// kvlMSA is an MSA block's layout: the indexer's key is one more kv head
// after the attention's, so the row is (NKVHead+1)*HeadDim (msa.go).
func (s *State) kvlMSA() kvLayout {
	l := s.kvl
	l.nKV++
	return l
}

// kvlSWA is the sliding layers' layout.
func (s *State) kvlSWA() kvLayout {
	l := s.kvl
	l.nKV, l.headDim = s.c.NKVHeadSWA, s.c.HeadDimSWA
	return l
}

// provisionAttn picks the attention kernel sets after the cache width is
// decided: the JIT's default (the global layers' geometry, AddAttnKV) for
// every layer, and a second set baked at the sliding layers' head width and
// stride where the two geometries differ.
func (s *State) provisionAttn() {
	s.attnG = s.jit.Attn()
	s.attnL = s.attnG
	if c := s.c; c.GeomSplit() {
		s.attnL = s.jit.AttnSetFor(c.HeadDimSWA, c.HeadDimSWA, s.kvlSWA().Stride(), s.kvF16)
	}
	// The indexer's scores read each cached row's IdxHeadDim tail at the row's
	// stride (indexer.go).
	if c := s.m.Cfg; c.Indexer() {
		s.idxAttn = s.jit.AttnSetFor(c.IdxHeadDim, c.IdxHeadDim, s.kvl.Stride(), s.kvF16)
	}
	// An MSA block's row carries the indexer's key as one more head, so its
	// stride is a head wider; its indexer scores are this set's too, the key
	// being a head of the same width (msa.go).
	s.attnM = s.attnG
	if c := s.m.Cfg; c.MSA() {
		s.attnM = s.jit.AttnSetFor(c.HeadDim, c.HeadDim, s.kvlMSA().Stride(), s.kvF16)
	}
}

// attnAt is the kernel set layer li attends with.
func (s *State) attnAt(li int) *nn.AttnSet {
	if s.c.MSAAt(li) {
		return s.attnM
	}
	if s.c.SWA(li) {
		return s.attnL
	}
	return s.attnG
}
