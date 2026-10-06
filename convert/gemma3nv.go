package convert

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"

	"github.com/samyfodil/jitllm/convert/meta"
	"github.com/samyfodil/jitllm/format/jlm"
)

// Gemma 3n's MobileNet-V5 tower (jlm.ProjGemma3nV), converter half.
//
// llama.cpp's mmproj names a block by its stage and its place in the stage,
// "v.blk.S.I.*"; the container's blocks are one index, stage by stage
// (towerNames), each carrying its place in a RoleVMNGeom vector. Every
// convolution weight is laid out tap-major (ky, kx, c), which is the order
// the engine's im2col and depthwise rows read: a full or pointwise one as a
// matrix of k = K*K*Cin, a depthwise filter as K*K rows of channels. A block's
// layer scale is folded into the norm or the matrix it follows.

func init() {
	for name, r := range map[string]jlm.Role{
		"v.conv_stem.conv.weight":        jlm.RoleVMNStem,
		"v.conv_stem.conv.bias":          jlm.RoleVMNStemBias,
		"v.conv_stem.bn.weight":          jlm.RoleVMNStemNorm,
		"v.conv_exp.weight":              jlm.RoleVMNConvExp,
		"v.bn1.weight":                   jlm.RoleVMNNorm1,
		"v.conv_pwl.weight":              jlm.RoleVMNConvPwl,
		"v.bn2.weight":                   jlm.RoleVMNNorm2,
		"v.dw_start.conv.weight":         jlm.RoleVMNDwStart,
		"v.dw_start.bn.weight":           jlm.RoleVMNDwStartNorm,
		"v.pw_exp.conv.weight":           jlm.RoleVMNPwExp,
		"v.pw_exp.bn.weight":             jlm.RoleVMNPwExpNorm,
		"v.dw_mid.conv.weight":           jlm.RoleVMNDwMid,
		"v.dw_mid.bn.weight":             jlm.RoleVMNDwMidNorm,
		"v.pw_proj.conv.weight":          jlm.RoleVMNPwProj,
		"v.pw_proj.bn.weight":            jlm.RoleVMNPwProjNorm,
		"v.layer_scale.gamma":            jlm.RoleVMNLayerScale,
		"v.norm.weight":                  jlm.RoleVAttnNorm,
		"v.attn.query.proj.weight":       jlm.RoleVAttnQ,
		"v.attn.key.proj.weight":         jlm.RoleVAttnK,
		"v.attn.value.proj.weight":       jlm.RoleVAttnV,
		"v.attn.output.proj.weight":      jlm.RoleVAttnOut,
		"v.attn.key.down_conv.weight":    jlm.RoleVMNKDown,
		"v.attn.key.norm.weight":         jlm.RoleVMNKDownNorm,
		"v.attn.value.down_conv.weight":  jlm.RoleVMNVDown,
		"v.attn.value.norm.weight":       jlm.RoleVMNVDownNorm,
		"v.msfa.ffn.pw_exp.conv.weight":  jlm.RoleVMNFusionExp,
		"v.msfa.ffn.pw_exp.bn.weight":    jlm.RoleVMNFusionExpNorm,
		"v.msfa.ffn.pw_proj.conv.weight": jlm.RoleVMNFusionProj,
		"v.msfa.ffn.pw_proj.bn.weight":   jlm.RoleVMNFusionProjNorm,
		"v.msfa.norm.weight":             jlm.RoleVMNFusionNorm,
		"mm.embedding.weight":            jlm.RoleVMNHardTable,
		"mm.hard_emb_norm.weight":        jlm.RoleVMNHardNorm,
	} {
		if was, ok := roleOf[name]; ok {
			panic(fmt.Sprintf("convert: %q is both %v and %v", name, was, r))
		}
		roleOf[name] = r
	}
	for _, r := range []jlm.Role{jlm.RoleVMNStem, jlm.RoleVMNConvExp, jlm.RoleVMNConvPwl, jlm.RoleVMNPwExp,
		jlm.RoleVMNPwProj, jlm.RoleVMNFusionExp, jlm.RoleVMNFusionProj} {
		towerMatrix[r] = true
	}
}

// mobilenetPlace parses "v.blk.S.I.rest", a MobileNet-V5 block tensor.
func mobilenetPlace(name string) (stage, idx int, rest string, ok bool) {
	r, found := strings.CutPrefix(name, "v.blk.")
	if !found {
		return 0, 0, "", false
	}
	f := strings.SplitN(r, ".", 3)
	if len(f) != 3 {
		return 0, 0, "", false
	}
	s, e1 := strconv.Atoi(f[0])
	i, e2 := strconv.Atoi(f[1])
	if e1 != nil || e2 != nil || s < 0 || i < 0 {
		return 0, 0, "", false
	}
	return s, i, f[2], true
}

// mobilenetPlaces is every block a Gemma 3n mmproj carries, in order: by
// stage, then by place in the stage.
func mobilenetPlaces(f *meta.File) [][2]int {
	seen := map[[2]int]bool{}
	var out [][2]int
	for i := range f.Tensors {
		s, b, _, ok := mobilenetPlace(f.Tensors[i].Name)
		if !ok || seen[[2]int{s, b}] {
			continue
		}
		seen[[2]int{s, b}] = true
		out = append(out, [2]int{s, b})
	}
	sort.Slice(out, func(a, b int) bool {
		if out[a][0] != out[b][0] {
			return out[a][0] < out[b][0]
		}
		return out[a][1] < out[b][1]
	})
	return out
}

// towerNames is the renames a tower file's tensors take before the name
// table: Gemma 3n's double-indexed blocks become one index. nil for every
// other tower. sourceOf and the role gate both read it.
func towerNames(f *meta.File, v *jlm.Vision) map[string]string {
	if v == nil || v.Projector != jlm.ProjGemma3nV {
		return nil
	}
	flat := map[[2]int]int{}
	for i, p := range mobilenetPlaces(f) {
		flat[p] = i
	}
	out := map[string]string{}
	for i := range f.Tensors {
		n := f.Tensors[i].Name
		if s, b, rest, ok := mobilenetPlace(n); ok {
			out[n] = "v.blk." + strconv.Itoa(flat[[2]int{s, b}]) + "." + rest
		}
	}
	return out
}

// mobilenetGeom appends each block's place, {stage, index in the stage}.
func mobilenetGeom(s *jlm.Source, places [][2]int) {
	for i, p := range places {
		e := jlm.Tensor{Role: jlm.RoleVMNGeom, Block: int32(i), Index: -1,
			Name: "v.blk." + strconv.Itoa(i) + ".mn_geom"}
		setF32Dims(&e, []float32{float32(p[0]), float32(p[1])}, 2)
		s.Tensors = append(s.Tensors, e)
	}
}

// gemma3nVocabOffset is the first of Gemma 3n's hard vision token ids
// (Gemma3nVisionConfig.vocab_offset): its end-of-image marker and image
// placeholder, which transformers embeds through the vision embedder rather
// than the text table. The mmproj does not state it; the architecture does.
const gemma3nVocabOffset = 262144

// prepareGemma3nV lays every convolution out tap-major, folds the layer
// scales and computes the hard tokens' embeddings, before quantizeTower rounds
// the matrices.
func prepareGemma3nV(s *jlm.Source) error {
	if err := gemma3nHardEmbd(s); err != nil {
		return err
	}
	scales := map[int32][]float32{}
	kept := s.Tensors[:0]
	for i := range s.Tensors {
		t := s.Tensors[i]
		if t.Role == jlm.RoleVMNLayerScale {
			x, err := vecOf(&t)
			if err != nil {
				return err
			}
			scales[t.Block] = x
			continue
		}
		kept = append(kept, t)
	}
	s.Tensors = kept
	for i := range s.Tensors {
		t := &s.Tensors[i]
		var err error
		switch t.Role {
		case jlm.RoleVMNStem, jlm.RoleVMNConvExp, jlm.RoleVMNConvPwl, jlm.RoleVMNPwExp, jlm.RoleVMNPwProj,
			jlm.RoleVMNFusionExp, jlm.RoleVMNFusionProj, jlm.RoleVAttnQ, jlm.RoleVAttnK, jlm.RoleVAttnV,
			jlm.RoleVAttnOut:
			err = convMatrix(t)
		case jlm.RoleVMNDwStart, jlm.RoleVMNDwMid, jlm.RoleVMNKDown, jlm.RoleVMNVDown:
			err = depthwiseTaps(t)
		case jlm.RoleVMNStemBias:
			var x []float32
			if x, err = vecOf(t); err == nil {
				setF32Dims(t, x, uint64(len(x)))
			}
		}
		if err != nil {
			return err
		}
	}
	// A layer scale multiplies its block's output channels: the projection's
	// norm in an inverted residual, the output matrix's rows in an attention
	// block. Both are per output channel, so the fold is exact.
	for b, g := range scales {
		folded := false
		for i := range s.Tensors {
			t := &s.Tensors[i]
			if t.Block != b {
				continue
			}
			switch t.Role {
			case jlm.RoleVMNPwProjNorm:
				x, err := vecOf(t)
				if err != nil {
					return err
				}
				if len(x) != len(g) {
					return fmt.Errorf("convert: block %d's layer scale is %d and its norm %d", b, len(g), len(x))
				}
				for j := range x {
					x[j] *= g[j]
				}
				setF32Dims(t, x, uint64(len(x)))
				folded = true
			case jlm.RoleVAttnOut:
				vals, k, rows, err := floatRows(t)
				if err != nil {
					return err
				}
				if rows != len(g) {
					return fmt.Errorf("convert: block %d's layer scale is %d and its output %d rows", b, len(g), rows)
				}
				for r := 0; r < rows; r++ {
					for j := 0; j < k; j++ {
						vals[r*k+j] *= g[r]
					}
				}
				setF32Dims(t, vals, uint64(k), uint64(rows))
				folded = true
			}
		}
		if !folded {
			return fmt.Errorf("convert: block %d has a layer scale and nothing to fold it into", b)
		}
	}
	return nil
}

// convMatrix turns a [kw kh Cin Cout] convolution (torch's [Cout][Cin][kh][kw])
// into the matrix the engine multiplies by: Cout rows of k = kh*kw*Cin in
// (ky, kx, c) order, padded to a whole quantization block. A 2-D tensor is
// already one (a linear's).
func convMatrix(t *jlm.Tensor) error {
	if t.NDim == 2 {
		return padK(t)
	}
	if t.NDim != 4 {
		return fmt.Errorf("convert: %s is %d-dimensional, want a convolution", t.Name, t.NDim)
	}
	kw, kh, cin, cout := int(t.Dims[0]), int(t.Dims[1]), int(t.Dims[2]), int(t.Dims[3])
	src, err := vecOf(t)
	if err != nil {
		return err
	}
	k := kh * kw * cin
	out := make([]float32, k*cout)
	for o := 0; o < cout; o++ {
		for c := 0; c < cin; c++ {
			for y := 0; y < kh; y++ {
				for x := 0; x < kw; x++ {
					out[o*k+(y*kw+x)*cin+c] = src[((o*cin+c)*kh+y)*kw+x]
				}
			}
		}
	}
	setF32Dims(t, out, uint64(k), uint64(cout))
	return padK(t)
}

// padK pads a matrix's k to a whole quantization block with zero columns.
func padK(t *jlm.Tensor) error {
	k, rows := int(t.Dims[0]), int(t.Dims[1])
	if k%q8Elems == 0 {
		return nil
	}
	vals, _, _, err := floatRows(t)
	if err != nil {
		return err
	}
	padded := (k + q8Elems - 1) / q8Elems * q8Elems
	out := make([]float32, padded*rows)
	for r := 0; r < rows; r++ {
		copy(out[r*padded:], vals[r*k:(r+1)*k])
	}
	setF32Dims(t, out, uint64(padded), uint64(rows))
	return nil
}

// depthwiseTaps turns a [kw kh 1 C] depthwise filter (torch's [C][1][kh][kw])
// into K*K rows of C channels, tap-major.
func depthwiseTaps(t *jlm.Tensor) error {
	if t.NDim != 4 || t.Dims[2] != 1 || t.Dims[0] != t.Dims[1] {
		return fmt.Errorf("convert: %s is %v, want a square depthwise filter [K K 1 C]", t.Name, t.Dims[:t.NDim])
	}
	kk, c := int(t.Dims[0]*t.Dims[1]), int(t.Dims[3])
	src, err := vecOf(t)
	if err != nil {
		return err
	}
	out := make([]float32, kk*c)
	for ch := 0; ch < c; ch++ {
		for tap := 0; tap < kk; tap++ {
			out[tap*c+ch] = src[ch*kk+tap]
		}
	}
	setF32Dims(t, out, uint64(c), uint64(kk))
	return nil
}

// gemma3nHardEmbd folds the embedder's hard-token path into one table:
// transformers' Gemma3nMultimodalEmbedder for an id, post_norm(W · norm(
// table[id - offset])), with W the projection the picture's rows go through
// and post_norm unweighted. Offline arithmetic in float64, so the reader does
// none of it.
func gemma3nHardEmbd(s *jlm.Source) error {
	var tab, nrm, proj *jlm.Tensor
	kept := s.Tensors[:0]
	for i := range s.Tensors {
		t := &s.Tensors[i]
		switch t.Role {
		case jlm.RoleVMNHardTable:
			x := *t
			tab = &x
			continue
		case jlm.RoleVMNHardNorm:
			x := *t
			nrm = &x
			continue
		case jlm.RoleVProj:
			x := *t
			proj = &x
		}
		kept = append(kept, *t)
	}
	s.Tensors = kept
	if tab == nil && nrm == nil {
		return nil
	}
	if tab == nil || nrm == nil || proj == nil {
		return fmt.Errorf("convert: gemma3nv's hard-token embedder is missing its table, its norm or its projection")
	}
	tv, err := vecOf(tab)
	if err != nil {
		return err
	}
	nv, err := vecOf(nrm)
	if err != nil {
		return err
	}
	pv, k, rows, err := floatRows(proj)
	if err != nil {
		return err
	}
	if len(nv) != k || len(tv)%k != 0 {
		return fmt.Errorf("convert: gemma3nv's hard table is %d floats and its norm %d, for a %d-wide projection",
			len(tv), len(nv), k)
	}
	eps := float64(s.Vision.Eps)
	n := len(tv) / k
	out := make([]float32, n*rows)
	x := make([]float64, k)
	y := make([]float64, rows)
	for i := 0; i < n; i++ {
		row := tv[i*k : (i+1)*k]
		ss := 0.0
		for j, v := range row {
			ss += float64(v) * float64(v)
			x[j] = float64(v)
		}
		inv := 1 / math.Sqrt(ss/float64(k)+eps)
		for j := range x {
			x[j] *= inv * float64(nv[j])
		}
		ss = 0
		for r := 0; r < rows; r++ {
			acc := 0.0
			w := pv[r*k : (r+1)*k]
			for j := range x {
				acc += float64(w[j]) * x[j]
			}
			y[r] = acc
			ss += acc * acc
		}
		inv = 1 / math.Sqrt(ss/float64(rows)+eps)
		for r := range y {
			out[i*rows+r] = float32(y[r] * inv)
		}
	}
	e := jlm.Tensor{Role: jlm.RoleVMNHardEmbd, Block: jlm.DenseBlock, Index: -1, Name: "mm.hard_embd"}
	setF32Dims(&e, out, uint64(rows), uint64(n))
	o := jlm.Tensor{Role: jlm.RoleVMNHardOffset, Block: jlm.DenseBlock, Index: -1, Name: "mm.hard_offset"}
	setF32Dims(&o, []float32{gemma3nVocabOffset}, 1)
	s.Tensors = append(s.Tensors, e, o)
	return nil
}
