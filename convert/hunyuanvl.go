package convert

import (
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
)

// hunyuanvlRole is HunyuanVL's merger as llama.cpp's converter names it
// (jlm.ProjHunyuanVL): before_rms, the 2x2 convolution (proj.0), the 1x1 one
// (proj.2), the linear (mlp), after_rms and the three learned rows.
var hunyuanvlRole = map[string]jlm.Role{
	"mm.pre_norm.weight":  jlm.RoleVProjInNorm,
	"mm.0.weight":         jlm.RoleVMergeConv,
	"mm.0.bias":           jlm.RoleVMergeConvBias,
	"mm.2.weight":         jlm.RoleVProj,
	"mm.2.bias":           jlm.RoleVProjBias,
	"mm.model.fc.weight":  jlm.RoleVProj2,
	"mm.model.fc.bias":    jlm.RoleVProj2Bias,
	"mm.post_norm.weight": jlm.RoleVProjNorm,
	"v.image_newline":     jlm.RoleVImgNewline,
	"mm.image_begin":      jlm.RoleVImgBegin,
	"mm.image_end":        jlm.RoleVImgEnd,
}

// hunyuanvlUnread is the one merger tensor the reference never reads:
// HunYuanVLVisionPatchMerger declares image_sep and its forward writes only
// the begin and end rows around a picture.
const hunyuanvlUnread = "v.view_seperator"

// reshapePointConv lays a 1x1 convolution [1 1 in out] out as the matrix
// [k=in, rows=out] it is: the bytes already are, a 1x1 kernel having no
// spatial axis to move.
func reshapePointConv(t *jlm.Tensor) error {
	if t.NDim == 2 {
		return nil
	}
	if t.NDim != 4 || t.Dims[0] != 1 || t.Dims[1] != 1 {
		return fmt.Errorf("convert: %s is %v, want a [1 1 in out] convolution", t.Name, t.Dims[:t.NDim])
	}
	in, out := t.Dims[2], t.Dims[3]
	t.NDim = 2
	t.Dims = [4]uint64{in, out}
	return nil
}
