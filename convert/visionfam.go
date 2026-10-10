package convert

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/jitllm/jitllm/format/jlm"
)

// The converter's half of the vision families whose codes sit at the top of
// the code space (format/jlm/visionfam.go): the names their towers' tensors
// carry, and the transforms they need, each exact.

func init() {
	for name, r := range map[string]jlm.Role{
		// Gemma 4's tower block.
		"v.attn_post_norm.weight": jlm.RoleVPostAttnNorm,
		"v.ffn_post_norm.weight":  jlm.RoleVPostFFNNorm,
		"v.attn_q_norm.weight":    jlm.RoleVAttnQNorm,
		"v.attn_k_norm.weight":    jlm.RoleVAttnKNorm,
		"v.std_bias":              jlm.RoleVStdBias,
		"v.std_scale":             jlm.RoleVStdScale,
	} {
		if was, ok := roleOf[name]; ok {
			panic(fmt.Sprintf("convert: %q is both %v and %v", name, was, r))
		}
		roleOf[name] = r
	}
}

// clampSlot is where a clipped linear's bounds go in a block's RoleVClamp
// vector: its matrix's place, times four.
var clampSlot = map[string]int{
	"attn_q": 0, "attn_k": 1, "attn_v": 2, "attn_out": 3,
	"ffn_gate": 4, "ffn_up": 5, "ffn_down": 6,
}

// clampWhich is a bound's place among a matrix's four.
var clampWhich = map[string]int{"input_min": 0, "input_max": 1, "output_min": 2, "output_max": 3}

// clampOf parses one of Gemma 4's clipped-linear bounds,
// "v.blk.N.<matrix>.<input|output>_<min|max>": a scalar with no role of its
// own, gathered into the block's RoleVClamp vector (clampTensors).
func clampOf(name string) (block int32, at int, ok bool) {
	rest, found := strings.CutPrefix(name, "v.blk.")
	if !found {
		return 0, 0, false
	}
	f := strings.Split(rest, ".")
	if len(f) != 3 {
		return 0, 0, false
	}
	n, err := strconv.Atoi(f[0])
	s, okS := clampSlot[f[1]]
	w, okW := clampWhich[f[2]]
	if err != nil || n < 0 || !okS || !okW {
		return 0, 0, false
	}
	return int32(n), 4*s + w, true
}

// clampTensors writes each block's gathered bounds as its RoleVClamp vector;
// a bound the file did not carry is infinite, which clamps nothing.
func clampTensors(s *jlm.Source, clamps map[int32]*[28]float32) error {
	for b, v := range clamps {
		if b < 0 || uint32(b) >= s.Vision.NLayer {
			return fmt.Errorf("convert: clipped-linear bounds for block %d of %d", b, s.Vision.NLayer)
		}
		e := jlm.Tensor{Role: jlm.RoleVClamp, Block: b, Index: -1, Name: fmt.Sprintf("v.blk.%d.clamp", b)}
		setF32Dims(&e, v[:], 28)
		s.Tensors = append(s.Tensors, e)
	}
	return nil
}

// newClamps is a block's bounds before the file has stated any: all
// infinite.
func newClamps() *[28]float32 {
	var c [28]float32
	for i := range c {
		c[i] = float32(math.Inf(1))
		if i%2 == 0 {
			c[i] = float32(math.Inf(-1))
		}
	}
	return &c
}

// halfNeoxToNeox lays Gemma 4's per-half rotary out as one NEOX rotation
// over the whole head. The reference turns each half of a head on its own,
// NEOX-paired inside the half -- (i, i + d/4) in the first, (d/2 + i,
// 3d/4 + i) in the second -- where a whole-head NEOX table pairs (j, j + d/2).
// Swapping the head's second and third quarters makes the one the other, and
// since q and k are permuted alike every score is unchanged; their norms'
// weights move with them. The rotary tables then are Qwen-VL's (two runs of
// d/4 pairs, the column's then the row's).
func halfNeoxToNeox(s *jlm.Source) error {
	v := s.Vision
	hd := int(v.NEmbd / v.NHead)
	if hd%4 != 0 {
		return fmt.Errorf("convert: a head of %d does not split into quarters", hd)
	}
	q := hd / 4
	perm := func(x []float32) {
		for h := 0; h+hd <= len(x); h += hd {
			for i := 0; i < q; i++ {
				x[h+q+i], x[h+2*q+i] = x[h+2*q+i], x[h+q+i]
			}
		}
	}
	for i := range s.Tensors {
		t := &s.Tensors[i]
		switch t.Role {
		case jlm.RoleVAttnQ, jlm.RoleVAttnK:
			vals, k, rows, err := floatRows(t)
			if err != nil {
				return err
			}
			if rows%hd != 0 {
				return fmt.Errorf("convert: %s has %d rows, not whole heads of %d", t.Name, rows, hd)
			}
			// Rows are the head's dimensions: permute whole rows.
			out := make([]float32, len(vals))
			for r := 0; r < rows; r++ {
				h, d := r/hd, r%hd
				src := d
				switch {
				case d >= q && d < 2*q:
					src = d + q
				case d >= 2*q && d < 3*q:
					src = d - q
				}
				copy(out[r*k:(r+1)*k], vals[(h*hd+src)*k:(h*hd+src+1)*k])
			}
			setF32Dims(t, out, uint64(k), uint64(rows))
		case jlm.RoleVAttnQBias, jlm.RoleVAttnKBias, jlm.RoleVAttnQNorm, jlm.RoleVAttnKNorm:
			x, err := vecOf(t)
			if err != nil {
				return err
			}
			if len(x)%hd != 0 {
				return fmt.Errorf("convert: %s is %d floats, not whole heads of %d", t.Name, len(x), hd)
			}
			perm(x)
			setF32Dims(t, x, uint64(len(x)))
		}
	}
	return nil
}
