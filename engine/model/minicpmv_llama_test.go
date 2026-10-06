package model

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// mcvNode is one node of a llama-mtmd-debug dump (scripts/mtmddump.py): the
// first and last three rows, three leading and three trailing values of each,
// and the whole tensor's sum.
type mcvNode struct {
	Shape []int
	Rows  [][]float64
	Sum   float64
}

func readMtmdNodes(t testing.TB, name string) map[string]mcvNode {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("..", "..", "testdata", "golden", name))
	if err != nil {
		t.Fatalf("GOLDEN MISSING: %v", err)
	}
	var m map[string]mcvNode
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	return m
}

// compareMtmdNode holds a [rows][width] tensor to its dump: relative RMS over
// the 36 printed values, and the sum relative to the tensor's own scale (a sum
// near zero cancels, so it is weighed against sum |x|).
func compareMtmdNode(t testing.TB, name string, x []float32, width int, ref mcvNode) (rel, sumRel float64) {
	t.Helper()
	rows := len(x) / width
	if len(ref.Rows) != 6 || ref.Shape[0] != width || ref.Shape[1] != rows {
		t.Fatalf("%s: the dump is %v with %d printed rows; jitllm has %d rows of %d", name, ref.Shape,
			len(ref.Rows), rows, width)
	}
	at := []int{0, 1, 2, rows - 3, rows - 2, rows - 1}
	var sse, sy2 float64
	for i, r := range at {
		for j := 0; j < 6; j++ {
			col := j
			if j >= 3 {
				col = width - 6 + j
			}
			got, want := float64(x[r*width+col]), ref.Rows[i][j]
			sse += (got - want) * (got - want)
			sy2 += want * want
		}
	}
	var sum, abs float64
	for _, v := range x {
		sum += float64(v)
		abs += math.Abs(float64(v))
	}
	rel = math.Sqrt(sse / sy2)
	sumRel = math.Abs(sum-ref.Sum) / abs
	if math.IsNaN(rel) || math.IsNaN(sumRel) || sy2 == 0 {
		t.Fatalf("%s: a degenerate comparison (rel %v, sum %v)", name, rel, sumRel)
	}
	return rel, sumRel
}

// TestMiniCPMVTowerMatchesLlamaCpp runs llama-mtmd-debug's own input -- its
// 448 checkerboard, raw, no normalisation -- through the tower and resampler
// and holds them node by node to b10825's dump:
//
//	llama-mtmd-debug -m MiniCPM-V-4-Q4_K_M.gguf --mmproj mmproj-MiniCPM-V-4-f16.gguf \
//	  --image cb -n 448 -p encode
//	python3 scripts/mtmddump.py DUMP testdata/golden/minicpmv4-mtmd-cb448.json \
//	  pos_embed layer_out-0 layer_out-13 layer_out-26 resampler_attn_out node_894
//
// A checkerboard of a period dividing the patch makes every patch the same,
// but the bucketed position table makes every row differ before block 0, so
// this input is not degenerate for this tower (the Qwen2-VL lesson is about a
// tower whose positions only rotate q and k). The bar is percent, not ulp:
// jitllm quantizes activations to int8 per matmul and llama.cpp runs the f16
// weights in f32 with f16 flash-attention K/V.
func TestMiniCPMVTowerMatchesLlamaCpp(t *testing.T) {
	m := openMiniCPMV(t)
	defer m.Close()
	tw := m.Tower()
	c := tw.Cfg
	ref := readMtmdNodes(t, "minicpmv4-mtmd-cb448.json")
	s := tw.testState()
	defer s.Close()
	got := map[string][]float32{}
	s.vis.afterBlock = func(li int, x []float32) {
		switch li {
		case -1:
			got["pos_embed"] = append([]float32(nil), x...)
		case 0:
			got["layer_out-0"] = append([]float32(nil), x...)
		case 13:
			got["layer_out-13"] = append([]float32(nil), x...)
		case 26:
			got["layer_out-26"] = append([]float32(nil), x...)
		}
	}
	out, err := s.EncodeGrid(towerCb(448), 32, 32)
	if err != nil {
		t.Fatal(err)
	}
	got["node_894"] = append([]float32(nil), out...)
	for _, n := range []struct {
		name  string
		width int
		bar   float64
	}{
		// The bars are the int8 activation's on a tower with massive
		// activations (see TestMiniCPMVMatchesTransformers): llama.cpp's f16
		// run is within 0.5% of transformers' float32 at block 26
		// (testdata/golden/minicpmv4-hf-cb448.json) and this engine is not.
		{"pos_embed", c.NEmbd, 0.01}, {"layer_out-0", c.NEmbd, 0.03},
		{"layer_out-13", c.NEmbd, 0.08}, {"layer_out-26", c.NEmbd, 0.25}, {"node_894", c.ProjDim, 0.08},
	} {
		x := got[n.name]
		if x == nil {
			t.Fatalf("%s was not captured", n.name)
		}
		rel, sr := compareMtmdNode(t, n.name, x, n.width, ref[n.name])
		t.Logf("%-13s relative RMS %.4f over 36 values, sum off by %.4f of sum|x|", n.name, rel, sr)
		if rel > n.bar || sr > n.bar {
			t.Errorf("%s: relative RMS %.4f, sum %.4f (bar %.2f)", n.name, rel, sr, n.bar)
		}
	}
}
