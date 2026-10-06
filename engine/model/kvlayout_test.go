package model

import (
	"math"
	"os"
	"testing"
)

// TestKVLayoutsAgree holds the head-major KV cache to the row-major one.
//
// Bit equality: the layouts hold the same numbers, and the kernels sum
// positions in the same order given a base and a stride. A wrong address does
// not fault; it attends over a permutation of the history and answers fluently.
// Decode, prefill and batch each derive their own addresses through kvLayout.
func TestKVLayoutsAgree(t *testing.T) {
	for name, path := range models {
		if name == "tinyllama" && os.Getenv("JITLLM_SLOW") == "" {
			continue
		}
		t.Run(name, func(t *testing.T) {
			if _, err := os.Stat(path); err != nil {
				t.Skipf("model not present: %s", path)
			}
			ids := []int32{7, 19, 33, 51, 64, 12, 88, 41, 5, 77}

			// One Model per layout: AddAttn bakes the stride into the attention
			// kernels, and the JIT is built once per model.
			run := func(head bool, mode string) []float32 {
				// A load option configures only this arm's model. noGEMM makes
				// prefill exact too.
				m, err := Open(jlmOf(t, path), noTune, noGEMM, WithKVHeadMajor(head))
				if err != nil {
					t.Fatal(err)
				}
				defer m.Close()
				for i := range ids {
					ids[i] %= int32(m.Cfg.NVocab)
				}
				st := m.NewState(len(ids) + 8)
				defer st.Close()
				var out []float32
				switch mode {
				case "decode":
					for _, id := range ids {
						if out, err = st.Forward(id); err != nil {
							t.Fatal(err)
						}
					}
				case "prefill":
					if out, err = st.Prefill(ids); err != nil {
						t.Fatal(err)
					}
				case "mixed":
					// Prefill part, decode the rest: the two writers must agree
					// with each other as well as across layouts.
					if _, err = st.Prefill(ids[:5]); err != nil {
						t.Fatal(err)
					}
					for _, id := range ids[5:] {
						if out, err = st.Forward(id); err != nil {
							t.Fatal(err)
						}
					}
				}
				return append([]float32(nil), out...)
			}
			for _, mode := range []string{"decode", "prefill", "mixed"} {
				row, head := run(false, mode), run(true, mode)
				if len(row) != len(head) {
					t.Fatalf("%s: %d logits vs %d", mode, len(row), len(head))
				}
				for i := range row {
					if math.Float32bits(row[i]) != math.Float32bits(head[i]) {
						t.Fatalf("%s: logit %d is %v head-major and %v row-major "+
							"-- the layouts hold the same numbers, so an address is wrong",
							mode, i, head[i], row[i])
					}
				}
				t.Logf("%s: bit-identical across both layouts", mode)
			}
		})
	}
}
