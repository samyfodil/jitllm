package model

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/samyfodil/jitllm/format/jlm"
	"github.com/samyfodil/jitllm/internal/testmodels"
)

// Laya against its own code (rl_common.py, rl_agent_api.py), the goldens
// written by scripts/layagold.py: the sequence each question is asked as,
// token for token, and the scorer's logits at the markers.
//
//	laya-fixture  a ModernBERT and head built from Laya's classes with every
//	              parameter and buffer random, converted by llama.cpp's own
//	              convert_hf_to_gguf.py (layagold.py fixture)
//	laya          the published checkpoint, ggml-org/Laya-GGUF BF16

type layaGold struct {
	Answers map[string]json.RawMessage `json:"answers"`
	Logits  map[string][]float64       `json:"logits"`
	IDs     map[string][]int32         `json:"ids"`
	Markers map[string][]int           `json:"markers"`
}

var layaCases = []struct {
	name, gguf, gold string
	tol              float64 // on a raw logit
}{
	// f32 weights: what is left is f32 reduction order (worst 1e-6).
	{"fixture", "laya/fixture/laya-fixture.gguf", "laya-fixture", 2e-5},
	// BF16 weights read as the reference reads them (exactly); 28 blocks of
	// f32 order (worst 0.027).
	{"laya", "laya/gguf/Laya-BF16.gguf", "laya", 0.035},
}

func TestLayaMatchesReference(t *testing.T) {
	for _, c := range layaCases {
		t.Run(c.name, func(t *testing.T) {
			m := openDecision(t, c.gguf, jlm.DecisionLaya)
			defer m.Close()
			d, err := m.NewDecider(0)
			if err != nil {
				t.Fatal(err)
			}
			defer d.Close()
			for _, req := range decisionRequests {
				t.Run(req, func(t *testing.T) {
					var g layaGold
					b, err := os.ReadFile(filepath.Join("testdata", "decision", c.gold+"."+req+".reference.json"))
					if err != nil {
						testmodels.Missing(t, "%v (scripts/layagold.py writes it)", err)
					}
					if err := json.Unmarshal(b, &g); err != nil {
						t.Fatal(err)
					}
					state, qs := decisionRequest(t, filepath.Join("testdata", "decision", req+".json"))
					worst := 0.0
					for i := range qs {
						q := &qs[i]
						ids, markers, err := d.layaSequence(state, q)
						if err != nil {
							t.Fatal(err)
						}
						if !equalIDs(ids, g.IDs[q.ID]) || !equalInts(markers, g.Markers[q.ID]) {
							t.Fatalf("%s: the sequence differs from the reference's\n got %v %v\nwant %v %v",
								q.ID, ids, markers, g.IDs[q.ID], g.Markers[q.ID])
						}
						s, err := d.layaRun(state, q)
						if err != nil {
							t.Fatal(err)
						}
						want := g.Logits[q.ID]
						if len(s) != len(want) {
							t.Fatalf("%s: %d scores, the reference %d", q.ID, len(s), len(want))
						}
						for j := range s {
							if math.IsNaN(float64(s[j])) || math.IsInf(float64(s[j]), 0) {
								t.Fatalf("%s: score %d is %v", q.ID, j, s[j])
							}
							worst = max(worst, math.Abs(float64(s[j])-want[j]))
							if math.Abs(float64(s[j])-want[j]) > c.tol {
								t.Errorf("%s option %d: logit %.5f, the reference %.5f", q.ID, j, s[j], want[j])
							}
						}
					}
					t.Logf("worst logit difference from the reference: %.6f", worst)
					if d.emb.jit.FloatMatMulCalls() == 0 {
						t.Errorf("the encoder's float matrices never ran the float GEMM")
					}
				})
			}
		})
	}
}

func equalIDs(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestLayaFeaturesAreLoadBearing removes each feature of the graph from the
// fixture and requires the logits to leave the reference's by ten times the
// gate's tolerance: a feature the gate cannot see is not gated (RULE 10).
func TestLayaFeaturesAreLoadBearing(t *testing.T) {
	m := openDecision(t, "laya/fixture/laya-fixture.gguf", jlm.DecisionLaya)
	defer m.Close()
	var g layaGold
	b, err := os.ReadFile(filepath.Join("testdata", "decision", "laya-fixture.support.reference.json"))
	if err != nil {
		testmodels.Missing(t, "%v (scripts/layagold.py writes it)", err)
	}
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatal(err)
	}
	state, qs := decisionRequest(t, filepath.Join("testdata", "decision", "support.json"))
	for _, v := range []string{"no-window", "one-base", "up-gated", "type-row-0", "encoder-heads", "tanh-gelu"} {
		t.Run(v, func(t *testing.T) {
			d, err := m.NewDecider(0)
			if err != nil {
				t.Fatal(err)
			}
			defer d.Close()
			d.emb.violation = v
			worst := 0.0
			for i := range qs {
				q := &qs[i]
				if v == "type-row-0" && q.Type == jlm.QuestionChoice {
					continue // row 0 is the choice row: nothing removed
				}
				s, err := d.layaRun(state, q)
				if err != nil {
					t.Fatal(err)
				}
				for j := range s {
					worst = max(worst, math.Abs(float64(s[j])-g.Logits[q.ID][j]))
				}
			}
			t.Logf("%s moves a logit by %.4f", v, worst)
			if worst <= 10*layaCases[0].tol {
				t.Errorf("removing %s moved no logit past %.3f: the gate does not see it", v, 10*layaCases[0].tol)
			}
		})
	}
}

// TestLayaPagesWithTheSameAnswer: the encoder and its head under a budget of
// one block page evict and re-read every block a question passes through, and
// answer bit for bit as with every page resident (the Paging principle).
func TestLayaPagesWithTheSameAnswer(t *testing.T) {
	state, qs := decisionRequest(t, filepath.Join("testdata", "decision", "support.json"))
	run := func(frames int) ([][]float32, int64) {
		m := openDecision(t, "laya/fixture/laya-fixture.gguf", jlm.DecisionLaya)
		defer m.Close()
		if frames > 0 {
			m.SetPageBudget(uint64(frames) * m.PageSize())
		}
		d, err := m.NewDecider(0)
		if err != nil {
			t.Fatal(err)
		}
		defer d.Close()
		var out [][]float32
		for i := range qs {
			s, err := d.layaRun(state, &qs[i])
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, s)
		}
		return out, m.PagerReads()
	}
	whole, _ := run(0)
	paged, reads := run(1)
	blocks := int64(m0Blocks(t))
	if reads <= blocks {
		t.Fatalf("a one-page budget read %d pages for %d blocks: nothing was evicted, so nothing was tested", reads, blocks)
	}
	for i := range whole {
		for j := range whole[i] {
			if whole[i][j] != paged[i][j] {
				t.Errorf("%s option %d: %v resident, %v paged", qs[i].ID, j, whole[i][j], paged[i][j])
			}
		}
	}
	t.Logf("%d page reads for %d blocks over %d questions", reads, blocks, len(qs))
}

// m0Blocks is the fixture container's block page count.
func m0Blocks(t *testing.T) int {
	m := openDecision(t, "laya/fixture/laya-fixture.gguf", jlm.DecisionLaya)
	defer m.Close()
	return m.container.NPages()
}
