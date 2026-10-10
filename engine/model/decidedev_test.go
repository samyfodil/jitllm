package model

import (
	"fmt"
	"math"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// The decision readouts on a device (RULE 11c): each decision model's answers
// with every block placed, and with a seam that leaves blocks home, against
// the host's on the same bytes, on every device present. The device's split
// tuner is off (tier.TuneOff), so its reduction order is fixed for the run.
//
// decisionDevTol is how far a probability may move between the host and a
// device: the f32 reduction-order band of the label logits (0.2-0.9 of a
// logit on a 20-wide scale; RULE 11c) seen through the readout's softmax at a
// near-tie (d1 measured 0.008-0.025 on the laptop's CUDA and Vulkan). The
// violation arm must move an answer past it.
var decisionDevTol = map[string]float64{"d1-3B": 0.03, "lev": 0.05, "laya": 0.01}

// decisionBlocks is how many blocks a Decider's readout runs: the decoder's,
// or the encoder segment's (a decision head's blocks with it).
func decisionBlocks(m *Model) int {
	if m.IsEncoder() {
		return m.encBlocks()
	}
	return m.Cfg.NLayer
}

// decideOn answers both decision requests on a Decider placed with n blocks
// on dev (nil: the host), and reports the blocks it ran on the device. ok
// sees the placement before anything runs; false returns at once, no answers.
// violation is the Decider's (a readout feature a test removes); picked is
// how many prompts' labels the device picked (nn.Head.Pick).
func decideOn(t *testing.T, m *Model, dev nn.Device, n int, ok func(placed int) bool, violation ...string) ([][]DecisionAnswer, int, string, int) {
	t.Helper()
	d, err := m.NewDecider(4096)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	if len(violation) > 0 {
		d.violation = violation[0]
	}
	why := ""
	gpu, _ := dev.(*tier.GPU)
	if dev != nil {
		if err := d.SetDevice(dev, n); err != nil {
			t.Fatal(err)
		}
		if gpu != nil {
			why = gpu.Err()
		}
		if d.st != nil {
			why = fmt.Sprintf("%v %s", d.st.DeviceDeclines(), why)
		} else {
			why = fmt.Sprintf("%v %s", d.emb.DeviceDeclines(), why)
		}
	}
	placed := d.DeviceBlocks()
	if ok != nil && !ok(placed) {
		return nil, placed, why, 0
	}
	var s0 int
	if gpu != nil {
		s0 = gpu.Stats().Submits
	}
	var out [][]DecisionAnswer
	for _, req := range decisionRequests {
		state, qs := decisionRequest(t, filepath.Join("testdata", "decision", req+".json"))
		ans, err := d.Decide(state, qs)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, ans)
	}
	if got := d.DeviceBlocks(); got != placed {
		t.Fatalf("the readout fell from %d device blocks to %d while it ran", placed, got)
	}
	// The device ran them: at least one submission a prompt, counted by the
	// device itself.
	if gpu != nil && placed > 0 && gpu.Stats().Submits-s0 < len(decisionRequests) {
		t.Fatalf("%d submission(s) for %d requests: the placed blocks did not run on the device",
			gpu.Stats().Submits-s0, len(decisionRequests))
	}
	return out, placed, why, d.picked
}

// decisionDiff is the largest difference between two runs' probabilities,
// nouls and expected levels (a score over its k-1 levels, so every value is on
// [0, 1]), infinite on a value that is not finite. It logs where it is.
func decisionDiff(t *testing.T, want, got [][]DecisionAnswer) float64 {
	t.Helper()
	worst, where := 0.0, ""
	for r := range want {
		for i := range want[r] {
			a, b := want[r][i], got[r][i]
			type val struct {
				what string
				w, g float64
			}
			vals := []val{{"noul", a.Noul, b.Noul}}
			if k := len(a.Probs); k > 1 {
				vals = append(vals, val{"score/(k-1)", a.Score / float64(k-1), b.Score / float64(k-1)})
			}
			for j := range a.Probs {
				vals = append(vals, val{fmt.Sprintf("p[%d]", j), a.Probs[j], b.Probs[j]})
			}
			for _, v := range vals {
				if math.IsNaN(v.g) || math.IsInf(v.g, 0) {
					return math.Inf(1)
				}
				if d := math.Abs(v.w - v.g); d > worst {
					worst = d
					where = fmt.Sprintf("request %s, question %d, %s: host %.4f, device %.4f",
						decisionRequests[r], i, v.what, v.w, v.g)
				}
			}
			if a.Choice != b.Choice {
				t.Logf("request %s, question %d: the device chose %q, the host %q", decisionRequests[r], i, b.Choice, a.Choice)
			}
		}
	}
	t.Logf("worst at %s", where)
	return worst
}

// decisionDevViolation is a feature taken out of the graph the device runs,
// read when the blocks are offered (the plan), so only the device arm sees it.
type decisionDevViolation struct {
	name string
	cut  func(m *Model) func()
}

var decisionDevViolations = map[string]decisionDevViolation{
	// The norm's epsilon is read into the plan (LayerPlan.RMSEps).
	"d1-3B": {"a norm epsilon of 1", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.RMSEps = 1 })
	}},
	"lev": {"a norm epsilon of 1", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.RMSEps = 1 })
	}},
	// The device reads which blocks are local from the plan (mb.local): every
	// block global there attends over every row at the global base.
	"laya": {"every block global on the device", func(m *Model) func() {
		mb := m.enc.mb
		was := mb.local
		mb.local = func(int) bool { return false }
		return func() { mb.local = was }
	}},
}

// TestDecisionOnEveryDevice runs each decision model with every block on each
// device present and with half of them, against the host. The every-block arm
// needs a card the model fits; where it does not, the arm says how many blocks
// the card held and skips, and the seam arm still runs.
func TestDecisionOnEveryDevice(t *testing.T) {
	for _, c := range decisionCases {
		t.Run(c.name, func(t *testing.T) {
			m := openDecision(t, c.gguf, c.kind)
			defer m.Close()
			if !m.IsEncoder() {
				m.SetKVF16(false) // the device's history is f32
			}
			tol := decisionDevTol[c.name]
			host, _, _, _ := decideOn(t, m, nil, 0, nil)
			nb := decisionBlocks(m)
			ran := 0
			for _, spec := range stepDevices() {
				probe, err := tier.OpenWith(tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff))
				if err != nil || probe == nil {
					t.Logf("%s: not present (%v)", spec, err)
					continue
				}
				probe.Close()
				ran++
				t.Run(spec, func(t *testing.T) {
					open := func() *tier.GPU {
						g, err := tier.OpenWith(append([]tier.Option{tier.WithDevices(spec),
							tier.WithDeviceTune(tier.TuneOff)}, testTierOpts(t)...)...)
						if err != nil {
							t.Fatal(err)
						}
						return g
					}
					t.Run("seam", func(t *testing.T) {
						g := open()
						defer g.Close()
						// Half the blocks, or what a small card holds of them: a
						// seam either way, the rest on the host.
						want := nb / 2
						seam := func(n int) bool { return n > 0 && n <= want }
						got, placed, why, _ := decideOn(t, m, g, want, seam)
						if !seam(placed) {
							t.Fatalf("placed %d of the %d blocks offered: %s", placed, want, why)
						}
						worst := decisionDiff(t, host, got)
						t.Logf("%d of %d blocks on %s: worst difference from the host %.5f", placed, nb, spec, worst)
						if worst > tol {
							t.Errorf("the answers moved %.5f from the host's, past %.3f", worst, tol)
						}
					})
					t.Run("every-block", func(t *testing.T) {
						g := open()
						defer g.Close()
						all := func(n int) bool { return n == nb }
						got, placed, why, picked := decideOn(t, m, g, -1, all)
						if placed != nb {
							t.Skipf("%s held %d of %d blocks (%s): this card is too small for the every-block arm",
								spec, placed, nb, why)
						}
						worst := decisionDiff(t, host, got)
						t.Logf("every block on %s: worst difference from the host %.5f", spec, worst)
						if worst > tol {
							t.Errorf("the answers moved %.5f from the host's, past %.3f", worst, tol)
						}
						if !m.IsEncoder() {
							// The labels' logits were picked on the device, and
							// are the row's: the same placement reading the whole
							// row back answers bit for bit. Picked one past each
							// label, the answers move (the pick is what they read).
							if picked == 0 {
								t.Fatalf("no prompt's labels were picked on %s: the head's whole row came home", spec)
							}
							row, _, _, rowPicked := decideOn(t, m, g, -1, all, "no-pick")
							if rowPicked != 0 {
								t.Fatalf("the no-pick arm picked %d prompts", rowPicked)
							}
							if w := decisionDiff(t, got, row); w != 0 {
								t.Errorf("the picked labels answer %.6f from the row's on the same device", w)
							}
							off, _, _, _ := decideOn(t, m, g, -1, all, "pick-off-by-one")
							seen := decisionDiff(t, got, off)
							t.Logf("%d prompts picked on %s; picked one past each label, the answers move %.5f", picked, spec, seen)
							if seen <= tol {
								t.Errorf("a pick one past each label moved the answers %.5f: the gate does not see the pick", seen)
							}
						}
						v, ok := decisionDevViolations[c.name]
						if !ok {
							t.Fatalf("%s has no device violation", c.name)
						}
						// A fresh device, with g's room given back first: the
						// clean arm's blocks stay resident on g for whoever
						// attaches next (a decoder State's Close leaves them),
						// and would be shared rather than offered.
						g.Close()
						gv := open()
						defer gv.Close()
						undo := v.cut(m)
						bad, n, _, _ := decideOn(t, m, gv, -1, all)
						undo()
						if bad == nil {
							t.Fatalf("%s: the device held %d of %d blocks", v.name, n, nb)
						}
						seen := decisionDiff(t, host, bad)
						t.Logf("%s on %d device blocks moves the answers %.5f", v.name, n, seen)
						if seen <= tol {
							t.Errorf("%s moved the answers %.5f on %d blocks: the gate does not see it", v.name, seen, n)
						}
					})
				})
			}
			if ran == 0 {
				noDevice(t, "device", nil)
			}
		})
	}
}
