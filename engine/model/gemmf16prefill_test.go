package model

import (
	"math"
	"os"
	"strings"
	"testing"

	"github.com/samyfodil/jitllm/internal/testmodels"
	"github.com/samyfodil/jitllm/jit/gpu/tier"
)

// TestF16GemmPrefillMatchesTheHost prefills real models on CUDA with every
// block placed, through the staged binary16 GEMM on the m16n8 instruction
// (tier's f16Gemm: sm_75 and later), and holds the prompt's logits AND the
// decode that reads the chunk's KV to the host. The chunk's attention
// accumulates on m16n8k8 (kernels.PagedAttnAccMMA) behind the m16n8k16
// scores, counted by Stats().PagedAccMMA.
//
// The models cover the formats and features the GEMM carries: Q4_K/Q6_K,
// Q3_K beside Q4_K/Q5_K/Q6_K (Q3_K has no binary16 dequant and keeps the int8
// matvec in the same block), Q8_0, biased q/k/v, Q4_0, the hybrid's
// projections and a mixture's dense ones. Stats().GemmF16 is the selection
// check, since the int8 twin also answers correctly; a card whose PTX target
// has the instruction and built none fails.
//
// The host uses int8 activations and the GEMM binary16, so the bar is the
// tile's: NMSE under 2e-2 and an argmax that agrees or is a tie (tieMargin).
// The split tuner is pinned off (RULE 11c).
func TestF16GemmPrefillMatchesTheHost(t *testing.T) {
	ran := 0
	for _, name := range []string{"Llama-3.2-1B-Instruct-Q4_K_M.gguf", "Qwen3-0.6B-Q8_0.gguf",
		"tinyllama-1.1b-q3_K_M.gguf", "Qwen2-1.5B-Instruct-Q4_K_M.gguf", "gemma-2b.gguf",
		"Qwen3.5-0.8B-Q4_K_M.gguf", "Qwen3-MOE-4x0.6B-Q4_K_M.gguf"} {
		p := testmodels.Path(name)
		if _, err := os.Stat(p); err != nil {
			if _, err := os.Stat(strings.TrimSuffix(p, ".gguf") + ".jlm"); err != nil {
				t.Logf("MODEL MISSING: %s", name)
				continue
			}
		}
		t.Run(name, func(t *testing.T) {
			m, err := Open(jlmOf(t, p), noTune, WithKVF16(false))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			text := strings.Repeat("The river rose every spring until the bridges were islands, "+
				"and the clerk counted barrels of salt on the upper floor. ", 40)
			ids := m.Vocab.Encode(text, true)[:200]
			drive := []int32{ids[3], ids[17], ids[40], ids[7]}
			run := func(s *State) [][]float32 {
				l, err := s.Prefill(ids)
				if err != nil {
					t.Fatal(err)
				}
				out := [][]float32{append([]float32(nil), l...)}
				for _, tok := range drive {
					if l, err = s.Forward(tok); err != nil {
						t.Fatal(err)
					}
					out = append(out, append([]float32(nil), l...))
				}
				return out
			}
			host := m.NewState(len(ids) + 8)
			want := run(host)
			host.Close()

			g, err := tier.OpenWith(tier.WithAPI("ptx"), tier.WithDeviceTune(tier.TuneOff))
			if err != nil || g == nil {
				noDevice(t, "cuda", err)
			}
			defer g.Close()
			if !strings.Contains(g.Name(), "[ptx]") {
				t.Skipf("the device is %s; the m16n8 GEMM is CUDA's", g.Name())
			}
			dev := m.NewState(len(ids) + 8)
			dev.SetDevice(g)
			if dev.GPULayers() != m.Cfg.NLayer {
				dev.Close()
				t.Skipf("CARD TOO SMALL: the device took %d of %d blocks (%s) -- this model proved nothing here",
					dev.GPULayers(), m.Cfg.NLayer, g.Err())
			}
			got := run(dev)
			demoted, placed := dev.DeviceDemotions(), dev.GPULayers()
			dev.SetGPULayers(0)
			dev.Close()
			if demoted != 0 || placed != m.Cfg.NLayer {
				t.Fatalf("%d demotion(s), %d of %d blocks on the device: the host answered",
					demoted, placed, m.Cfg.NLayer)
			}
			for i := range want {
				var num, den float64
				for j := range want[i] {
					v := float64(got[i][j])
					if math.IsNaN(v) || math.IsInf(v, 0) {
						t.Fatalf("step %d: logit %d is %v on the device", i, j, v)
					}
					d := v - float64(want[i][j])
					num += d * d
					den += float64(want[i][j]) * float64(want[i][j])
				}
				ga, wa := argmaxID(got[i]), argmaxID(want[i])
				t.Logf("step %d: logit NMSE %.3e, argmax %d (host %d)", i, num/den, ga, wa)
				if num/den > 2e-2 {
					t.Fatalf("step %d: logit NMSE %.3e against the host", i, num/den)
				}
				if ga != wa && float64(want[i][wa]-want[i][ga]) > tieMargin {
					t.Fatalf("step %d: argmax %d against the host's %d on a margin of %.3f",
						i, ga, wa, want[i][wa]-want[i][ga])
				}
			}
			// After the comparison, so a run with the GEMM switched off still
			// prints the int8 arm's numbers.
			st := g.Stats()
			t.Logf("%s: %d matvecs on the m16n8 GEMM, %d on sm_70's; %d attention accumulates on m16n8k8",
				g.Name(), st.GemmF16, st.VoltaMV, st.PagedAccMMA)
			if st.GemmF16 == 0 && ptxTargetSM(g.Name()) >= 75 {
				t.Fatalf("%s lowers the m16n8 binary16 instruction and built no GEMM on it: the int8 twin answered", g.Name())
			}
			// The accumulate rides the m16n8k16 scores, so from sm_80; a
			// hybrid's attention blocks and a mixture's take it as well.
			if st.PagedAccMMA == 0 && ptxTargetSM(g.Name()) >= 80 {
				t.Fatalf("%s: no prompt attention accumulated on m16n8k8: the FMA tiles answered", g.Name())
			}
			ran++
		})
	}
	if ran == 0 {
		testmodels.Missing(t, "no model ran: this gate proved nothing (set JITLLM_MODELS)")
	}
}

// ptxTargetSM reads the target out of a CUDA device's name, "... (sm_86)
// [ptx]", and is 0 when there is none.
func ptxTargetSM(name string) int {
	i := strings.LastIndex(name, "(sm_")
	if i < 0 {
		return 0
	}
	sm := 0
	for _, c := range name[i+4:] {
		if c < '0' || c > '9' {
			break
		}
		sm = sm*10 + int(c-'0')
	}
	return sm
}
