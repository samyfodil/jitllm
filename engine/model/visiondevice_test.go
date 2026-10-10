package model

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// The gemma3 and internvl towers on every device backend: every block placed,
// one block held to the host at the bound TestTowerOnDeviceMatchesHost sets,
// the whole tower's output held to the host's, and the blocks brought home
// again with the host's answer unchanged (relocation).

func nmseOf(got, want []float32) (nmse, worst float64) {
	var num, den float64
	for i := range want {
		d := float64(got[i] - want[i])
		num += d * d
		den += float64(want[i]) * float64(want[i])
		worst = math.Max(worst, math.Abs(d))
	}
	return num / den, worst
}

func towerOnDevices(t *testing.T, open func(testing.TB, ...Option) (*Model, *Tower)) {
	m, tw := open(t)
	defer m.Close()
	c := tw.Cfg
	px := rainbow(c.ImageSz)
	// Normalised as the tower's own preprocessing would, so the residual has
	// the magnitude a real picture gives it.
	for i := range px {
		ch := i % 3
		px[i] = float32((float64(px[i]) - c.Mean[ch]) / c.Std[ch])
	}
	encode := func(s *State) (after0, out []float32) {
		s.vis.afterBlock = func(li int, x []float32) {
			if li == 0 && after0 == nil {
				after0 = append([]float32(nil), x...)
			}
		}
		e, err := s.Encode(px)
		if err != nil {
			t.Fatal(err)
		}
		return after0, append([]float32(nil), e...)
	}
	host := tw.testState()
	want0, want := encode(host)
	host.Close()

	ran := 0
	for _, spec := range stepDevices() {
		g, err := tier.OpenWith(tier.WithDevices(spec))
		if err != nil || g == nil {
			t.Logf("%s: not present (%v)", spec, err)
			continue
		}
		ran++
		t.Run(spec, func(t *testing.T) {
			defer g.Close()
			// Every block, and the whole tower's output.
			all := tw.testState()
			all.SetDevice(g)
			if n := all.GPUBlocks(); n != c.NLayer {
				all.Close()
				t.Fatalf("%s took %d of %d tower blocks (%s)", spec, n, c.NLayer, g.Err())
			}
			_, gotAll := encode(all)
			nm, worst := nmseOf(gotAll, want)
			t.Logf("%s, all %d blocks: projector output NMSE %.3e, max|d| %.4f", spec, c.NLayer, nm, worst)
			if math.IsNaN(nm) || math.IsInf(nm, 0) || nm > 1e-2 {
				t.Errorf("%s: the whole tower on the device reads NMSE %.3e against the host", spec, nm)
			}
			// Relocation: the blocks come home through the one placement path
			// (State.SetDevice releases them) and the host's answer is the
			// host's again, bit for bit.
			if err := all.SetDevice(nil); err != nil {
				t.Fatalf("%s: bringing the tower home: %v", spec, err)
			}
			_, back := encode(all)
			all.Close()
			if nb, _ := nmseOf(back, want); nb != 0 {
				t.Errorf("%s: after the blocks came home the host reads NMSE %.3e against itself", spec, nb)
			}

			// One block, held to the host after it.
			defer func(was int) { tw.opt.towerGPULayers = was }(tw.opt.towerGPULayers)
			tw.opt.towerGPULayers = 1
			s := tw.testState()
			defer s.Close()
			s.SetDevice(g)
			if s.GPUBlocks() != 1 {
				t.Fatalf("asked for 1 device block, got %d", s.GPUBlocks())
			}
			got0, _ := encode(s)
			nm, worst = nmseOf(got0, want0)
			t.Logf("%s, after block 0: NMSE %.3e, max|d| %.4f", spec, nm, worst)
			if math.IsNaN(nm) || math.IsInf(nm, 0) || math.IsInf(worst, 0) {
				t.Fatalf("NMSE %v on %s: the comparison is degenerate", nm, spec)
			}
			bound := 5e-5
			if g.Stats().VoltaGemm > 0 {
				bound = 5e-4
			}
			if nm > bound {
				t.Fatalf("NMSE %.3e after ONE block on %s (bound %.0e)", nm, spec, bound)
			}
		})
	}
	if ran == 0 {
		t.Skip("no GPU backend on this host")
	}
}

func TestGemma3TowerOnDevice(t *testing.T)   { towerOnDevices(t, openGemma3) }
func TestInternVLTowerOnDevice(t *testing.T) { towerOnDevices(t, openInternVL) }
