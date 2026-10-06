//go:build jitllmfault

package model

import (
	"os"
	"strings"
	"testing"

	"github.com/samyfodil/jitllm/internal/testmodels"
	"github.com/samyfodil/jitllm/jit/gpu/tier"
)

// TestWindowReleaseGateDiscriminates runs the device arm of
// TestWindowedLayersHoldTheirWindow with the page each window starts in
// released too (tier.SetPagedFault "winrelease"), and demands that the logits
// move or the decode fail: a gate that passes it cannot tell a window read
// from a released page.
func TestWindowReleaseGateDiscriminates(t *testing.T) {
	specs := []string{"cuda", "vulkan", "metal"}
	if v := os.Getenv("JITLLM_STEP_DEVICES"); v != "" {
		specs = strings.Split(v, ",")
	}
	m, err := Open(jlmOf(t, testmodels.Path("synth-exaone4.gguf")), noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	const maxSeq = 1024
	decode := func(g *tier.GPU) ([]uint64, error) {
		s := m.NewState(maxSeq)
		defer s.Close()
		if err := s.SetDevice(g); err != nil {
			t.Fatal(err)
		}
		if s.GPULayers() != m.Cfg.NLayer {
			t.Skipf("CARD TOO SMALL: %d of %d blocks -- this arm proved nothing", s.GPULayers(), m.Cfg.NLayer)
		}
		var sums []uint64
		for i := 0; i < maxSeq-8; i++ {
			lg, err := s.Forward(int32(3 + (i*7)%61))
			if err != nil {
				return sums, err
			}
			sums = append(sums, hashLogits(lg))
		}
		return sums, nil
	}
	for _, spec := range specs {
		t.Run(spec, func(t *testing.T) {
			open := func() *tier.GPU {
				g, err := tier.OpenWith(append([]tier.Option{tier.WithDevices(spec),
					tier.WithDeviceTune(tier.TuneOff)}, testTierOpts(t)...)...)
				if err != nil || g == nil {
					noDevice(t, spec, err)
				}
				return g
			}
			g := open()
			want, err := decode(g)
			g.Close()
			if err != nil {
				t.Fatal(err)
			}
			tier.SetPagedFault("winrelease")
			defer tier.SetPagedFault("")
			g = open()
			defer g.Close()
			got, err := decode(g)
			if g.Stats().KVWindowReleased == 0 {
				t.Fatal("no windowed page was released: the violation was never exercised")
			}
			t.Logf("violation: %s", violationWhat(err, got, want))
			if err == nil && equalSums(got, want) {
				t.Fatal("releasing the page each window starts in changed no logit on the device")
			}
		})
	}
}
