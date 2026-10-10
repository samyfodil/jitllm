package model

import (
	"math"
	"testing"

	"github.com/samyfodil/jitllm/jit/gpu/tier"
)

// embedDevCos is how close a device's embedding must sit to the host's: the
// f32 reduction-order band of a unit vector, an order of magnitude inside
// the reference gate's 1e-4 (embedCosBound), so a device arm that passes
// here passes that gate wherever the host does.
const embedDevCos = 1 - 1e-5

// TestEmbeddingsOnEveryDevice runs every BERT-family embedding model with
// every encoder block on each device present, and with half of them, against
// the host and against the references TestEmbeddingsMatchReference holds the
// host to -- the encoder's segment through the one placement path
// (encdev.go). The violation is the output LayerNorm's weight taken out of
// what the device is handed: BERT's post-norm order is the device's own wiring
// (nn.LayerPlan.PostResidNorm), which a gate running the pre-norm block would
// not see.
func TestEmbeddingsOnEveryDevice(t *testing.T) {
	for _, g := range loadEmbedGoldens(t) {
		t.Run(g.Model, func(t *testing.T) {
			m := openEmbedModel(t, g)
			if !m.IsEncoder() {
				t.Skipf("%s is a decoder embedding model: its device path is its State's, not the encoder's", g.Model)
			}
			embed := func(e *Embedder) [][]float32 {
				var out [][]float32
				for _, text := range g.Texts {
					v, err := e.Embed(m.EmbedIDs(text))
					if err != nil {
						t.Fatal(err)
					}
					out = append(out, append([]float32(nil), v...))
				}
				return out
			}
			he, err := m.NewEmbedder()
			if err != nil {
				t.Fatal(err)
			}
			host := embed(he)
			he.Close()
			nb := m.encBlocks()
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
					arm := func(n int) ([][]float32, int) {
						gpu, err := tier.OpenWith(append([]tier.Option{tier.WithDevices(spec),
							tier.WithDeviceTune(tier.TuneOff)}, testTierOpts(t)...)...)
						if err != nil {
							t.Fatal(err)
						}
						defer gpu.Close()
						e, err := m.NewEmbedder()
						if err != nil {
							t.Fatal(err)
						}
						defer e.Close()
						if err := e.SetDeviceLayers(gpu, n); err != nil {
							t.Fatal(err)
						}
						placed := e.DeviceBlocks()
						if want := n; n < 0 && placed != nb || n >= 0 && placed != want {
							t.Fatalf("placed %d blocks, offered %d of %d: %v %s", placed, n, nb, e.DeviceDeclines(), gpu.Err())
						}
						// The device ran them: one submission a run of placed
						// blocks a text, counted by the device itself.
						s0 := gpu.Stats().Submits
						out := embed(e)
						if s := gpu.Stats().Submits - s0; s < len(g.Texts) {
							t.Fatalf("%d submission(s) for %d texts: the placed blocks did not run on the device", s, len(g.Texts))
						}
						return out, placed
					}
					closest := func(got [][]float32) float64 {
						w := 1.0
						for i := range host {
							h := make([]float64, len(host[i]))
							for j, x := range host[i] {
								h[j] = float64(x)
							}
							w = math.Min(w, cosine(got[i], h))
						}
						return w
					}
					for _, n := range []int{nb / 2, -1} {
						got, placed := arm(n)
						c := closest(got)
						t.Logf("%d of %d blocks on %s: worst cosine to the host %.9f", placed, nb, spec, c)
						if !(c >= embedDevCos) {
							t.Errorf("%d blocks on %s: cosine to the host %.9f < %v", placed, spec, c, embedDevCos)
						}
						if placed == nb {
							for i := range got {
								if i < len(g.Llamacpp) {
									if lc := cosine(got[i], g.Llamacpp[i]); !(lc >= embedCosBound) && g.LGGUF == "" {
										t.Errorf("text %d on %s: cosine to llama.cpp %.7f < %v", i, spec, lc, embedCosBound)
									}
								}
							}
						}
					}
					// The violation: the output norm's weight gone from what the
					// device is handed (encWeights reads it at the offer).
					saved := make([][]float32, len(m.enc.blocks))
					for i := range m.enc.blocks {
						b := &m.enc.blocks[i]
						saved[i] = b.ln2W
						ones := make([]float32, len(b.ln2W))
						for j := range ones {
							ones[j] = 1
						}
						b.ln2W = ones
					}
					bad, _ := arm(-1)
					for i := range m.enc.blocks {
						m.enc.blocks[i].ln2W = saved[i]
					}
					c := closest(bad)
					t.Logf("the output LayerNorm without its weight on %s: worst cosine to the host %.6f", spec, c)
					if c >= embedDevCos {
						t.Errorf("the violation reads cosine %.9f on %s: the gate does not see it", c, spec)
					}
				})
			}
			if ran == 0 {
				noDevice(t, "device", nil)
			}
		})
	}
}
