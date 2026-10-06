//go:build amd64 || arm64

package nn

import (
	"math"
	"testing"
)

// The multi-axis rotary table's gates. The oracle is ropeOracle's float64
// arithmetic per run: pair j of run u at position pos[u.Axis] and frequency
// index u.Freq + j*u.Step of the plain recurrence.

func ropeAxesOracle(r Rope, pos []int) []float64 {
	mscale := r.Scale
	if mscale == 0 {
		mscale = 1
	}
	step := math.Pow(r.Base, -2/float64(r.NRot))
	var out []float64
	for _, u := range r.Runs {
		for j := 0; j < u.Pairs; j++ {
			i := u.Freq + j*u.step()
			f := 1.0
			for k := 0; k < i; k++ {
				f *= step
			}
			if i < len(r.Freqs) {
				f /= float64(r.Freqs[i])
			}
			th := float64(pos[u.Axis]) * f
			out = append(out, math.Cos(th)*mscale, math.Sin(th)*mscale)
		}
	}
	return out
}

// ropeAxesConfigs are the three shapes the capability exists for, at the
// geometry of a real checkpoint, plus a ragged one narrower than a vector.
func ropeAxesConfigs() []struct {
	name string
	r    Rope
} {
	return []struct {
		name string
		r    Rope
	}{
		// Qwen2-VL-2B's text: head_dim 128, sections [16, 24, 24].
		{"qwen2vl-text", Rope{NRot: 128, Base: 1000000, Neox: true, Runs: MRopeRuns([]int{16, 24, 24, 0})}},
		// Qwen2-VL's ViT: head_dim 80, a half-head rotary (NRot 40) whose 20
		// frequencies serve the row and then again the column.
		{"qwen2vl-vit", Rope{NRot: 40, Base: 10000, Neox: true,
			Runs: []RopeRun{{Pairs: 20, Axis: 0}, {Pairs: 20, Axis: 1}}}},
		// Pixtral's ViT: the even frequencies to the row, the odd to the column.
		{"pixtral-vit", Rope{NRot: 64, Base: 10000, Neox: true,
			Runs: []RopeRun{{Pairs: 16, Axis: 0, Step: 2}, {Pairs: 16, Axis: 1, Freq: 1, Step: 2}}}},
		{"ragged", Rope{NRot: 14, Base: 10000, Runs: MRopeRuns([]int{3, 2, 2})}},
	}
}

// TestRopeTableAtMatchesTheTranscendentals holds every run to math.Cos and
// math.Sin at the plain table's bound, with the coordinates far apart so a run
// on the wrong axis is O(1).
func TestRopeTableAtMatchesTheTranscendentals(t *testing.T) {
	const bound = 3e-7
	ran := 0
	var worst float64
	for _, c := range ropeAxesConfigs() {
		cs := make([]float32, 2*RopeRuns(c.r.Runs))
		for _, pos := range [][]int{{0, 0, 0}, {7, 3, 11}, {1000, 1017, 1003}, {40000, 2, 39999}} {
			before := RopeTableCalls()
			c.r.TableAt(cs, pos)
			if RopeTableCalls()-before != int64(len(c.r.Runs)) {
				t.Fatalf("%s: %d kernel calls for %d runs -- the table did not come from generated code",
					c.name, RopeTableCalls()-before, len(c.r.Runs))
			}
			ran++
			for i, w := range ropeAxesOracle(c.r, pos) {
				d := math.Abs(float64(cs[i]) - w)
				worst = math.Max(worst, d)
				if d > bound {
					t.Errorf("%s at %v entry %d: %.9g against %.9g", c.name, pos, i, cs[i], w)
				}
			}
		}
	}
	t.Logf("%d multi-axis tables, worst |d| %.3e against %.1e", ran, worst, bound)
}

// TestRopeTableAtIsThePlainTableWhenTheAxesAgree is what makes M-RoPE free for
// a text row: equal coordinates over one frequency sequence are the plain
// table, bit for bit, so a text-only prompt is unchanged by construction.
func TestRopeTableAtIsThePlainTableWhenTheAxesAgree(t *testing.T) {
	r := ropeAxesConfigs()[0].r
	a, b := make([]float32, r.NRot), make([]float32, r.NRot)
	for _, p := range []int{0, 1, 5, 400, 4095, 30000} {
		r.Table(a, p)
		r.TableAt(b, []int{p, p, p})
		for i := range a {
			if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
				t.Fatalf("position %d entry %d: plain %v, multi-axis %v", p, i, a[i], b[i])
			}
		}
	}
	// The violation: the coordinates disagreeing must move the table, or the
	// equality above is two arms computing one thing.
	r.TableAt(b, []int{5, 9, 5})
	r.Table(a, 5)
	same := true
	for i := range a {
		same = same && a[i] == b[i]
	}
	if same {
		t.Fatal("a row at (5, 9, 5) has the table of (5, 5, 5): the h run never read its coordinate")
	}
}
