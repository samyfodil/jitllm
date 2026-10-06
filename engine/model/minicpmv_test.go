package model

import (
	"encoding/json"
	"fmt"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"testing"
)

// The MiniCPM-V gates that need no model: the processor's cut of a picture,
// the tower's position buckets and Pillow's resize, each against the reference
// itself (scripts/minicpmvgold.py --layouts and --pil, which run MiniCPM-V's
// own image processor, torch and Pillow).

// minicpmvTower is the geometry of MiniCPM-V 2.6/4.0/4.5's tower: a 448 slice
// of 14-pixel patches over a 70x70 bucketed table, normalised by 0.5.
func minicpmvTower() *Tower {
	o := defaultOpts()
	return &Tower{opt: &o, Cfg: TowerConfig{
		ImageSz: 448, PatchSz: 14, PosBuckets: 70, Queries: 64, NEmbd: 1152,
		Mean: [3]float64{0.5, 0.5, 0.5}, Std: [3]float64{0.5, 0.5, 0.5},
	}}
}

type minicpmvLayoutGold struct {
	Layouts []struct {
		W, H     int
		Overview [2]int
		Grid     []int
		Refine   []int
	}
	Buckets map[string][][2]int
}

func readMinicpmvLayoutGold(t *testing.T) minicpmvLayoutGold {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("..", "..", "testdata", "golden", "minicpmv-layout.json"))
	if err != nil {
		t.Fatalf("GOLDEN MISSING: %v (scripts/minicpmvgold.py DIR --layouts)", err)
	}
	var g minicpmvLayoutGold
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatal(err)
	}
	if len(g.Layouts) == 0 || len(g.Buckets) == 0 {
		t.Fatal("the golden is empty, so this gate proves nothing")
	}
	return g
}

// TestMiniCPMVLayoutMatchesProcessor holds Tower.Layout to the processor's own
// find_best_resize, get_sliced_grid and get_refine_size over pictures from a
// thumbnail to a 6000-pixel strip: the cut decides how many pieces and tokens
// the prompt holds, and a wrong one is a fluent answer about a different
// picture.
func TestMiniCPMVLayoutMatchesProcessor(t *testing.T) {
	g := readMinicpmvLayoutGold(t)
	tw := minicpmvTower()
	sliced := 0
	for _, want := range g.Layouts {
		lay, err := tw.Layout(want.W, want.H)
		if err != nil {
			t.Errorf("%dx%d: %v", want.W, want.H, err)
			continue
		}
		ov := lay.Pieces[0]
		if ov.W != want.Overview[0] || ov.H != want.Overview[1] {
			t.Errorf("%dx%d: overview %dx%d, the processor's %dx%d", want.W, want.H, ov.W, ov.H,
				want.Overview[0], want.Overview[1])
		}
		if want.Grid == nil {
			if lay.Rows != 0 || len(lay.Pieces) != 1 {
				t.Errorf("%dx%d: %dx%d slices where the processor cuts none", want.W, want.H, lay.Cols, lay.Rows)
			}
			continue
		}
		sliced++
		if lay.Cols != want.Grid[0] || lay.Rows != want.Grid[1] {
			t.Errorf("%dx%d: grid %dx%d, the processor's %dx%d", want.W, want.H, lay.Cols, lay.Rows,
				want.Grid[0], want.Grid[1])
		}
		if lay.RefW != want.Refine[0] || lay.RefH != want.Refine[1] {
			t.Errorf("%dx%d: refined %dx%d, the processor's %dx%d", want.W, want.H, lay.RefW, lay.RefH,
				want.Refine[0], want.Refine[1])
		}
		if len(lay.Pieces) != 1+lay.Rows*lay.Cols {
			t.Errorf("%dx%d: %d pieces for a %dx%d grid", want.W, want.H, len(lay.Pieces), lay.Cols, lay.Rows)
		}
	}
	if sliced == 0 {
		t.Fatal("no golden picture is sliced, so the slice arm proved nothing")
	}
	t.Logf("%d sizes, %d of them sliced, every cut the processor's", len(g.Layouts), sliced)
}

// TestBucketPositionsMatchTorch holds the tower's position rows to the
// rational floor(70i/n) on every axis length to 1499, and counts where the
// reference's float32 torch parts from it: the golden records each such place
// from torch itself, so a gate that could not tell the two apart is refused.
// See bucketPositions for why the rational is the choice.
func TestBucketPositionsMatchTorch(t *testing.T) {
	g := readMinicpmvLayoutGold(t)
	const B = 70
	diverge, lengths := 0, 0
	for n := 1; n < 1500; n++ {
		got := make([]int32, n)
		bucketPositions(got, 1, n, B)
		for i := range got {
			if want := int32(B * i / n); got[i] != want {
				t.Fatalf("axis of %d: patch %d buckets to %d, the rational to %d", n, i, got[i], want)
			}
		}
		if es := g.Buckets[fmt.Sprint(n)]; len(es) > 0 {
			lengths++
			for _, e := range es {
				if d := e[1] - (B*e[0])/n; d != 1 && d != -1 {
					t.Errorf("axis of %d: torch's bucket %d at patch %d is %d off the rational, "+
						"more than one rounding can move it", n, e[1], e[0], d)
				}
				diverge++
			}
		}
	}
	if diverge == 0 {
		t.Fatal("the golden records no float32 divergence, so this gate cannot tell torch from a rational floor")
	}
	// The 448 slice's own axis, 32 patches, is not one of them: a square
	// overview reads the same rows in torch, llama.cpp and here.
	if len(g.Buckets["32"]) != 0 {
		t.Errorf("torch parts from the rational on a 32-patch axis: %v", g.Buckets["32"])
	}
	t.Logf("1499 axis lengths at the rational; torch's float32 parts from it by one bucket at %d "+
		"coordinates over %d lengths, none of them 32", diverge, lengths)
}

// TestPILBicubicMatchesPillow holds imageproc.go's resampler, at Pillow's
// precision, to Pillow's own BICUBIC resize, up and down and unequal per axis:
// the same integer arithmetic, so every sample. The violation is torchvision's
// precision -- the same resampler with the widest int16 weights -- which
// rounds samples the other way.
func TestPILBicubicMatchesPillow(t *testing.T) {
	f, err := os.Open(testImage)
	if err != nil {
		t.Fatal(err)
	}
	img, _, err := image.Decode(f)
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	src := toRGB8(img)
	dir := filepath.Join("..", "..", "testdata", "golden", "minicpmv-pil")
	ents, err := os.ReadDir(dir)
	if err != nil || len(ents) == 0 {
		t.Fatalf("GOLDEN MISSING: %s (scripts/minicpmvgold.py --pil)", dir)
	}
	torchOff := 0
	for _, e := range ents {
		var tw, th int
		if _, err := fmt.Sscanf(e.Name(), "%dx%d.png", &tw, &th); err != nil {
			continue
		}
		gf, err := os.Open(filepath.Join(dir, e.Name()))
		if err != nil {
			t.Fatal(err)
		}
		gi, err := png.Decode(gf)
		gf.Close()
		if err != nil {
			t.Fatal(err)
		}
		want := toRGB8(gi)
		if want.w != tw || want.h != th {
			t.Fatalf("%s is %dx%d", e.Name(), want.w, want.h)
		}
		off := func(got rgb8) int {
			n := 0
			for i := range want.px {
				if got.px[i] != want.px[i] {
					n++
				}
			}
			return n
		}
		pil, torch := off(src.resizePIL(tw, th, aaBicubic)), off(src.resize(tw, th, aaBicubic))
		torchOff += torch
		t.Logf("%dx%d -> %dx%d: %d of %d samples off Pillow's; at torchvision's precision %d",
			src.w, src.h, tw, th, pil, len(want.px), torch)
		if pil != 0 {
			t.Errorf("%dx%d: %d samples differ from Pillow -- not Pillow's resize", tw, th, pil)
		}
	}
	if torchOff == 0 {
		t.Error("torchvision's precision matches Pillow on every golden too, so the gate cannot tell them apart")
	}
}
