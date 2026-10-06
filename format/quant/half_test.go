package quant

import (
	"math"
	"testing"
)

// TestEncodeHalfRoundTrips checks all 65536 half patterns, not a sample. NaN is
// excluded from the identity because every NaN payload maps to one quiet NaN.
func TestEncodeHalfRoundTrips(t *testing.T) {
	var subnormals, zeros int
	for u := 0; u < 1<<16; u++ {
		h := uint16(u)
		if (h>>10)&0x1F == 0x1F && h&0x3FF != 0 {
			continue // NaN
		}
		f := float32(DecodeHalf(h))
		got := EncodeHalf(f)
		if got != h {
			t.Fatalf("0x%04X decoded to %v and re-encoded to 0x%04X", h, f, got)
		}
		switch {
		case (h>>10)&0x1F != 0:
		case h&0x3FF == 0:
			zeros++
		default:
			subnormals++
		}
	}
	if subnormals != 2046 || zeros != 2 {
		t.Fatalf("covered %d subnormals and %d zeros; the sweep is not covering the range",
			subnormals, zeros)
	}

	// Rounding, range and the two saturations, which the identity above cannot
	// reach because they are f32 values with no exact half.
	for _, c := range []struct {
		in   float32
		want float32
	}{
		{1 + 1.0/2048, 1},             // exactly halfway, ties to even
		{1 + 3.0/2048, 1 + 2.0/1024},  // rounds up
		{65520, float32(math.Inf(1))}, // just past the top
		{65504, 65504},                // the largest finite half
		{1e-8, 0},                     // under the smallest subnormal
		{-1e-8, 0},                    // and its sign
	} {
		if got := float32(DecodeHalf(EncodeHalf(c.in))); got != c.want {
			t.Errorf("EncodeHalf(%v) -> %v, want %v", c.in, got, c.want)
		}
	}
}
