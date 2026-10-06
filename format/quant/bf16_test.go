package quant

import (
	"encoding/binary"
	"math"
	"testing"
)

// TestBF16IsExact asserts zero ULP, in both destinations, for all 65536 bit
// patterns: bf16 decoding is a shift, so anything but exact equality (on
// subnormals, signed zeros, infinities and NaN payloads too) is a defect.
func TestBF16IsExact(t *testing.T) {
	src := make([]byte, 2*65536)
	for i := 0; i < 65536; i++ {
		binary.LittleEndian.PutUint16(src[i*2:], uint16(i))
	}
	got64 := make([]float64, 65536)
	if err := Dequant(BF16, src, got64); err != nil {
		t.Fatal(err)
	}
	got32 := make([]float32, 65536)
	if err := Dequant32(BF16, src, got32); err != nil {
		t.Fatal(err)
	}

	nan, subnormal, inf := 0, 0, 0
	for i := 0; i < 65536; i++ {
		want := math.Float32frombits(uint32(i) << 16)
		switch {
		case math.IsNaN(float64(want)):
			nan++
			if !math.IsNaN(float64(got32[i])) || !math.IsNaN(got64[i]) {
				t.Fatalf("pattern %#04x is NaN; got32 %v got64 %v", i, got32[i], got64[i])
			}
			continue
		case math.IsInf(float64(want), 0):
			inf++
		case want != 0 && math.Abs(float64(want)) < math.SmallestNonzeroFloat32*(1<<23):
			subnormal++
		}
		// Bits, not values: -0.0 == +0.0 compares equal.
		if math.Float32bits(got32[i]) != math.Float32bits(want) {
			t.Fatalf("pattern %#04x: got32 %#08x, want %#08x",
				i, math.Float32bits(got32[i]), math.Float32bits(want))
		}
		if got64[i] != float64(want) {
			t.Fatalf("pattern %#04x: got64 %v, want %v", i, got64[i], float64(want))
		}
	}
	if nan == 0 || inf == 0 || subnormal == 0 {
		t.Fatalf("the sweep saw %d NaN, %d Inf, %d subnormal -- it is not covering the domain", nan, inf, subnormal)
	}
	if !Dequantable(BF16) {
		t.Fatal("BF16 decodes and Dequantable says otherwise")
	}
	t.Logf("65536 patterns exact in both destinations (%d NaN, %d Inf, %d subnormal)", nan, inf, subnormal)
}
