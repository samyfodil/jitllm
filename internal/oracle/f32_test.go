package oracle

import "testing"

// The f32 ops must agree with the f64 oracle to f32 precision.
func TestF32OpsMatchOracle(t *testing.T) {
	n := 2048
	x64 := make([]float64, n)
	w64 := make([]float64, n)
	for i := range x64 {
		x64[i] = float64((i%37)-18) * 0.137
		w64[i] = 1 + float64(i%11)*0.01
	}
	x32 := make([]float32, n)
	w32 := make([]float32, n)
	Narrow(x32, x64)
	Narrow(w32, w64)

	y64 := make([]float64, n)
	y32 := make([]float32, n)
	RMSNorm(y64, x64, w64, 1e-6)
	RMSNorm32(y32, x32, w32, 1e-6)
	var sse, sy2 float64
	for i := range y64 {
		d := float64(y32[i]) - y64[i]
		sse, sy2 = sse+d*d, sy2+y64[i]*y64[i]
	}
	if nmse := sse / sy2; nmse > 1e-12 {
		t.Errorf("RMSNorm32 NMSE %.3e against the f64 oracle", nmse)
	}

	s64 := append([]float64(nil), x64[:512]...)
	s32 := make([]float32, 512)
	Narrow(s32, s64)
	Softmax(s64)
	Softmax32(s32)
	sse, sy2 = 0, 0
	for i := range s64 {
		d := float64(s32[i]) - s64[i]
		sse, sy2 = sse+d*d, sy2+s64[i]*s64[i]
	}
	if nmse := sse / sy2; nmse > 1e-12 {
		t.Errorf("Softmax32 NMSE %.3e against the f64 oracle", nmse)
	}
}
