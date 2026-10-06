package model

import (
	"math"
	"testing"
)

// TestSamplerDrawsTheTemperedSoftmax draws many tokens from four logits and
// checks the frequencies against softmax(logit/T), with top-k cutting the
// fourth. The softmax runs on the generated kernel, so this is also the gate
// that the sampler hands it the right row.
func TestSamplerDrawsTheTemperedSoftmax(t *testing.T) {
	logits := []float32{2, 1, 0.5, 3.5}
	const temp, n = 1.5, 200000
	s := &Sampler{Temp: temp, TopK: 3, Seed: 7}
	counts := map[int32]int{}
	for i := 0; i < n; i++ {
		counts[s.Sample(logits)]++
	}
	// Top-3 by logit is tokens 3, 0, 1: token 2 must never be drawn.
	if counts[2] != 0 {
		t.Fatalf("top-k 3 drew the fourth-best token %d times", counts[2])
	}
	var z float64
	for _, id := range []int{3, 0, 1} {
		z += math.Exp(float64(logits[id]) / temp)
	}
	for _, id := range []int{3, 0, 1} {
		want := math.Exp(float64(logits[id])/temp) / z
		got := float64(counts[int32(id)]) / n
		if math.Abs(got-want) > 0.01 {
			t.Errorf("token %d drawn %.4f of the time, want %.4f", id, got, want)
		}
	}
	// And the zero value is greedy.
	if g := (&Sampler{}).Sample(logits); g != 3 {
		t.Fatalf("the zero Sampler picked %d, want the argmax 3", g)
	}
}
