package oracle

import "math"

// DecisionAnswer is the reference arithmetic of a decision's answer, in
// float64 as llama.cpp's server-decision.cpp (format_answer) and the models'
// own code compute it: each variant's scores scaled by 1/t and softmaxed,
// the second variant's put back in reverse, the variants averaged; then the
// expected level, the mode, TypeSafe's two confidences and Laya's entropy
// confidence.
type DecisionAnswer struct {
	Probs                       []float64
	Expected                    float64 // sum i * p_i
	Mode                        int
	ConfChoice, ConfScore, Laya float64
}

// Decide computes DecisionAnswer over variants of n scores each.
func Decide(variants [][]float32, t float64) DecisionAnswer {
	n := len(variants[0])
	a := DecisionAnswer{Probs: make([]float64, n)}
	for v, s := range variants {
		mx := math.Inf(-1)
		for _, x := range s {
			mx = math.Max(mx, float64(x))
		}
		p := make([]float64, n)
		sum := 0.0
		for i, x := range s {
			p[i] = math.Exp((float64(x) - mx) / t)
			sum += p[i]
		}
		for i := range p {
			j := i
			if v == 1 {
				j = n - 1 - i
			}
			a.Probs[j] += p[i] / sum / float64(len(variants))
		}
	}
	for i, p := range a.Probs {
		a.Expected += float64(i) * p
		if p > a.Probs[a.Mode] {
			a.Mode = i
		}
	}
	if n < 2 {
		a.ConfChoice, a.ConfScore, a.Laya = 1, 1, 1
		return a
	}
	u := 1 / float64(n)
	a.ConfChoice = math.Max(0, (a.Probs[a.Mode]-u)/(1-u))
	var dist, uni, h float64
	for i, p := range a.Probs {
		dist += p * math.Abs(float64(i-a.Mode))
		uni += math.Abs(float64(i)-float64(n-1)/2) / float64(n)
		h -= p * math.Log(math.Max(p, 1e-300))
	}
	a.ConfScore = math.Max(0, 1-dist/uni)
	a.Laya = 1 - h/math.Log(float64(n))
	return a
}
