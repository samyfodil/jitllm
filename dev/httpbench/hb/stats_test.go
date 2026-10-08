package hb

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
	"time"
)

// TestPercentileInterpolatesAsNumpy: values checked against
// numpy.percentile's default (linear) on the same inputs.
func TestPercentileInterpolatesAsNumpy(t *testing.T) {
	shuffled := []float64{40, 15, 50, 35, 20} // unsorted input must not matter
	for _, c := range []struct{ p, want float64 }{
		{0, 15}, {25, 20}, {40, 29}, {50, 35}, {90, 46}, {99, 49.6}, {100, 50},
	} {
		if got := Percentile(shuffled, c.p); math.Abs(got-c.want) > 1e-9 {
			t.Errorf("p%v = %v, want %v", c.p, got, c.want)
		}
	}
	if got := Percentile([]float64{7}, 99); got != 7 {
		t.Errorf("p99 of one value = %v", got)
	}
	if !math.IsNaN(Percentile(nil, 50)) {
		t.Error("an empty set has a percentile; it must be NaN, never 0")
	}
	if shuffled[0] != 40 {
		t.Error("Percentile sorted its caller's slice")
	}
}

// TestGateRefusesWhatRULE2Refuses: a tight series of six passes; a wide one,
// a short one and one with a failed round are refused.
func TestGateRefusesWhatRULE2Refuses(t *testing.T) {
	g := Gate([]float64{1.00, 1.01, 0.99, 1.02, 1.00, 0.98})
	if g.Refused != "" || math.Abs(float64(g.Median)-1.0) > 1e-9 || g.N != 6 {
		t.Fatalf("a tight series: %+v", g)
	}
	// sorted .98 .99 1 1 1.01 1.02: IQR = s[4]-s[1] = 0.02
	if math.Abs(float64(g.IQR)-0.02) > 1e-9 {
		t.Fatalf("IQR %v, want 0.02", g.IQR)
	}
	for _, c := range []struct {
		rs   []float64
		want string
	}{
		{[]float64{1, 1.3, 0.8, 1.2, 0.9, 1.1}, "IQR"},
		{[]float64{1, 1, 1}, "fewer than 6"},
		{[]float64{1, 1, 1, 1, 1, 1, math.NaN()}, "no ratio"},
		{nil, "no round"},
	} {
		if g := Gate(c.rs); !strings.Contains(g.Refused, c.want) {
			t.Errorf("%v: refused %q, want %q", c.rs, g.Refused, c.want)
		}
	}
}

// TestSummariseCountsWhatItSays: two good requests and a failed one at a
// known wall time.
func TestSummariseCountsWhatItSays(t *testing.T) {
	ms := time.Millisecond
	ss := []Sample{
		{TTFT: 100 * ms, ITL: []time.Duration{10 * ms, 10 * ms, 10 * ms}, Chunks: 4, CompletionTokens: 4, UsageSeen: true},
		{TTFT: 300 * ms, ITL: []time.Duration{30 * ms, 50 * ms}, Chunks: 3, CompletionTokens: 3, UsageSeen: true},
		{Err: "HTTP 500"},
	}
	c := Config{MaxTokens: 4, SLOTTFT: 200 * ms, SLOTPOT: 20 * ms}
	st := Summarise(ss, 2*time.Second, c)
	if st.Requests != 3 || st.Errors != 1 || st.CompletionTokens != 7 || st.AtMax != 1 {
		t.Fatalf("counts %+v", st)
	}
	if st.AggRate != 3.5 || st.Goodput != 0.5 {
		t.Fatalf("aggregate %v tok/s, goodput %v req/s; want 3.5 and 0.5", st.AggRate, st.Goodput)
	}
	if st.TTFTp50 != 200 || st.ITLp50 != 10 {
		t.Fatalf("TTFT p50 %v ITL p50 %v; want 200 and 10", st.TTFTp50, st.ITLp50)
	}
	// Request rates: 3 tokens over 30 ms = 100/s, 2 over 80 ms = 25/s.
	if math.Abs(float64(st.ReqRate)-62.5) > 1e-9 {
		t.Fatalf("request rate p50 %v, want 62.5", st.ReqRate)
	}
	// A level with no samples marshals; NaN is null, not an error.
	b, err := json.Marshal(Summarise(nil, 0, c))
	if err != nil || !strings.Contains(string(b), `"ttft_p50_ms":null`) {
		t.Fatalf("an empty level marshals as %s, %v", b, err)
	}
}
