package server

import (
	"context"
	"strings"
	"testing"

	"github.com/samyfodil/jitllm/dev/httpbench/hb"
)

// TestHTTPBenchAgainstTheRealServer runs dev/httpbench's harness against a
// real engine behind the real handler, over HTTP/1.1 and HTTP/2 (h2c), at
// concurrency 1 to 4, as an A/A pair. It holds the harness to what the
// server said: every request parsed to [DONE], its token count is the
// stream's usage and is max_tokens (ignore_eos), and the A/A ratios are
// computed and refused for having fewer than six rounds.
func TestHTTPBenchAgainstTheRealServer(t *testing.T) {
	_, _, c := loadedEngine(t, smallModel, "small", LoadOptions{})
	for _, h2 := range []bool{false, true} {
		cfg := hb.Config{
			Arms: []hb.Arm{
				{Name: "A", URL: c.url, HTTP2: h2},
				{Name: "B", URL: c.url, HTTP2: h2},
			},
			Models:     []hb.Model{{Name: "small", Weight: 1}},
			Levels:     []int{1, 2, 4},
			Requests:   4,
			Rounds:     2,
			Warmup:     1,
			PromptLens: []int{4, 12},
			MaxTokens:  8,
		}
		res, err := hb.Run(context.Background(), cfg, nil)
		if err != nil {
			t.Fatalf("http2=%v: %v", h2, err)
		}
		if len(res.Levels) != 3*2*2 {
			t.Fatalf("http2=%v: %d levels ran, want 12", h2, len(res.Levels))
		}
		for _, lv := range res.Levels {
			for _, s := range lv.Samples {
				if s.Err != "" {
					t.Fatalf("http2=%v c=%d: %s", h2, lv.Level, s.Err)
				}
				if !s.UsageSeen || s.CompletionTokens != 8 || s.PromptTokens < 4 || s.Chunks == 0 || s.TTFT <= 0 ||
					s.FinishReason != "length" || s.Model != "small" {
					t.Fatalf("http2=%v c=%d: sample %+v", h2, lv.Level, s)
				}
				if want := map[bool]string{false: "HTTP/1.1", true: "HTTP/2.0"}[h2]; s.Proto != want {
					t.Fatalf("http2=%v: the response came over %s", h2, s.Proto)
				}
				if len(s.ITL) != s.Chunks-1 {
					t.Fatalf("http2=%v: %d gaps between %d chunks", h2, len(s.ITL), s.Chunks)
				}
			}
		}
		rep := hb.Reduce(res)
		if len(rep.Rows) != 6 || len(rep.Comparisons) != 3 {
			t.Fatalf("http2=%v: %d rows, %d comparisons", h2, len(rep.Rows), len(rep.Comparisons))
		}
		for _, r := range rep.Rows {
			if r.All.Requests != 8 || r.All.Errors != 0 || r.All.AtMax != 8 || r.All.CompletionTokens != 64 || r.All.AggRate <= 0 {
				t.Fatalf("http2=%v: row %+v", h2, r)
			}
		}
		for _, cmp := range rep.Comparisons {
			if cmp.AggRate.N != 2 || !strings.Contains(cmp.AggRate.Refused, "fewer than 6") {
				t.Fatalf("http2=%v: a two-round ratio was not refused: %+v", h2, cmp.AggRate)
			}
		}
	}
}
