package server

import (
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// serveMetrics is /metrics: the engine's counters in Prometheus's text
// exposition format (version 0.0.4), written with the standard library. It
// reads the same counters GetStats does; it keeps no state of its own.
//
// Per model: sessions, generates, tokens generated and prefilled, prefill
// and decode time as summaries (sum and count, no quantiles: the engine keeps
// totals, not samples), and the pager's resident bytes, budget, page-ins,
// evictions and bytes read. Per session: KV bytes as of its last snapshot.
func (e *Engine) serveMetrics(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	var b strings.Builder
	models := e.Models()
	sessions := e.Sessions()

	family(&b, "jitllm_models_loaded", "gauge", "Models loaded.")
	fmt.Fprintf(&b, "jitllm_models_loaded %d\n", len(models))
	family(&b, "jitllm_sessions", "gauge", "Open sessions, per model.")
	perModel := map[string]int{}
	for _, s := range sessions {
		perModel[s.modelID]++
	}
	for _, lm := range models {
		fmt.Fprintf(&b, "jitllm_sessions{model=%s} %d\n", quote(lm.id), perModel[lm.id])
	}

	type metric struct {
		name, kind, help string
		val              func(lm *LoadedModel) float64
	}
	for _, m := range []metric{
		{"jitllm_generates_total", "counter", "Generates finished.",
			func(lm *LoadedModel) float64 { return float64(lm.generates.Load()) }},
		{"jitllm_tokens_generated_total", "counter", "Tokens decoded.",
			func(lm *LoadedModel) float64 { return float64(lm.tokensGenerated.Load()) }},
		{"jitllm_tokens_prefilled_total", "counter", "Prompt tokens prefilled.",
			func(lm *LoadedModel) float64 { return float64(lm.tokensPrefilled.Load()) }},
		{"jitllm_host_resident_bytes", "gauge", "Weight bytes resident on the host.",
			func(lm *LoadedModel) float64 { return float64(lm.m.HostBytes()) }},
		{"jitllm_page_budget_bytes", "gauge", "The pager's budget.",
			func(lm *LoadedModel) float64 { return float64(lm.m.PageBudget()) }},
		{"jitllm_page_ins_total", "counter", "Pages read in.",
			func(lm *LoadedModel) float64 { _, in, _ := lm.m.PageStats(); return float64(in) }},
		{"jitllm_page_evictions_total", "counter", "Pages evicted.",
			func(lm *LoadedModel) float64 { _, _, out := lm.m.PageStats(); return float64(out) }},
		{"jitllm_page_read_bytes_total", "counter", "Bytes the pager read from the container.",
			func(lm *LoadedModel) float64 { return float64(lm.m.BytesRead()) }},
	} {
		family(&b, m.name, m.kind, m.help)
		for _, lm := range models {
			fmt.Fprintf(&b, "%s{model=%s} %g\n", m.name, quote(lm.id), m.val(lm))
		}
	}
	for _, s := range []struct {
		name, help string
		nanos      func(lm *LoadedModel) int64
	}{
		{"jitllm_prefill_seconds", "Each generate's prefill: its time to first token, queueing aside.",
			func(lm *LoadedModel) int64 { return lm.prefillNanos.Load() }},
		{"jitllm_decode_seconds", "Each generate's decode.",
			func(lm *LoadedModel) int64 { return lm.decodeNanos.Load() }},
	} {
		family(&b, s.name, "summary", s.help)
		for _, lm := range models {
			fmt.Fprintf(&b, "%s_sum{model=%s} %g\n", s.name, quote(lm.id), time.Duration(s.nanos(lm)).Seconds())
			fmt.Fprintf(&b, "%s_count{model=%s} %d\n", s.name, quote(lm.id), lm.generates.Load())
		}
	}
	family(&b, "jitllm_session_kv_bytes", "gauge", "A session's KV history bytes, as of its last snapshot.")
	for _, s := range sessions {
		fmt.Fprintf(&b, "jitllm_session_kv_bytes{model=%s,session=%s} %d\n", quote(s.modelID), quote(s.id), s.snapKV.Load())
	}
	io.WriteString(w, b.String())
}

// family writes a metric family's HELP and TYPE lines.
func family(b *strings.Builder, name, kind, help string) {
	fmt.Fprintf(b, "# HELP %s %s\n# TYPE %s %s\n", name, help, name, kind)
}

// quote is a label value as the exposition format escapes it.
func quote(s string) string {
	return `"` + strings.NewReplacer(`\`, `\\`, `"`, `\"`, "\n", `\n`).Replace(s) + `"`
}
