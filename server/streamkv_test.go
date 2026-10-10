package server

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"

	"github.com/samyfodil/jitllm/jit/gpu/tier"
)

// TestSixtyFourRequestsWhoseHistoryStreams is the load that killed jitllmd on
// a V100 (httpbench at 64 concurrent requests, Llama 3.1 8B): 64 completions
// at once over HTTP, with the card's budget held below what their histories
// take, so the step loop's sessions send their oldest pages home and stream
// them back. Every request completes: 0 errors, and the tier counted
// evictions and streamed passes, or the budget never bound and the gate
// proved nothing.
//
// A second wave on the pooled States -- the restart over history at home that
// crashed -- is engine/model's TestManySessionsStreamTheirHistory. Here it
// does not finish in fifteen minutes: with most sessions' pages at home every
// joint step is refused (two sequences with pages at home in one call) and
// each session streams its evicted prefix alone, a Sync per pass.
func TestSixtyFourRequestsWhoseHistoryStreams(t *testing.T) {
	const name = "Llama-3.2-1B-Instruct-Q4_K_M.jlm"
	path := modelPath(t, name)
	e := New(Config{Probe: oneCardProbe, Version: "test", DefaultMaxSeq: 1024, MaxBatchRows: 64})
	t.Cleanup(e.Close)
	lm, err := e.LoadModel(LoadOptions{Path: path, ModelID: "dev", DeviceIDs: []string{"gpu:0"},
		tierConfig: func(c *tier.Config) { c.KVPage = 64 }})
	if err != nil {
		t.Fatalf("loading %s onto -devices gpu:0: %v", name, err)
	}
	if lm.loop == nil || lm.gpu == nil {
		t.Fatal("a model loaded onto a device has no step loop")
	}
	requireWholeOnDevice(t, e, lm)
	c := serveEngine(t, e)
	if _, err := lm.gpu.SetBudget(lm.gpu.Stats().BudgetUsed + 1<<30); err != nil {
		t.Fatal(err)
	}
	words := strings.Fields(strings.Repeat("the clerk counted barrels of salt on the upper floor while rain fell ", 50))
	const n = 64
	errs := make([]error, n)
	var wg sync.WaitGroup
	for i := range n {
		wg.Add(1)
		go func() {
			defer wg.Done()
			w := 128
			if i%2 == 1 {
				w = 512
			}
			errs[i] = complete64(c.url, strings.Join(words[:w], " "))
		}()
	}
	wg.Wait()
	failed := 0
	for i, err := range errs {
		if err != nil {
			failed++
			if failed <= 3 {
				t.Errorf("request %d: %v", i, err)
			}
		}
	}
	st := lm.gpu.Stats()
	t.Logf("%d of %d requests failed; %d evictions, %d streamed passes", failed, len(errs), st.KVEvictions, st.KVStreamPasses)
	if failed > 0 {
		t.Fatalf("%d of %d requests failed", failed, len(errs))
	}
	if st.KVEvictions == 0 || st.KVStreamPasses == 0 {
		t.Fatalf("the budget never bound: %d evictions, %d streamed passes", st.KVEvictions, st.KVStreamPasses)
	}
}

// complete64 posts one greedy 128-token completion of prompt and reports
// anything but a 200 carrying a choice and its tokens.
func complete64(url, prompt string) error {
	// The prompt is plain lowercase words: %q is its JSON string.
	body := fmt.Sprintf(`{"model":"dev","prompt":%q,"max_tokens":128,"temperature":0,"ignore_eos":true}`, prompt)
	resp, err := http.Post(url+"/v1/completions", "application/json", strings.NewReader(body))
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	var out struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	if resp.StatusCode != 200 || json.Unmarshal(b, &out) != nil || len(out.Choices) != 1 ||
		out.Usage.CompletionTokens == 0 {
		return fmt.Errorf("status %d: %s", resp.StatusCode, b)
	}
	return nil
}
