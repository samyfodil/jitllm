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
// a V100 (httpbench at 64 concurrent requests, Llama 3.1 8B) and then failed
// 172 of 256 requests there: 64 completions at once over HTTP, twice -- the
// second wave on the pooled States the first left -- with the card's budget
// held below what their histories take. Admission (stepLoop.kvRoom) keeps the
// admitted rows' histories on the card and holds the rest back until rows
// retire, so the rows that run step together: every request completes, the
// budget bound (admissions were put off), and joint steps ran in both waves.
// Before admission every row was admitted, the oldest pages of most sessions
// went home, and a step with two sequences' pages at home is refused as one
// step, so each session ran alone and re-streamed its evicted prefix every
// token, a Sync a pass.
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
	st := &lm.loop.stats
	for wave := range 2 {
		j0, w0 := st.jointSteps.Load(), st.kvWaits.Load()
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
					t.Errorf("wave %d, request %d: %v", wave, i, err)
				}
			}
		}
		ts := lm.gpu.Stats()
		joint, waits := st.jointSteps.Load()-j0, st.kvWaits.Load()-w0
		t.Logf("wave %d: %d of %d requests failed; %d joint steps, %d admissions put off for room, "+
			"%d refusals in all; %d evictions, %d streamed passes in all",
			wave, failed, n, joint, waits, st.refusals.Load(), ts.KVEvictions, ts.KVStreamPasses)
		if failed > 0 {
			t.Fatalf("wave %d: %d of %d requests failed", wave, failed, n)
		}
		if joint == 0 {
			t.Fatalf("wave %d: no joint step ran: the rows stepped one session at a time", wave)
		}
		if waits == 0 {
			t.Fatalf("wave %d: no admission was put off: the budget never bound and the gate proved nothing", wave)
		}
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
