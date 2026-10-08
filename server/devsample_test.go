package server

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/samyfodil/jitllm/engine/model"
)

// TestTheRowComesHomeWhenTheReplyReadsIt: with the head on the device, a
// sampled generate that wants no row after the prompt selects its candidates
// there (model.State.ForwardSample), and one that asks for logprobs or runs
// under a grammar reads every row back and samples on the host -- those read
// the raw logits, which a device-selected token never brings home. Asserted
// on the session's own count of device-selected tokens, so a generate that
// silently skipped the device path cannot pass the first half, and one that
// took it under a grammar or logprobs cannot pass the second.
func TestTheRowComesHomeWhenTheReplyReadsIt(t *testing.T) {
	path := modelPath(t, instructModel)
	e := New(Config{ModelDir: filepath.Dir(path), Version: "test", MaxBatchRows: 1})
	defer e.Close()
	if _, err := e.LoadModel(LoadOptions{Path: path, ModelID: "inst", DeviceIDs: []string{"gpu:0"}, Sessions: 1}); err != nil {
		t.Skipf("NO DEVICE: %v -- this gate proved nothing", err)
	}
	sm := model.Sampler{Temp: 0.9, TopK: 40, TopP: 0.9, Seed: 3}
	s, err := e.CreateSession(SessionOptions{ModelID: "inst", SessionID: "a", MaxSeq: 256, Sampling: sm})
	if err != nil {
		t.Fatal(err)
	}
	onDev := func() (n int64, head bool) {
		s.Inspect(func(st *model.State) { n, head = st.SampledOnDevice, st.HeadOnDevice() })
		return n, head
	}
	if _, head := onDev(); !head {
		t.Skip("CARD TOO SMALL: the head is not on the device -- this gate proved nothing")
	}
	gen := func(o GenerateOptions) (tokens, logprobs int) {
		t.Helper()
		o.SessionID, o.Prompt, o.MaxTokens, o.IgnoreEOS = "a", Prompt{Kind: PromptText, Text: "The capital of France is"}, 12, o.Grammar == ""
		err := e.Generate(context.Background(), o, func(ev Event) error {
			if ev.Kind == EventToken && ev.Token.ID >= 0 {
				tokens++
				if ev.Token.Logprob != nil {
					logprobs++
				}
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		return tokens, logprobs
	}

	n0, _ := onDev()
	tok, _ := gen(GenerateOptions{})
	n1, _ := onDev()
	if n1-n0 < int64(tok-1) || tok < 2 {
		t.Fatalf("a plain sampled generate of %d tokens selected %d on the device", tok, n1-n0)
	}
	tok, lps := gen(GenerateOptions{Logprobs: true})
	n2, _ := onDev()
	if n2 != n1 || lps != tok {
		t.Fatalf("logprobs: %d tokens selected on the device, %d of %d tokens carry a logprob", n2-n1, lps, tok)
	}
	tok, _ = gen(GenerateOptions{Grammar: `root ::= "Paris" | "Lyon"` + "\n"})
	n3, _ := onDev()
	if n3 != n2 || tok == 0 {
		t.Fatalf("grammar: %d of %d tokens selected on the device", n3-n2, tok)
	}
	t.Logf("plain: %d on the device; logprobs and grammar: none", n1-n0)
}
