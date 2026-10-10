package server

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/format/jlm"
)

// /v1/systemone's wire shape, against a scripted backend: what reaches Decide
// from a TypeSafe request, and the answers' TypeSafe shape on the way out.

func TestSystemOneSpeaksTypeSafe(t *testing.T) {
	var got DecideOptions
	b := &fakeBackend{decide: func(o DecideOptions) (*DecideResult, error) {
		got = o
		ans := make([]model.DecisionAnswer, len(o.Questions))
		for i, q := range o.Questions {
			ans[i] = model.DecisionAnswer{Type: q.Type}
			switch q.Type {
			case jlm.QuestionNoul:
				ans[i].Noul = 0.25
			case jlm.QuestionChoice:
				ans[i].Probs = []float64{0.1, 0.9}
				ans[i].Choice = q.Options[1].Key
				ans[i].Confidence = 0.8
			case jlm.QuestionScore:
				ans[i].Probs = []float64{0.5, 0.25, 0.25}
				ans[i].Score = 0.75
				ans[i].Confidence = 0.5
			}
		}
		return &DecideResult{ModelID: "m1", Answers: ans, InputTokens: 42}, nil
	}}
	srv := httptest.NewServer(CompatHandler(b))
	defer srv.Close()
	body := `{"model": "lev", "state": {"z": 1, "a": "two"}, "questions": {
		"urgent": {"type": "noul", "instructions": "Is it urgent?", "criteria": {"true": "now"}},
		"team": {"type": "choice", "criteria": {"billing": "money", "tech": null}},
		"mood": {"type": "score", "instructions": "How angry?", "criteria": ["calm", {"level": "mid"}, "angry"]}}}`
	resp, err := http.Post(srv.URL+"/v1/systemone", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d: %s", resp.StatusCode, raw)
	}
	// The request reached Decide in its own order, keys and all.
	if got.Model != "lev" || got.State.JSON() != `{"z": 1, "a": "two"}` {
		t.Errorf("model %q, state %s", got.Model, got.State.JSON())
	}
	var ids []string
	for _, q := range got.Questions {
		ids = append(ids, q.ID)
	}
	if strings.Join(ids, ",") != "urgent,team,mood" {
		t.Errorf("questions arrived as %v", ids)
	}
	if q := got.Questions[0]; len(q.Options) != 2 || q.Options[0].Key != "false" || !q.Options[0].Description.IsNone() ||
		q.Options[1].Description.JSON() != `"now"` {
		t.Errorf("the noul's options are %+v", q.Options)
	}
	if !got.Questions[1].Instructions.IsNone() {
		t.Errorf("an absent instruction arrived as %s", got.Questions[1].Instructions.JSON())
	}
	// The answers in order and in TypeSafe's shape.
	want := `{"model":"m1","answers":{"urgent":{"type":"noul","noul":0.25},` +
		`"team":{"type":"choice","choice":"tech","confidence":0.8,"probabilities":{"billing":0.1,"tech":0.9}},` +
		`"mood":{"type":"score","score":0.75,"confidence":0.5,"legend":{"0":"calm","1":{"level": "mid"},"2":"angry"},` +
		`"probabilities":{"0":0.5,"1":0.25,"2":0.25}}},"usage":{"input_tokens":42,"output_tokens":0}}`
	if string(raw) != want {
		t.Errorf("response\n%s\nwant\n%s", raw, want)
	}
	var check map[string]any
	if err := json.Unmarshal(raw, &check); err != nil {
		t.Errorf("the response is not JSON: %v", err)
	}
}

func TestSystemOneRefusesInFastAPIShape(t *testing.T) {
	b := &fakeBackend{decide: func(o DecideOptions) (*DecideResult, error) {
		t.Errorf("a malformed request reached Decide")
		return nil, nil
	}}
	srv := httptest.NewServer(CompatHandler(b))
	defer srv.Close()
	for _, c := range []struct{ name, body, loc string }{
		{"no state", `{"questions": {"a": {"type": "noul"}}}`, "state"},
		{"no questions", `{"state": "x"}`, "questions"},
		{"empty questions", `{"state": "x", "questions": {}}`, "questions"},
		{"bad type", `{"state": "x", "questions": {"a": {"type": "maybe"}}}`, "questions"},
		{"choice without criteria", `{"state": "x", "questions": {"a": {"type": "choice"}}}`, "questions"},
		{"score as object", `{"state": "x", "questions": {"a": {"type": "score", "criteria": {"x": 1}}}}`, "questions"},
	} {
		t.Run(c.name, func(t *testing.T) {
			resp, err := http.Post(srv.URL+"/v1/systemone", "application/json", strings.NewReader(c.body))
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()
			var e struct {
				Detail []s1Detail `json:"detail"`
			}
			if err := json.NewDecoder(resp.Body).Decode(&e); err != nil {
				t.Fatal(err)
			}
			if resp.StatusCode != http.StatusUnprocessableEntity || len(e.Detail) != 1 ||
				len(e.Detail[0].Loc) < 2 || e.Detail[0].Loc[1] != c.loc {
				t.Errorf("status %d, detail %+v; want 422 at body.%s", resp.StatusCode, e.Detail, c.loc)
			}
		})
	}
}
