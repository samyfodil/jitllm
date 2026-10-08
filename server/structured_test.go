package server

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"slices"
	"strings"
	"testing"

	"github.com/samyfodil/jitllm/engine/grammar"
	"github.com/samyfodil/jitllm/engine/model"
	"github.com/samyfodil/jitllm/internal/schemacheck"
)

// The structured-output gates. Every output, sampled at temperature 1 over
// many seeds, parses and validates against its schema -- held by a validator
// written apart from the grammar (internal/schemacheck); a grammar that
// allows everything leaves greedy decode as it was; and the mask left off one
// step is caught.

// structuredModels are a tiny base model with a 128-token context and a real
// instruct model.
var structuredModels = []struct {
	file, id string
	schemas  []string
	maxTok   int32
	prompt   string
}{
	{"stories15M-q8_0.jlm", "s15", []string{
		`{"type":"object","properties":{"color":{"enum":["red","green","blue"]},"ok":{"type":"boolean"}},` +
			`"required":["color","ok"]}`,
		`{"type":"array","items":{"type":"integer"},"minItems":1,"maxItems":3}`,
	}, 96, "Once upon a time"},
	{"Llama-3.2-1B-Instruct-Q4_K_M.jlm", "llama1b", []string{
		`{"type":"object","properties":{"name":{"type":"string","maxLength":16},"age":{"type":"integer"},` +
			`"pets":{"type":"array","items":{"type":"object","properties":{"kind":{"enum":["cat","dog"]},` +
			`"legs":{"type":"number"}},"required":["kind"]},"maxItems":2}},"required":["name","age"]}`,
		`{"anyOf":[{"type":"string","maxLength":10},{"type":"null"},{"type":"boolean"}]}`,
	}, 400, "Describe a person as JSON."},
}

// structuredSeeds is how many sampled outputs each schema is held to.
const structuredSeeds = 24

// genText runs one generate through the engine and returns its text and
// finish reason.
func genText(t *testing.T, e *Engine, o GenerateOptions) (string, []int32, FinishReason) {
	t.Helper()
	var b strings.Builder
	var toks []int32
	var reason FinishReason
	err := e.Generate(context.Background(), o, func(ev Event) error {
		switch ev.Kind {
		case EventToken:
			b.WriteString(ev.Token.Text)
			if ev.Token.ID >= 0 {
				toks = append(toks, ev.Token.ID)
			}
		case EventFinished:
			reason = ev.Finished.Reason
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return b.String(), toks, reason
}

func TestStructuredOutputValidates(t *testing.T) {
	for _, sm := range structuredModels {
		t.Run(sm.id, func(t *testing.T) {
			e, _, _ := loadedEngine(t, sm.file, sm.id, LoadOptions{})
			for _, schemaText := range sm.schemas {
				src, err := grammar.FromJSONSchema([]byte(schemaText))
				if err != nil {
					t.Fatal(err)
				}
				var schema any
				if err := json.Unmarshal([]byte(schemaText), &schema); err != nil {
					t.Fatal(err)
				}
				distinct := map[string]bool{}
				for seed := range uint64(structuredSeeds) {
					out, _, reason := genText(t, e, GenerateOptions{ModelID: sm.id, Prompt: Prompt{Kind: PromptText,
						Text: sm.prompt}, MaxTokens: int(sm.maxTok), Grammar: src,
						Sampling: &model.Sampler{Temp: 1, Seed: int64(seed) + 1}})
					var doc any
					if err := schemacheck.Parse(out, &doc); err != nil {
						t.Fatalf("seed %d: %q does not parse (%v, finish %v)", seed, out, err, reason)
					}
					if err := schemacheck.Validate(schema, schema, doc); err != nil {
						t.Fatalf("seed %d: %s does not validate: %v", seed, out, err)
					}
					if reason != FinishEOS {
						t.Fatalf("seed %d: %q finished %v, not on the grammar's end", seed, out, reason)
					}
					distinct[out] = true
				}
				// The selection check: a mask that let one output through
				// every seed would pass the rest trivially.
				if len(distinct) < structuredSeeds/4 {
					t.Fatalf("%d distinct outputs of %d seeds: the sampling is not exercised", len(distinct),
						structuredSeeds)
				}
				t.Logf("%s: %d seeds valid, %d distinct, e.g. %s", schemaText[:min(40, len(schemaText))],
					structuredSeeds, len(distinct), firstKey(distinct))
			}
		})
	}
}

func firstKey(m map[string]bool) string {
	ks := make([]string, 0, len(m))
	for k := range m {
		ks = append(ks, k)
	}
	slices.Sort(ks)
	return ks[0]
}

// TestAnAllowingGrammarIsPlainGreedy: a grammar that allows any text leaves
// greedy decode token for token as it was.
func TestAnAllowingGrammarIsPlainGreedy(t *testing.T) {
	for _, sm := range structuredModels {
		t.Run(sm.id, func(t *testing.T) {
			e, _, _ := loadedEngine(t, sm.file, sm.id, LoadOptions{})
			o := GenerateOptions{ModelID: sm.id, Prompt: Prompt{Kind: PromptText, Text: sm.prompt}, MaxTokens: 48}
			_, want, _ := genText(t, e, o)
			o.Grammar = "root ::= .*\n"
			_, got, _ := genText(t, e, o)
			if !slices.Equal(got, want) {
				t.Fatalf("under an allowing grammar %v, plain greedy %v", got, want)
			}
			if len(want) < 8 {
				t.Fatalf("%d tokens: too short to show anything", len(want))
			}
		})
	}
}

// TestStructuredGateDiscriminates leaves the mask off one step and demands an
// invalid output among the seeds: the validation gate can see the break.
func TestStructuredGateDiscriminates(t *testing.T) {
	sm := structuredModels[1]
	e, _, _ := loadedEngine(t, sm.file, sm.id, LoadOptions{})
	schemaText := sm.schemas[0]
	src, err := grammar.FromJSONSchema([]byte(schemaText))
	if err != nil {
		t.Fatal(err)
	}
	var schema any
	if err := json.Unmarshal([]byte(schemaText), &schema); err != nil {
		t.Fatal(err)
	}
	for seed := range uint64(structuredSeeds) {
		out, _, _ := genText(t, e, GenerateOptions{ModelID: sm.id, Prompt: Prompt{Kind: PromptText, Text: sm.prompt},
			MaxTokens: int(sm.maxTok), Grammar: src, Sampling: &model.Sampler{Temp: 1, Seed: int64(seed) + 1}, maskSkip: 3})
		var doc any
		if err := schemacheck.Parse(out, &doc); err != nil {
			t.Logf("seed %d: the unmasked step gave %q, which does not parse -- caught", seed, out[:min(60, len(out))])
			return
		}
		if err := schemacheck.Validate(schema, schema, doc); err != nil {
			t.Logf("seed %d: the unmasked step gave %s: %v -- caught", seed, out, err)
			return
		}
	}
	t.Fatal("every output with an unmasked step still validated: the gate cannot see the mask")
}

// TestResponseFormatOverHTTP: OpenAI's response_format on
// /v1/chat/completions -- json_object and json_schema -- and a schema
// keyword the converter does not build refused with 400 naming it.
func TestResponseFormatOverHTTP(t *testing.T) {
	sm := structuredModels[1]
	_, _, c := loadedEngine(t, sm.file, sm.id, LoadOptions{})
	for seed := range 4 {
		out := postOAText(t, c.url, fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":"Give me a JSON `+
			`object about a city."}],"max_tokens":300,"temperature":1,"seed":%d,"response_format":`+
			`{"type":"json_object"}}`, sm.id, seed))
		var m map[string]any
		if err := json.Unmarshal([]byte(out), &m); err != nil {
			t.Fatalf("json_object: %q is not an object: %v", out, err)
		}
	}
	out := postOAText(t, c.url, fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":"Describe a person."}],`+
		`"max_tokens":400,"temperature":1,"seed":7,"response_format":{"type":"json_schema","json_schema":`+
		`{"name":"p","strict":true,"schema":%s}}}`, sm.id, sm.schemas[0]))
	var doc, schema any
	if err := schemacheck.Parse(out, &doc); err != nil {
		t.Fatalf("json_schema: %q: %v", out, err)
	}
	if err := json.Unmarshal([]byte(sm.schemas[0]), &schema); err != nil {
		t.Fatal(err)
	}
	if err := schemacheck.Validate(schema, schema, doc); err != nil {
		t.Fatalf("json_schema: %s: %v", out, err)
	}
	status, body := postRaw(t, c.url+"/v1/chat/completions", fmt.Sprintf(`{"model":%q,"messages":[{"role":"user",`+
		`"content":"x"}],"response_format":{"type":"json_schema","json_schema":{"name":"p","schema":`+
		`{"type":"string","pattern":"^a+$"}}}}`, sm.id))
	if status != 400 || !strings.Contains(body, "pattern") {
		t.Fatalf("an unsupported keyword: %d %s, want 400 naming pattern", status, body)
	}
}

// postRaw posts body to url and returns the status and the body.
func postRaw(t *testing.T, url, body string) (int, string) {
	t.Helper()
	resp, err := http.Post(url, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	return resp.StatusCode, string(b)
}
