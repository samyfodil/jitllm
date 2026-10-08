package server

import (
	"encoding/json"
	"fmt"
	"sync"

	"github.com/samyfodil/jitllm/engine/grammar"
)

// Structured output: a generate constrained to a grammar (GBNF, or a JSON
// Schema compiled to it). Each step the tokens the grammar does not allow go
// to -inf in a copy of the logits before the sampler reads them, and the
// sampled token advances the grammar. A grammar is compiled, and its states'
// allowed sets kept, once per (model, grammar text): JSON revisits its states,
// so after the first few fields a step is a lookup.

// grammars is a model's compiled grammars, by their text. It is bounded: a
// server fed endless distinct schemas drops them all and starts again rather
// than growing.
type grammars struct {
	mu sync.Mutex
	m  map[string]*grammar.Matcher
	// toks is the model's tokenizer indexed for the masks, built on the
	// first grammar and shared by every one.
	toks *grammar.Tokens
}

// maxGrammars is how many compiled grammars a model keeps.
const maxGrammars = 64

func (lm *LoadedModel) matcher(src string) (*grammar.Matcher, error) {
	g := &lm.grammars
	g.mu.Lock()
	defer g.mu.Unlock()
	if m, ok := g.m[src]; ok {
		return m, nil
	}
	gr, err := grammar.Parse(src)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalid, err)
	}
	if g.m == nil || len(g.m) >= maxGrammars {
		g.m = map[string]*grammar.Matcher{}
	}
	if g.toks == nil {
		g.toks = grammar.IndexTokens(lm.m.Vocab)
	}
	m := grammar.NewMatcher(gr, g.toks)
	g.m[src] = m
	return m, nil
}

// constraint is one generate's place in its grammar.
type constraint struct {
	m  *grammar.Matcher
	st grammar.State
	// lg is the masked copy of the step's logits.
	lg []float32
}

// mask is logits with every token the grammar does not allow at -inf, in
// the constraint's own copy.
func (c *constraint) mask(logits []float32) []float32 {
	if cap(c.lg) < len(logits) {
		c.lg = make([]float32, len(logits))
	}
	c.lg = c.lg[:len(logits)]
	copy(c.lg, logits)
	c.m.Mask(c.st, c.lg)
	return c.lg
}

// accept advances past tok; false where the grammar does not allow it.
func (c *constraint) accept(tok int32) bool {
	st, ok := c.m.Accept(c.st, tok)
	if ok {
		c.st = st
	}
	return ok
}

// oaResponseFormat is OpenAI's response_format as a grammar's text: none for
// "text" or an absent field, any JSON object for "json_object", and the
// schema's grammar for "json_schema". A schema the converter does not build
// is an ErrInvalid naming what.
func oaResponseFormat(raw json.RawMessage) (string, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return "", nil
	}
	var rf struct {
		Type       string `json:"type"`
		JSONSchema *struct {
			Name   string          `json:"name"`
			Schema json.RawMessage `json:"schema"`
			Strict *bool           `json:"strict"`
		} `json:"json_schema"`
	}
	if err := json.Unmarshal(raw, &rf); err != nil {
		return "", fmt.Errorf("%w: response_format: %v", ErrInvalid, err)
	}
	switch rf.Type {
	case "text":
		return "", nil
	case "json_object":
		return grammar.JSONObject(), nil
	case "json_schema":
		if rf.JSONSchema == nil || len(rf.JSONSchema.Schema) == 0 {
			return "", fmt.Errorf("%w: response_format json_schema needs json_schema.schema", ErrInvalid)
		}
		src, err := grammar.FromJSONSchema(rf.JSONSchema.Schema)
		if err != nil {
			return "", fmt.Errorf("%w: response_format: %v", ErrInvalid, err)
		}
		return src, nil
	}
	return "", fmt.Errorf("%w: response_format type %q: text, json_object or json_schema", ErrInvalid, rf.Type)
}
