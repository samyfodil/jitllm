package server

import (
	"encoding/json"

	"github.com/samyfodil/jitllm/engine/model"
)

// toolText sits between the engine's token stream and a shim's output when a
// request declares tools. Text before a tool call streams as it arrives; from
// the first byte that is -- or may be becoming -- a call, it is held, and
// finish parses what was held. Both shims use it, so they agree on what is a
// call.
type toolText struct {
	on    bool
	names []string
	buf   string
	sent  int
}

func newToolText(tools []byte) *toolText {
	t := &toolText{on: len(tools) > 0}
	if t.on {
		t.names = toolNames(tools)
	}
	return t
}

// push takes one token's text and returns what is safe to emit now.
func (t *toolText) push(s string) string {
	if !t.on {
		return s
	}
	t.buf += s
	limit := len(t.buf)
	if h := model.ToolCallHold(t.buf); h >= 0 {
		limit = max(h, t.sent)
	}
	out := t.buf[t.sent:limit]
	t.sent = limit
	return out
}

// finish returns the text a stream still owes (the held text, when it turned
// out not to be a call), the whole completion's non-call text, and the calls.
func (t *toolText) finish() (tail, content string, calls []model.ToolCall) {
	if !t.on {
		return "", t.buf, nil
	}
	content, calls = model.ParseToolCalls(t.buf, t.names)
	if len(calls) == 0 {
		return t.buf[t.sent:], t.buf, nil
	}
	return "", content, calls
}

// toolNames reads the declared function names out of an OpenAI-shaped list.
func toolNames(tools []byte) []string {
	var ts []struct {
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if json.Unmarshal(tools, &ts) != nil {
		return nil
	}
	names := make([]string, 0, len(ts))
	for _, x := range ts {
		names = append(names, x.Function.Name)
	}
	return names
}
