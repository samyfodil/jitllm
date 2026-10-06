package model

import (
	"bytes"
	"encoding/json"
	"fmt"
	"slices"
	"strings"

	"github.com/samyfodil/jitllm/tok/jinja"
)

// ToolCall is one function call an assistant turn made. Arguments is the JSON
// object text, compact, which is what the OpenAI API carries as a string and
// what the Anthropic API carries as an object.
type ToolCall struct {
	ID, Name  string
	Arguments string
}

// toolFields adds a message's tool calls and tool-result identity in the shape
// HuggingFace templates read: tool_calls as [{id, type, function: {name,
// arguments}}] with arguments a MAPPING (Llama-3.1 pipes it through tojson,
// Qwen3 checks `is string` first), and a "tool" turn's tool_call_id and name.
// stringArgs hands the arguments as the JSON text the request carried instead,
// for a template that only reads that (renderChatTemplate).
func toolFields(mm map[string]any, x ChatMessage, stringArgs bool) error {
	if x.ToolCallID != "" {
		mm["tool_call_id"] = x.ToolCallID
	}
	if x.Name != "" {
		mm["name"] = x.Name
	}
	if len(x.ToolCalls) == 0 {
		return nil
	}
	calls := make([]any, len(x.ToolCalls))
	for i, c := range x.ToolCalls {
		var args any = c.Arguments
		if a := strings.TrimSpace(c.Arguments); a == "" {
			args = map[string]any{}
			if stringArgs {
				args = "{}"
			}
		} else if stringArgs {
			args = c.Arguments
		} else if v, err := jinja.FromJSON([]byte(a)); err == nil && v.IsDict() {
			args = v
		} else if err != nil {
			return fmt.Errorf("tool call %q: arguments are not JSON: %w", c.Name, err)
		}
		calls[i] = map[string]any{
			"id":       c.ID,
			"type":     "function",
			"function": map[string]any{"name": c.Name, "arguments": args},
		}
	}
	mm["tool_calls"] = calls
	return nil
}

// ParseToolCalls reads the tool calls out of a completion, returning the text
// that is not part of any call and the calls. names are the declared tools; a
// call to anything else is not a call and stays text. nil names accepts any.
//
// The format is the model's, and two shapes cover the families here:
//
//   - Hermes, which Qwen2.5, Qwen3 and most fine-tunes use:
//     <tool_call>{"name": ..., "arguments": {...}}</tool_call>, any number of
//     them, anywhere in the text.
//   - A bare JSON call opening the reply: Llama-3.1/3.2 write
//     {"name": ..., "parameters": {...}}, Mistral v0.3 writes
//     [TOOL_CALLS][{"name": ..., "arguments": {...}}] -- and [TOOL_CALLS] is a
//     CONTROL token the detokenizer drops, so what reaches this is the array.
//
// The declared-name check is what keeps a model asked to answer IN JSON from
// having its answer taken for a call.
func ParseToolCalls(text string, names []string) (string, []ToolCall) {
	known := func(n string) bool { return n != "" && (names == nil || slices.Contains(names, n)) }
	if strings.Contains(text, hermesOpen) {
		var content strings.Builder
		var calls []ToolCall
		rest := text
		for {
			i := strings.Index(rest, hermesOpen)
			if i < 0 {
				content.WriteString(rest)
				break
			}
			content.WriteString(rest[:i])
			body := rest[i+len(hermesOpen):]
			j := strings.Index(body, hermesClose)
			next := ""
			if j >= 0 {
				body, next = body[:j], body[j+len(hermesClose):]
			}
			body = strings.TrimSpace(body)
			c, ok := decodeCall([]byte(body))
			// Qwen2.5's own template shows the call as
			// {{"name": <function-name>, "arguments": ...}} -- a doubled brace
			// transformers renders verbatim -- and the model copies it.
			// Qwen2.5-1.5B doubles only the opener, so both are tried.
			if !ok && strings.HasPrefix(body, "{{") {
				if c, ok = decodeCall([]byte(body[1:])); !ok && strings.HasSuffix(body, "}}") {
					c, ok = decodeCall([]byte(body[1 : len(body)-1]))
				}
			}
			if !ok || !known(c.Name) {
				return text, nil
			}
			calls = append(calls, c)
			if j < 0 {
				break
			}
			rest = next
		}
		return strings.TrimSpace(content.String()), calls
	}
	t := strings.TrimSpace(text)
	if t == "" || (t[0] != '{' && t[0] != '[') {
		return text, nil
	}
	d := json.NewDecoder(strings.NewReader(t))
	var calls []ToolCall
	for {
		var raw json.RawMessage
		if err := d.Decode(&raw); err != nil {
			break
		}
		var list []json.RawMessage
		if raw[0] == '[' {
			if json.Unmarshal(raw, &list) != nil {
				return text, nil
			}
		} else {
			list = []json.RawMessage{raw}
		}
		for _, r := range list {
			c, ok := decodeCall(r)
			if !ok || !known(c.Name) {
				return text, nil
			}
			calls = append(calls, c)
		}
		// Llama-3.1 separates parallel calls with ';'.
		rest := strings.TrimLeft(t[d.InputOffset():], " \t\r\n;")
		if rest == "" || (rest[0] != '{' && rest[0] != '[') {
			return rest, calls
		}
		d = json.NewDecoder(strings.NewReader(rest))
		t = rest
	}
	if len(calls) == 0 {
		return text, nil
	}
	return strings.TrimSpace(t[d.InputOffset():]), calls
}

const hermesOpen, hermesClose = "<tool_call>", "</tool_call>"

// decodeCall reads {"name", "arguments"|"parameters"}; arguments may be an
// object or a string holding one (Qwen emits both).
func decodeCall(b []byte) (ToolCall, bool) {
	var c struct {
		Name       string          `json:"name"`
		Arguments  json.RawMessage `json:"arguments"`
		Parameters json.RawMessage `json:"parameters"`
	}
	if json.Unmarshal(b, &c) != nil || c.Name == "" {
		return ToolCall{}, false
	}
	a := c.Arguments
	if len(a) == 0 {
		a = c.Parameters
	}
	var s string
	if json.Unmarshal(a, &s) == nil {
		a = json.RawMessage(s)
	}
	if len(bytes.TrimSpace(a)) == 0 {
		a = json.RawMessage("{}")
	}
	var buf bytes.Buffer
	if json.Compact(&buf, a) != nil || buf.Bytes()[0] != '{' {
		return ToolCall{}, false
	}
	return ToolCall{Name: c.Name, Arguments: buf.String()}, true
}

// ToolCallHold is where a stream must stop emitting text because a tool call
// has begun, or may be beginning, at that byte of text: -1 when nothing needs
// holding. Text before it is safe to send; ParseToolCalls decides the rest at
// the end.
func ToolCallHold(text string) int {
	if i := strings.Index(text, hermesOpen); i >= 0 {
		return i
	}
	t := strings.TrimLeft(text, " \t\r\n")
	if t == "" || t[0] == '{' || t[0] == '[' {
		return len(text) - len(t)
	}
	for k := len(hermesOpen) - 1; k > 0; k-- {
		if strings.HasSuffix(text, hermesOpen[:k]) {
			return len(text) - k
		}
	}
	return -1
}
