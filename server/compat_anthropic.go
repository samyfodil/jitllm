package server

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/samyfodil/jitllm/engine/model"
)

// Anthropic-compatible HTTP: POST /v1/messages.
//
// The same adapter over Engine.Generate as the OpenAI shim; only the wire
// shapes differ:
//
//  1. system is a top-level field; Engine.encode prepends it as a message.
//  2. content is a string or a block list; SDKs send the list.
//  3. The stream is named SSE events (message_start, content_block_start,
//     content_block_delta, content_block_stop, message_delta, message_stop)
//     with no [DONE] sentinel: message_stop ends it.
//  4. usage is input_tokens/output_tokens.
//  5. stop_reason is "end_turn" | "max_tokens" | "stop_sequence".

type anMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

type anRequest struct {
	Model     string          `json:"model"`
	Messages  []anMessage     `json:"messages"`
	System    json.RawMessage `json:"system"`
	MaxTokens int             `json:"max_tokens"`

	Temperature   *float64 `json:"temperature"`
	TopP          *float64 `json:"top_p"`
	TopK          *int     `json:"top_k"`
	StopSequences []string `json:"stop_sequences"`
	Stream        bool     `json:"stream"`

	// Tools are rendered by the model's own template, converted to the
	// OpenAI shape it reads; tool_choice {"type": "none"} withholds them.
	Tools      []anTool        `json:"tools"`
	ToolChoice json.RawMessage `json:"tool_choice"`

	JitllmSession string `json:"jitllm_session,omitempty"`
	// JitllmPriority is "high" or "normal" (Engine.prioritise).
	JitllmPriority string `json:"jitllm_priority,omitempty"`
}

type anContentBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type anTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema"`
	// CacheControl marks a cache breakpoint (anCachePrompt).
	CacheControl json.RawMessage `json:"cache_control,omitempty"`
}

// anInBlock is a request content block: text, tool_use (an assistant's call)
// or tool_result (the user turn answering one).
type anInBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text"`
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Input     json.RawMessage `json:"input"`
	ToolUseID string          `json:"tool_use_id"`
	Content   json.RawMessage `json:"content"`
}

// anOutBlock is a response content block: text, or a tool_use carrying its
// input as an object.
type anOutBlock struct {
	Type  string          `json:"type"`
	Text  *string         `json:"text,omitempty"`
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`
}

type anUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	// CacheReadInputTokens is how many input tokens the memory cache
	// restored; CacheCreationInputTokens how many it computed and kept, for
	// a request that marked a cache_control breakpoint.
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
}

// anCachePrompt says whether a request marks any cache_control breakpoint.
//
// The memory cache keeps every prompt's prefix, so a breakpoint is not
// needed for a later request to restore one; what it maps onto is the usage
// report, cache_creation_input_tokens. Where the breakpoint sits is not read:
// the store keeps pages up to the whole prompt, a superset of any
// breakpoint's prefix.
func anCachePrompt(req *anRequest) bool {
	marked := func(raw json.RawMessage) bool { return bytes.Contains(raw, []byte(`"cache_control"`)) }
	if marked(req.System) {
		return true
	}
	for _, m := range req.Messages {
		if marked(m.Content) {
			return true
		}
	}
	for _, t := range req.Tools {
		if len(t.CacheControl) > 0 && string(t.CacheControl) != "null" {
			return true
		}
	}
	return false
}

// anUsageOf is a finished generate's usage.
func anUsageOf(fin *Finished, cache bool) anUsage {
	u := anUsage{InputTokens: fin.PromptTokens, OutputTokens: fin.CompletionTokens,
		CacheReadInputTokens: fin.Restored}
	if cache {
		u.CacheCreationInputTokens = fin.PromptTokens - fin.Restored
	}
	return u
}

type anResponse struct {
	ID           string       `json:"id"`
	Type         string       `json:"type"`
	Role         string       `json:"role"`
	Model        string       `json:"model"`
	Content      []anOutBlock `json:"content"`
	StopReason   *string      `json:"stop_reason"`
	StopSequence *string      `json:"stop_sequence"`
	Usage        anUsage      `json:"usage"`

	// The same extension the OpenAI shim carries, for the same reason: the
	// Anthropic body has nowhere to say which blocks ran where or how long the
	// request queued behind another session on the same card.
	Jitllm *oaJitllmExtra `json:"jitllm,omitempty"`
}

type anErrorBody struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

type anError struct {
	Type  string      `json:"type"`
	Error anErrorBody `json:"error"`
}

// anText flattens `string | []block` and also accepts the top-level `system`
// field in either form, which the API allows.
func anText(raw json.RawMessage) (string, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return "", nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s, nil
	}
	var blocks []anContentBlock
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return "", errors.New("content must be a string or an array of content blocks")
	}
	var b strings.Builder
	for _, bl := range blocks {
		if bl.Type == "text" || bl.Type == "" {
			b.WriteString(bl.Text)
		}
	}
	return b.String(), nil
}

func anStopReason(r FinishReason) string {
	switch r {
	case FinishMaxTokens:
		return "max_tokens"
	case FinishStop:
		return "stop_sequence"
	case FinishEOS:
		return "end_turn"
	case FinishCancelled:
		return "end_turn"
	}
	return "end_turn"
}

func (e *compat) anthropicMessages(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		anFail(w, http.StatusMethodNotAllowed, "invalid_request_error", "POST only")
		return
	}
	var req anRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		anFail(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	// max_tokens is required by this API, unlike OpenAI's; refusing a
	// missing one matches the real service.
	if req.MaxTokens <= 0 {
		anFail(w, http.StatusBadRequest, "invalid_request_error", "max_tokens is required and must be positive")
		return
	}

	chat := &ChatInput{AddGenerationPrompt: true}
	if sys, err := anText(req.System); err != nil {
		anFail(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	} else if sys != "" {
		chat.System, chat.HasSystem = sys, true
	}
	for _, m := range req.Messages {
		ms, err := anMessages(m)
		if err != nil {
			anFail(w, http.StatusBadRequest, "invalid_request_error", err.Error())
			return
		}
		chat.Messages = append(chat.Messages, ms...)
	}
	if t, err := anTools(req.Tools, req.ToolChoice); err != nil {
		anFail(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	} else {
		chat.Tools = t
	}
	tt := newToolText(chat.Tools)

	o := GenerateOptions{
		Prompt:      Prompt{Kind: PromptChat, Chat: chat},
		MaxTokens:   req.MaxTokens,
		Stop:        req.StopSequences,
		CachePrompt: anCachePrompt(&req),
		Priority:    req.JitllmPriority,
	}
	// Anthropic's default temperature is 1, as OpenAI's is (oaSampling).
	q := oaSampling{Temperature: req.Temperature, TopP: req.TopP, TopK: req.TopK}
	if sm, err := q.sampler(req.JitllmSession); err != nil {
		anFail(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	} else {
		o.Sampling = sm
	}
	if err := e.b.BindTarget(&o, req.JitllmSession, req.Model); err != nil {
		anFail(w, http.StatusNotFound, "not_found_error", err.Error())
		return
	}

	id := "msg_" + e.b.NextID("an")
	if !req.Stream {
		var text strings.Builder
		var started *Started
		var fin *Finished
		err := e.b.Generate(r.Context(), o, func(ev Event) error {
			switch ev.Kind {
			case EventStarted:
				started = ev.Started
			case EventToken:
				tt.push(ev.Token.Text)
				text.WriteString(ev.Token.Text)
			case EventFinished:
				fin = ev.Finished
			}
			return nil
		})
		if err != nil {
			anFailErr(w, err)
			return
		}
		if fin == nil {
			anFail(w, http.StatusInternalServerError, "api_error",
				"the generate finished without a finished event")
			return
		}
		reason := anStopReason(fin.Reason)
		txt := text.String()
		blocks := []anOutBlock{{Type: "text", Text: &txt}}
		if _, content, calls := tt.finish(); len(calls) > 0 {
			reason, blocks = "tool_use", nil
			if content != "" {
				blocks = append(blocks, anOutBlock{Type: "text", Text: &content})
			}
			for _, c := range calls {
				blocks = append(blocks, anOutBlock{Type: "tool_use", ID: "toolu_" + e.b.NextID("tc"),
					Name: c.Name, Input: json.RawMessage(c.Arguments)})
			}
		}
		out := anResponse{
			ID: id, Type: "message", Role: "assistant", Model: req.Model,
			Content:    blocks,
			StopReason: &reason,
			Usage:      anUsageOf(fin, o.CachePrompt),
			Jitllm:     jitllmExtra(started, fin),
		}
		if fin.StopMatched != "" {
			m := fin.StopMatched
			out.StopSequence = &m
		}
		writeJSON(w, http.StatusOK, out)
		return
	}

	if err := admits(e.b, o); err != nil {
		anFailErr(w, err)
		return
	}
	sse, err := newSSE(w)
	if err != nil {
		anFail(w, http.StatusInternalServerError, "api_error", err.Error())
		return
	}
	var fin *Finished
	var started *Started
	gerr := e.b.Generate(r.Context(), o, func(ev Event) error {
		switch ev.Kind {
		case EventStarted:
			started = ev.Started
			// message_start carries the shell of the message with EMPTY
			// content and the input token count; the content arrives as
			// deltas. A client builds its message object from this frame.
			if err := sse.sendNamed("message_start", map[string]any{
				"type": "message_start",
				"message": anResponse{
					ID: id, Type: "message", Role: "assistant", Model: req.Model,
					Content: []anOutBlock{},
					Usage: anUsage{InputTokens: ev.Started.PromptTokens,
						CacheReadInputTokens: ev.Started.Restored},
				},
			}); err != nil {
				return err
			}
			return sse.sendNamed("content_block_start", map[string]any{
				"type": "content_block_start", "index": 0,
				"content_block": anContentBlock{Type: "text"},
			})
		case EventToken:
			txt := tt.push(ev.Token.Text)
			if txt == "" {
				return nil
			}
			return sse.sendNamed("content_block_delta", map[string]any{
				"type": "content_block_delta", "index": 0,
				"delta": map[string]string{"type": "text_delta", "text": txt},
			})
		case EventFinished:
			fin = ev.Finished
		}
		return nil
	})
	if gerr != nil {
		sse.sendError(gerr)
		return
	}
	if fin == nil {
		sse.sendError(errors.New("the generate finished without a finished event"))
		return
	}
	tail, _, calls := tt.finish()
	if tail != "" {
		sse.sendNamed("content_block_delta", map[string]any{
			"type": "content_block_delta", "index": 0,
			"delta": map[string]string{"type": "text_delta", "text": tail},
		})
	}
	sse.sendNamed("content_block_stop", map[string]any{"type": "content_block_stop", "index": 0})
	stop := anStopReason(fin.Reason)
	// A call streams as its own tool_use block: start with an empty input,
	// the arguments as one input_json_delta, stop -- the frames an Anthropic
	// client assembles a tool_use from.
	for i, c := range calls {
		stop = "tool_use"
		sse.sendNamed("content_block_start", map[string]any{
			"type": "content_block_start", "index": i + 1,
			"content_block": anOutBlock{Type: "tool_use", ID: "toolu_" + e.b.NextID("tc"),
				Name: c.Name, Input: json.RawMessage("{}")},
		})
		sse.sendNamed("content_block_delta", map[string]any{
			"type": "content_block_delta", "index": i + 1,
			"delta": map[string]string{"type": "input_json_delta", "partial_json": c.Arguments},
		})
		sse.sendNamed("content_block_stop", map[string]any{"type": "content_block_stop", "index": i + 1})
	}
	delta := map[string]any{"stop_reason": stop, "stop_sequence": nil}
	if fin.StopMatched != "" {
		delta["stop_sequence"] = fin.StopMatched
	}
	sse.sendNamed("message_delta", map[string]any{
		"type":  "message_delta",
		"delta": delta,
		"usage": anUsage{OutputTokens: fin.CompletionTokens},
		// Not part of the upstream shape; see the comment on anResponse.
		"jitllm": jitllmExtra(started, fin),
	})
	// message_stop ends the stream; there is no [DONE] here.
	sse.sendNamed("message_stop", map[string]any{"type": "message_stop"})
}

func anFail(w http.ResponseWriter, status int, typ, msg string) {
	writeJSON(w, status, anError{Type: "error", Error: anErrorBody{Type: typ, Message: msg}})
}

func anFailErr(w http.ResponseWriter, err error) {
	status, typ := http.StatusInternalServerError, "api_error"
	switch {
	case errors.Is(err, ErrNotFound):
		status, typ = http.StatusNotFound, "not_found_error"
	case errors.Is(err, ErrInvalid):
		// A 5xx tells a client to retry the identical request.
		status, typ = http.StatusBadRequest, "invalid_request_error"
	case errors.Is(err, ErrQueueTimeout):
		status, typ = http.StatusTooManyRequests, "overloaded_error"
	case errors.Is(err, ErrOverloaded):
		retryAfter(w, err)
		status, typ = http.StatusTooManyRequests, "overloaded_error"
	}
	anFail(w, status, typ, err.Error())
}

// anMessages turns one Anthropic message into chat turns. A tool_result block
// becomes a "tool" turn of its own -- the shape every chat template reads --
// and tool_use blocks become the assistant turn's calls.
func anMessages(m anMessage) ([]model.ChatMessage, error) {
	var blocks []anInBlock
	if json.Unmarshal(m.Content, &blocks) != nil {
		c, err := anText(m.Content)
		if err != nil {
			return nil, err
		}
		return []model.ChatMessage{chatMessage(m.Role, c)}, nil
	}
	var out []model.ChatMessage
	var text strings.Builder
	var calls []model.ToolCall
	for _, b := range blocks {
		switch b.Type {
		case "text", "":
			text.WriteString(b.Text)
		case "tool_use":
			args := string(b.Input)
			if args == "" || args == "null" {
				args = "{}"
			}
			calls = append(calls, model.ToolCall{ID: b.ID, Name: b.Name, Arguments: args})
		case "tool_result":
			c, err := anText(b.Content)
			if err != nil {
				return nil, err
			}
			out = append(out, model.ChatMessage{Role: "tool", Content: c, ToolCallID: b.ToolUseID})
		}
	}
	if text.Len() > 0 || len(calls) > 0 || len(out) == 0 {
		cm := chatMessage(m.Role, text.String())
		cm.ToolCalls = calls
		out = append(out, cm)
	}
	return out, nil
}

// anTools converts the Anthropic tool list to the OpenAI shape a chat
// template reads, keeping input_schema's bytes (and so its key order).
func anTools(tools []anTool, choice json.RawMessage) ([]byte, error) {
	var c struct {
		Type string `json:"type"`
	}
	if len(tools) == 0 || (json.Unmarshal(choice, &c) == nil && c.Type == "none") {
		return nil, nil
	}
	type fn struct {
		Name        string          `json:"name"`
		Description string          `json:"description,omitempty"`
		Parameters  json.RawMessage `json:"parameters"`
	}
	type tool struct {
		Type     string `json:"type"`
		Function fn     `json:"function"`
	}
	out := make([]tool, len(tools))
	for i, t := range tools {
		if t.Name == "" {
			return nil, fmt.Errorf("tools[%d] has no name", i)
		}
		schema := t.InputSchema
		if len(schema) == 0 {
			schema = json.RawMessage(`{"type": "object", "properties": {}}`)
		}
		out[i] = tool{"function", fn{t.Name, t.Description, schema}}
	}
	return json.Marshal(out)
}
