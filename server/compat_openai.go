package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/samyfodil/jitllm/engine/model"
)

// OpenAI-compatible HTTP.
//
// It is an adapter: every handler builds a GenerateOptions and calls
// Engine.Generate, the same path as the ConnectRPC InferenceService. Nothing
// here prefills, samples or detokenizes.

type oaMessage struct {
	Role string `json:"role"`
	// Content is `string | []{type,text}` in the real API. Both are accepted;
	// see oaContent.
	Content json.RawMessage `json:"content"`
	// An assistant turn's calls, and a "tool" turn's answer to one.
	ToolCalls  []oaToolCall `json:"tool_calls,omitempty"`
	ToolCallID string       `json:"tool_call_id,omitempty"`
	Name       string       `json:"name,omitempty"`
}

type oaToolCall struct {
	// Index is set on a streamed delta only, as the API does.
	Index    *int   `json:"index,omitempty"`
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type oaChatRequest struct {
	Model       string          `json:"model"`
	Messages    []oaMessage     `json:"messages"`
	MaxTokens   *int            `json:"max_tokens"`
	Temperature *float64        `json:"temperature"`
	TopP        *float64        `json:"top_p"`
	Stream      bool            `json:"stream"`
	Stop        json.RawMessage `json:"stop"`
	Seed        *int64          `json:"seed"`
	User        string          `json:"user"`
	// Tools is handed to the model's own chat template as `tools`, bytes
	// untouched so the schema keeps the key order the client wrote.
	// tool_choice "none" withholds them; "auto", "required" and a named
	// function are all rendered as "auto": nothing constrains the sampler.
	Tools      json.RawMessage `json:"tools"`
	ToolChoice json.RawMessage `json:"tool_choice"`

	// MaxCompletionTokens is the newer spelling; the OpenAI SDKs send it for
	// reasoning models and clients in the wild send either.
	MaxCompletionTokens *int `json:"max_completion_tokens"`

	// Logprobs and TopLogprobs report each token's log-probability and the
	// most likely alternatives (compat_choices.go); N asks for that many
	// continuations of the prompt, prefilled once.
	Logprobs    *bool `json:"logprobs"`
	TopLogprobs *int  `json:"top_logprobs"`
	N           *int  `json:"n"`

	// IgnoreEOS is the vLLM and llama-server extension: generate past an
	// end-of-generation token, up to max_tokens.
	IgnoreEOS bool `json:"ignore_eos"`

	// ---- jitllm extensions. Prefixed so they cannot collide with a future
	// OpenAI field, and ignored by any client that does not know them.
	JitllmSession string `json:"jitllm_session,omitempty"`
}

type oaCompletionRequest struct {
	Model       string          `json:"model"`
	Prompt      json.RawMessage `json:"prompt"`
	MaxTokens   *int            `json:"max_tokens"`
	Temperature *float64        `json:"temperature"`
	TopP        *float64        `json:"top_p"`
	Stream      bool            `json:"stream"`
	Stop        json.RawMessage `json:"stop"`
	Seed        *int64          `json:"seed"`
	Echo        bool            `json:"echo"`
	IgnoreEOS   bool            `json:"ignore_eos"`
	// Logprobs is the alternative count, 0..20; N the continuations.
	Logprobs *int `json:"logprobs"`
	N        *int `json:"n"`

	JitllmSession string `json:"jitllm_session,omitempty"`
}

type oaUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type oaChatChoice struct {
	Index        int             `json:"index"`
	Message      *oaOutMsg       `json:"message,omitempty"`
	Delta        *oaOutMsg       `json:"delta,omitempty"`
	FinishReason *string         `json:"finish_reason"`
	Logprobs     *oaChatLogprobs `json:"logprobs"`
}

type oaOutMsg struct {
	Role      string       `json:"role,omitempty"`
	Content   string       `json:"content,omitempty"`
	ToolCalls []oaToolCall `json:"tool_calls,omitempty"`
}

type oaChatResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []oaChatChoice `json:"choices"`
	Usage   *oaUsage       `json:"usage,omitempty"`

	// Jitllm is the one field this shim adds: the queue wait and placement,
	// which an OpenAI response has nowhere else to carry. Clients that do not
	// know it ignore it.
	Jitllm *oaJitllmExtra `json:"jitllm,omitempty"`
}

type oaJitllmExtra struct {
	SessionID     string   `json:"session_id"`
	QueuedMillis  int64    `json:"queued_ms"`
	QueueDepth    int32    `json:"queue_depth_on_entry"`
	DeviceBlocks  int      `json:"device_blocks"`
	HostBlocks    int      `json:"host_blocks"`
	DeviceIDs     []string `json:"device_ids,omitempty"`
	PrefillMillis int64    `json:"prefill_ms"`
	DecodeMillis  int64    `json:"decode_ms"`
	TokensPerSec  float64  `json:"decode_tokens_per_second"`
	BytesPerToken uint64   `json:"bytes_per_token"`
	// Seeds and PromptRestored are set for several choices: each choice's
	// seed, and how many prompt positions each restored rather than ran.
	Seeds          []int64 `json:"seeds,omitempty"`
	PromptRestored []int   `json:"prompt_restored,omitempty"`
}

type oaError struct {
	Error oaErrorBody `json:"error"`
}

type oaErrorBody struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code,omitempty"`
}

// oaContent accepts both content shapes the API has shipped: a bare string,
// and the content-part array every SDK sends. A shim that only took
// the string form rejects the default request of the official python client.
func oaContent(raw json.RawMessage) (string, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return "", nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s, nil
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &parts); err != nil {
		return "", fmt.Errorf("content must be a string or an array of content parts")
	}
	var b strings.Builder
	for _, p := range parts {
		if p.Type == "text" || p.Type == "" {
			b.WriteString(p.Text)
		}
	}
	return b.String(), nil
}

// oaStop accepts `"x"` and `["x","y"]`, both of which the API allows.
func oaStop(raw json.RawMessage) []string {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var one string
	if err := json.Unmarshal(raw, &one); err == nil {
		if one == "" {
			return nil
		}
		return []string{one}
	}
	var many []string
	if err := json.Unmarshal(raw, &many); err == nil {
		return many
	}
	return nil
}

func (e *compat) openAIModels(w http.ResponseWriter, r *http.Request) {
	type entry struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		OwnedBy string `json:"owned_by"`
	}
	out := struct {
		Object string  `json:"object"`
		Data   []entry `json:"data"`
	}{Object: "list", Data: []entry{}}
	for _, lm := range e.b.ListLoaded() {
		out.Data = append(out.Data, entry{
			ID: lm.ID, Object: "model", Created: lm.LoadedAt.Unix(), OwnedBy: "jitllm",
		})
		if lm.Name != lm.ID && lm.Name != "" {
			// The file name is an alias, so a client configured with the model
			// name rather than the generated id still resolves.
			out.Data = append(out.Data, entry{
				ID: lm.Name, Object: "model", Created: lm.LoadedAt.Unix(), OwnedBy: "jitllm",
			})
		}
	}
	writeJSON(w, http.StatusOK, out)
}

func (e *compat) openAIChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		oaFail(w, http.StatusMethodNotAllowed, "POST only", "invalid_request_error")
		return
	}
	var req oaChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		oaFail(w, http.StatusBadRequest, err.Error(), "invalid_request_error")
		return
	}
	chat := &ChatInput{AddGenerationPrompt: true}
	for _, m := range req.Messages {
		c, err := oaContent(m.Content)
		if err != nil {
			oaFail(w, http.StatusBadRequest, err.Error(), "invalid_request_error")
			return
		}
		cm := chatMessage(m.Role, c)
		cm.ToolCallID, cm.Name = m.ToolCallID, m.Name
		for _, tc := range m.ToolCalls {
			cm.ToolCalls = append(cm.ToolCalls, model.ToolCall{ID: tc.ID, Name: tc.Function.Name,
				Arguments: tc.Function.Arguments})
		}
		chat.Messages = append(chat.Messages, cm)
	}
	if t, err := oaTools(req.Tools, req.ToolChoice); err != nil {
		oaFail(w, http.StatusBadRequest, err.Error(), "invalid_request_error")
		return
	} else {
		chat.Tools = t
	}
	maxTok := 0
	if req.MaxTokens != nil {
		maxTok = *req.MaxTokens
	} else if req.MaxCompletionTokens != nil {
		maxTok = *req.MaxCompletionTokens
	}
	lpr, err := oaChatLogprobsOf(req.Logprobs, req.TopLogprobs)
	if err != nil {
		oaFail(w, http.StatusBadRequest, err.Error(), "invalid_request_error")
		return
	}
	seeds, err := oaChoiceSeeds(req.N, req.Seed)
	if err != nil {
		oaFail(w, http.StatusBadRequest, err.Error(), "invalid_request_error")
		return
	}
	o := GenerateOptions{
		Prompt:    Prompt{Kind: PromptChat, Chat: chat},
		MaxTokens: maxTok,
		Stop:      oaStop(req.Stop),
		Sampling:  oaSampling(req.Temperature, req.TopP, req.Seed, seeds),
		IgnoreEOS: req.IgnoreEOS,
		Seeds:     seeds,
	}
	lpr.apply(&o)
	if err := e.b.BindTarget(&o, req.JitllmSession, req.Model); err != nil {
		oaFail(w, http.StatusNotFound, err.Error(), "invalid_request_error")
		return
	}
	e.runOpenAI(w, r, o, req.Model, req.Stream, true)
}

func (e *compat) openAICompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		oaFail(w, http.StatusMethodNotAllowed, "POST only", "invalid_request_error")
		return
	}
	var req oaCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		oaFail(w, http.StatusBadRequest, err.Error(), "invalid_request_error")
		return
	}
	// `prompt` is `string | []string | []int | [][]int`. The first two are
	// honoured; a token-id prompt goes through as ids, which this engine can
	// take directly.
	var p Prompt
	var s string
	var ids []int32
	switch {
	case json.Unmarshal(req.Prompt, &s) == nil:
		p = Prompt{Kind: PromptText, Text: s}
	case json.Unmarshal(req.Prompt, &ids) == nil:
		p = Prompt{Kind: PromptIDs, IDs: ids}
	default:
		var many []string
		if err := json.Unmarshal(req.Prompt, &many); err == nil && len(many) > 0 {
			// Batched prompts would need one choice each and the engine's
			// batched path has no accelerator arm, so only the first is run
			// and the response says so by carrying one choice.
			p = Prompt{Kind: PromptText, Text: many[0]}
		} else {
			oaFail(w, http.StatusBadRequest, "prompt must be a string, an array of strings, or token ids", "invalid_request_error")
			return
		}
	}
	maxTok := 0
	if req.MaxTokens != nil {
		maxTok = *req.MaxTokens
	}
	lpr, err := oaLegacyLogprobsOf(req.Logprobs)
	if err != nil {
		oaFail(w, http.StatusBadRequest, err.Error(), "invalid_request_error")
		return
	}
	if lpr.on && req.Echo {
		oaFail(w, http.StatusBadRequest, "logprobs with echo would need the prompt's own "+
			"log-probabilities, which are not built", "invalid_request_error")
		return
	}
	seeds, err := oaChoiceSeeds(req.N, req.Seed)
	if err != nil {
		oaFail(w, http.StatusBadRequest, err.Error(), "invalid_request_error")
		return
	}
	o := GenerateOptions{
		Prompt:    p,
		MaxTokens: maxTok,
		Stop:      oaStop(req.Stop),
		Echo:      req.Echo,
		Sampling:  oaSampling(req.Temperature, req.TopP, req.Seed, seeds),
		IgnoreEOS: req.IgnoreEOS,
		Seeds:     seeds,
	}
	lpr.apply(&o)
	if err := e.b.BindTarget(&o, req.JitllmSession, req.Model); err != nil {
		oaFail(w, http.StatusNotFound, err.Error(), "invalid_request_error")
		return
	}
	e.runOpenAI(w, r, o, req.Model, req.Stream, false)
}

// oaChoice is one continuation's state while a response is built.
type oaChoice struct {
	tt      *toolText
	text    strings.Builder
	started *Started
	fin     *Finished
	// The logprobs not yet sent: a streamed token whose text is held back
	// (a partial rune, a possible stop or tool call) sends its logprobs with
	// the next chunk that carries text, or with the last.
	pend   []oaLogprobContent
	legacy *oaLegacyLogprobs
	offset int
}

func (c *oaChoice) take(t *Token, lp bool) {
	if !lp || t.Logprob == nil {
		return
	}
	c.pend = append(c.pend, oaChatEntry(t.Logprob))
	c.legacy.add(t.Logprob, c.offset)
	c.offset += len(t.Logprob.Text)
}

// chatLogprobs hands over the pending entries in the chat shape.
func (c *oaChoice) chatLogprobs(lp bool) *oaChatLogprobs {
	if !lp {
		return nil
	}
	out := &oaChatLogprobs{Content: c.pend}
	if out.Content == nil {
		out.Content = []oaLogprobContent{}
	}
	c.pend = nil
	return out
}

// legacyLogprobs hands over the entries since the last call in the legacy
// shape: a streamed chunk carries its own tokens, a whole response all.
func (c *oaChoice) legacyLogprobs(lp bool) *oaLegacyLogprobs {
	if !lp {
		return nil
	}
	out := c.legacy
	c.legacy = &oaLegacyLogprobs{}
	return out
}

// runOpenAI is the one place a generate becomes an OpenAI response, streaming
// or not. Both entry points funnel through it so the two shapes cannot drift.
// Several choices (n) arrive as one generate whose events carry their index.
func (e *compat) runOpenAI(w http.ResponseWriter, r *http.Request, o GenerateOptions, modelName string, stream, chat bool) {
	id := "chatcmpl-" + e.b.NextID("oa")
	object, deltaObject := "chat.completion", "chat.completion.chunk"
	if !chat {
		id = "cmpl-" + e.b.NextID("oa")
		object, deltaObject = "text_completion", "text_completion"
	}
	created := time.Now().Unix()
	lp := o.Logprobs

	var tools []byte
	if o.Prompt.Chat != nil {
		tools = o.Prompt.Chat.Tools
	}
	n := max(len(o.Seeds), 1)
	cs := make([]*oaChoice, n)
	for i := range cs {
		cs[i] = &oaChoice{tt: newToolText(tools), legacy: &oaLegacyLogprobs{}}
	}
	usage := func() *oaUsage {
		u := &oaUsage{}
		for i, c := range cs {
			if c.fin == nil {
				continue
			}
			// The prompt is counted once: every choice after the first
			// restored it rather than running it.
			if i == 0 {
				u.PromptTokens = c.fin.PromptTokens
			}
			u.CompletionTokens += c.fin.CompletionTokens
		}
		u.TotalTokens = u.PromptTokens + u.CompletionTokens
		return u
	}
	extra := func() *oaJitllmExtra {
		x := jitllmExtra(cs[0].started, cs[0].fin)
		if x != nil && len(o.Seeds) > 0 {
			x.Seeds = o.Seeds
			for _, c := range cs {
				if c.started != nil {
					x.PromptRestored = append(x.PromptRestored, c.started.Restored)
				}
			}
		}
		return x
	}
	choice := func(ev Event) *oaChoice {
		if ev.Choice < 0 || ev.Choice >= n {
			return cs[0]
		}
		return cs[ev.Choice]
	}

	if !stream {
		err := e.b.Generate(r.Context(), o, func(ev Event) error {
			c := choice(ev)
			switch ev.Kind {
			case EventStarted:
				c.started = ev.Started
			case EventToken:
				c.tt.push(ev.Token.Text)
				c.text.WriteString(ev.Token.Text)
				c.take(ev.Token, lp)
			case EventFinished:
				c.fin = ev.Finished
			}
			return nil
		})
		if err != nil {
			oaFailErr(w, err)
			return
		}
		// A Backend that returned nil without a Finished event would nil-deref
		// below. It cannot happen with the engine, and a shim that panics on
		// an unexpected backend is a shim that takes the process out.
		for _, c := range cs {
			if c.fin == nil {
				oaFail(w, http.StatusInternalServerError,
					"the generate finished without a finished event", "server_error")
				return
			}
		}
		if !chat {
			// A legacy completion carries `text`, not a message: the two
			// response bodies are genuinely different shapes.
			out := legacyBody(id, object, created, modelName)
			for i, c := range cs {
				reason := oaFinish(c.fin.Reason)
				out.Choices = append(out.Choices, legacyChoice{Index: i, Text: c.text.String(),
					FinishReason: &reason, Logprobs: c.legacyLogprobs(lp)})
			}
			out.Usage, out.Jitllm = usage(), extra()
			writeJSON(w, http.StatusOK, out)
			return
		}
		resp := oaChatResponse{ID: id, Object: object, Created: created, Model: modelName}
		for i, c := range cs {
			reason := oaFinish(c.fin.Reason)
			msg := &oaOutMsg{Role: "assistant", Content: c.text.String()}
			if _, content, calls := c.tt.finish(); len(calls) > 0 {
				reason = "tool_calls"
				msg.Content, msg.ToolCalls = content, e.oaCalls(calls, false)
			}
			resp.Choices = append(resp.Choices, oaChatChoice{Index: i, Message: msg,
				FinishReason: &reason, Logprobs: c.chatLogprobs(lp)})
		}
		resp.Usage, resp.Jitllm = usage(), extra()
		writeJSON(w, http.StatusOK, resp)
		return
	}

	// ---- streaming: Server-Sent Events, `data: {...}` per chunk, terminated
	// by `data: [DONE]`. That terminator is OpenAI's and is NOT part of SSE;
	// clients written against the spec wait for it, so omitting it hangs them.
	sse, err := newSSE(w)
	if err != nil {
		oaFail(w, http.StatusInternalServerError, err.Error(), "server_error")
		return
	}
	chunk := func(ch oaChatChoice) error {
		return sse.send(oaChatResponse{
			ID: id, Object: deltaObject, Created: created, Model: modelName,
			Choices: []oaChatChoice{ch},
		})
	}
	// end is a choice's last chunks, sent when its Finished arrives so the
	// next choice's stream starts after it.
	end := func(i int, c *oaChoice) {
		reason := oaFinish(c.fin.Reason)
		if !chat {
			out := legacyBody(id, object, created, modelName)
			out.Choices = []legacyChoice{{Index: i, FinishReason: &reason, Logprobs: c.legacyLogprobs(lp)}}
			if n == 1 {
				out.Usage, out.Jitllm = usage(), extra()
			}
			sse.send(out)
			return
		}
		// What the tool holdback kept back goes out now: the calls, or the
		// text that turned out not to be one.
		tail, _, calls := c.tt.finish()
		if tail != "" {
			chunk(oaChatChoice{Index: i, Delta: &oaOutMsg{Content: tail}, Logprobs: c.chatLogprobs(lp)})
		}
		if len(calls) > 0 {
			reason = "tool_calls"
			chunk(oaChatChoice{Index: i, Delta: &oaOutMsg{ToolCalls: e.oaCalls(calls, true)}})
		}
		var lps *oaChatLogprobs
		if len(c.pend) > 0 {
			lps = c.chatLogprobs(lp)
		}
		last := oaChatResponse{
			ID: id, Object: deltaObject, Created: created, Model: modelName,
			Choices: []oaChatChoice{{Index: i, Delta: &oaOutMsg{}, FinishReason: &reason, Logprobs: lps}},
		}
		if n == 1 {
			last.Usage, last.Jitllm = usage(), extra()
		}
		sse.send(last)
	}
	gerr := e.b.Generate(r.Context(), o, func(ev Event) error {
		c := choice(ev)
		i := ev.Choice
		switch ev.Kind {
		case EventStarted:
			c.started = ev.Started
			if chat {
				// The first chunk carries the role and no content, which is
				// what the reference implementation emits and what several
				// clients key their state machine on.
				return chunk(oaChatChoice{Index: i, Delta: &oaOutMsg{Role: "assistant"}})
			}
		case EventToken:
			c.take(ev.Token, lp)
			if ev.Token.Text == "" {
				return nil
			}
			if chat {
				txt := c.tt.push(ev.Token.Text)
				if txt == "" {
					return nil
				}
				return chunk(oaChatChoice{Index: i, Delta: &oaOutMsg{Content: txt}, Logprobs: c.chatLogprobs(lp)})
			}
			out := legacyBody(id, object, created, modelName)
			out.Choices = []legacyChoice{{Index: i, Text: ev.Token.Text, Logprobs: c.legacyLogprobs(lp)}}
			return sse.send(out)
		case EventFinished:
			c.fin = ev.Finished
			end(i, c)
		}
		return nil
	})
	if gerr != nil {
		// The stream is already open, so an error cannot become a status code.
		// SSE's own `event: error` frame is what a client can act on.
		sse.sendError(gerr)
		sse.done()
		return
	}
	for _, c := range cs {
		if c.fin == nil {
			sse.sendError(errors.New("the generate finished without a finished event"))
			sse.done()
			return
		}
	}
	// One choice's usage rode on its last chunk; several choices' usage,
	// once for the whole response, rides on a chunk of its own.
	switch {
	case n == 1:
	case chat:
		sse.send(oaChatResponse{
			ID: id, Object: deltaObject, Created: created, Model: modelName,
			Choices: []oaChatChoice{}, Usage: usage(), Jitllm: extra(),
		})
	default:
		out := legacyBody(id, object, created, modelName)
		out.Choices = []legacyChoice{}
		out.Usage, out.Jitllm = usage(), extra()
		sse.send(out)
	}
	sse.done()
}

// legacyChoice is /v1/completions' choice, which is NOT the chat one: it
// carries `text` rather than a message or a delta.
type legacyChoice struct {
	Index        int               `json:"index"`
	Text         string            `json:"text"`
	FinishReason *string           `json:"finish_reason"`
	Logprobs     *oaLegacyLogprobs `json:"logprobs"`
}

// legacyResponse is /v1/completions' body.
type legacyResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []legacyChoice `json:"choices"`
	Usage   *oaUsage       `json:"usage,omitempty"`
	Jitllm  *oaJitllmExtra `json:"jitllm,omitempty"`
}

func legacyBody(id, object string, created int64, modelName string) legacyResponse {
	return legacyResponse{ID: id, Object: object, Created: created, Model: modelName}
}

func jitllmExtra(s *Started, f *Finished) *oaJitllmExtra {
	if s == nil {
		return nil
	}
	x := &oaJitllmExtra{
		SessionID:     s.SessionID,
		QueuedMillis:  s.QueuedFor.Milliseconds(),
		QueueDepth:    s.QueueDepth,
		DeviceBlocks:  s.DeviceBlocks,
		HostBlocks:    s.HostBlocks,
		DeviceIDs:     s.DeviceIDs,
		PrefillMillis: s.Prefill.Milliseconds(),
	}
	if f != nil {
		x.DecodeMillis = f.Decode.Milliseconds()
		x.TokensPerSec = f.TokensPerSecond
		x.BytesPerToken = f.BytesPerToken
	}
	return x
}

func oaFinish(r FinishReason) string {
	switch r {
	case FinishMaxTokens:
		return "length"
	case FinishStop, FinishEOS:
		return "stop"
	case FinishCancelled:
		return "cancelled"
	}
	return "stop"
}

// BindTarget resolves the `model` field, or the jitllm_session extension, onto
// the engine's own ids. A session id wins: continuing a conversation is a
// stronger statement than naming a model.
func (e *Engine) BindTarget(o *GenerateOptions, sessionID, modelName string) error {
	if sessionID != "" {
		if _, err := e.Session(sessionID); err != nil {
			return err
		}
		o.SessionID = sessionID
		o.Continue = false
		return nil
	}
	lm, err := e.resolveModel(modelName)
	if err != nil {
		return err
	}
	o.ModelID = lm.id
	return nil
}

func samplerFrom(temp, topP *float64, seed *int64) *model.Sampler {
	if temp == nil && topP == nil && seed == nil {
		return nil
	}
	s := &model.Sampler{}
	if temp != nil {
		s.Temp = *temp
	}
	if topP != nil {
		s.TopP = *topP
	}
	if seed != nil {
		s.Seed = *seed
	}
	return s
}

func oaFail(w http.ResponseWriter, status int, msg, typ string) {
	writeJSON(w, status, oaError{Error: oaErrorBody{Message: msg, Type: typ}})
}

func oaFailErr(w http.ResponseWriter, err error) {
	status, typ := http.StatusInternalServerError, "server_error"
	switch {
	case errors.Is(err, ErrNotFound):
		status, typ = http.StatusNotFound, "invalid_request_error"
	case errors.Is(err, ErrInvalid):
		// A 5xx tells a client to retry the identical request.
		status, typ = http.StatusBadRequest, "invalid_request_error"
	case errors.Is(err, ErrQueueTimeout):
		// 429, because retrying is correct: the session was queued behind
		// another on the same device, never refused.
		status, typ = http.StatusTooManyRequests, "rate_limit_error"
	}
	oaFail(w, status, err.Error(), typ)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

// oaTools validates the request's tool list and applies tool_choice "none".
func oaTools(raw, choice json.RawMessage) ([]byte, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var none string
	if json.Unmarshal(choice, &none) == nil && none == "none" {
		return nil, nil
	}
	var ts []struct {
		Type     string `json:"type"`
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if err := json.Unmarshal(raw, &ts); err != nil {
		return nil, fmt.Errorf("tools must be an array of {type, function}: %v", err)
	}
	for i, t := range ts {
		if t.Type != "function" || t.Function.Name == "" {
			return nil, fmt.Errorf("tools[%d]: only {\"type\": \"function\"} with a name is supported", i)
		}
	}
	if len(ts) == 0 {
		return nil, nil
	}
	return raw, nil
}

// oaCalls is calls in the API's shape; a streamed delta carries each index.
func (e *compat) oaCalls(calls []model.ToolCall, stream bool) []oaToolCall {
	out := make([]oaToolCall, len(calls))
	for i, c := range calls {
		out[i].ID, out[i].Type = "call_"+e.b.NextID("tc"), "function"
		out[i].Function.Name, out[i].Function.Arguments = c.Name, c.Arguments
		if stream {
			out[i].Index = &i
		}
	}
	return out
}
