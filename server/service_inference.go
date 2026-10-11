package server

import (
	"context"
	"fmt"
	"image"
	"strings"
	"time"

	"connectrpc.com/connect"

	"github.com/jitllm/jitllm/engine/grammar"

	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// InferenceService implements jitllm.v1.InferenceService. It holds a Backend
// so its streaming path can be gated without a loaded model.
type InferenceService struct {
	B Backend
	// Images bounds a chat's pictures, as it does on the HTTP shims.
	Images ImagePolicy
}

// generateOptions turns the wire request into the engine's one request shape,
// the same struct the HTTP shims build, with the default picture limits.
func generateOptions(m *v1.GenerateRequest) (GenerateOptions, error) {
	return generateOptionsWith(m, ImagePolicy{})
}

// generateOptionsWith is generateOptions with a chat's pictures bounded by p.
func generateOptionsWith(m *v1.GenerateRequest, p ImagePolicy) (GenerateOptions, error) {
	p = p.withDefaults()
	o := GenerateOptions{
		SessionID:    m.SessionId,
		ModelID:      m.ModelId,
		MaxTokens:    int(m.MaxTokens),
		Sampling:     pbSampling(m.Sampling),
		Stop:         m.Stop,
		Continue:     m.ContinueSession,
		Echo:         m.EchoPrompt,
		QueueTimeout: time.Duration(m.QueueTimeoutMillis) * time.Millisecond,
		IgnoreEOS:    m.IgnoreEos,
		Speculation:  pbSpeculation(m.Speculation),
	}
	switch {
	case m.Grammar != "" && m.JsonSchema != "":
		return o, invalid("generate: grammar and json_schema are one constraint each; give one")
	case m.JsonSchema != "":
		g, err := grammar.FromJSONSchema([]byte(m.JsonSchema))
		if err != nil {
			return o, invalid("generate: json_schema: %v", err)
		}
		o.Grammar = g
	default:
		o.Grammar = m.Grammar
	}
	switch in := m.Prompt.GetInput().(type) {
	case *v1.PromptInput_Text:
		o.Prompt = Prompt{Kind: PromptText, Text: in.Text}
	case *v1.PromptInput_TokenIds:
		o.Prompt = Prompt{Kind: PromptIDs, IDs: in.TokenIds.GetIds()}
	case *v1.PromptInput_Chat:
		c := &ChatInput{
			AddGenerationPrompt: in.Chat.GetAddGenerationPrompt(),
			TemplateName:        in.Chat.GetTemplateName(),
		}
		if in.Chat.System != nil {
			c.System, c.HasSystem = in.Chat.GetSystem(), true
		}
		pics := &pictures{p: p}
		for i, msg := range in.Chat.GetMessages() {
			cm := chatMessage(msg.GetRole(), msg.GetContent())
			for j, img := range msg.GetImages() {
				err := pics.add(fmt.Sprintf("prompt.chat.messages[%d].images[%d]", i, j), func() (image.Image, error) {
					mt := ""
					if img.GetMediaType() != "" {
						var ok bool
						if mt, ok = imageMediaType(img.GetMediaType()); !ok {
							return nil, fmt.Errorf("has media_type %q; the pictures accepted are image/png, "+
								"image/jpeg, image/webp and image/gif", img.GetMediaType())
						}
					}
					return p.decodeImage(img.GetData(), mt)
				})
				if err != nil {
					return o, invalid("generate: %v", err)
				}
				cm.Images++
			}
			c.Messages = append(c.Messages, cm)
		}
		c.Images = pics.imgs
		o.Prompt = Prompt{Kind: PromptChat, Chat: c}
	case nil:
		if !m.ContinueSession {
			return o, invalid("generate: prompt is required")
		}
	}
	return o, nil
}

func (s *InferenceService) Generate(ctx context.Context, req *connect.Request[v1.GenerateRequest], st *connect.ServerStream[v1.GenerateResponse]) error {
	o, err := generateOptionsWith(req.Msg, s.Images)
	if err != nil {
		return err
	}
	err = s.B.Generate(ctx, o, func(ev Event) error {
		return st.Send(pbEvent(ev))
	})
	return connectErr(err)
}

// Complete is the streaming path with the events collected: the same call
// with a different sink.
func (s *InferenceService) Complete(ctx context.Context, req *connect.Request[v1.GenerateRequest]) (*connect.Response[v1.CompleteResponse], error) {
	o, err := generateOptionsWith(req.Msg, s.Images)
	if err != nil {
		return nil, err
	}
	out := &v1.CompleteResponse{}
	var text strings.Builder
	err = s.B.Generate(ctx, o, func(ev Event) error {
		switch ev.Kind {
		case EventStarted:
			out.Started = pbStarted(ev.Started)
		case EventToken:
			text.WriteString(ev.Token.Text)
			if ev.Token.ID >= 0 {
				out.TokenIds = append(out.TokenIds, ev.Token.ID)
			}
		case EventFinished:
			out.Finished = pbFinished(ev.Finished)
		}
		return nil
	})
	if err != nil {
		return nil, connectErr(err)
	}
	out.Text = text.String()
	return connect.NewResponse(out), nil
}

func (s *InferenceService) Cancel(ctx context.Context, req *connect.Request[v1.CancelRequest]) (*connect.Response[v1.CancelResponse], error) {
	was, err := s.B.CancelSession(req.Msg.SessionId)
	if err != nil {
		return nil, connectErr(err)
	}
	return connect.NewResponse(&v1.CancelResponse{WasGenerating: was}), nil
}

func pbEvent(ev Event) *v1.GenerateResponse {
	switch ev.Kind {
	case EventStarted:
		return &v1.GenerateResponse{Event: &v1.GenerateResponse_Started{Started: pbStarted(ev.Started)}}
	case EventToken:
		return &v1.GenerateResponse{Event: &v1.GenerateResponse_Token{Token: &v1.GenerateToken{
			TokenId: ev.Token.ID, Text: ev.Token.Text, Index: int32(ev.Token.Index),
		}}}
	default:
		return &v1.GenerateResponse{Event: &v1.GenerateResponse_Finished{Finished: pbFinished(ev.Finished)}}
	}
}

func pbStarted(s *Started) *v1.GenerateStarted {
	return &v1.GenerateStarted{
		SessionId:         s.SessionID,
		ModelId:           s.ModelID,
		EphemeralSession:  s.Ephemeral,
		PromptTokens:      int32(s.PromptTokens),
		QueuedMillis:      millis(s.QueuedFor),
		QueueDepthOnEntry: s.QueueDepth,
		DeviceBlocks:      int32(s.DeviceBlocks),
		HostBlocks:        int32(s.HostBlocks),
		DeviceIds:         s.DeviceIDs,
		PrefillMillis:     millis(s.Prefill),
		Batched:           s.Batched,
	}
}

func pbFinished(f *Finished) *v1.GenerateFinished {
	return &v1.GenerateFinished{
		Reason:                pbFinish(f.Reason),
		StopMatched:           f.StopMatched,
		PromptTokens:          int32(f.PromptTokens),
		CompletionTokens:      int32(f.CompletionTokens),
		PrefillMillis:         millis(f.Prefill),
		DecodeMillis:          millis(f.Decode),
		DecodeTokensPerSecond: f.TokensPerSecond,
		BytesPerToken:         f.BytesPerToken,
		Position:              int32(f.Position),
	}
}

// Embed is Engine.Embed with the wire request turned into EmbedOptions, the
// same struct /v1/embeddings builds.
func (s *InferenceService) Embed(ctx context.Context, req *connect.Request[v1.EmbedRequest]) (*connect.Response[v1.EmbedResponse], error) {
	m := req.Msg
	o := EmbedOptions{
		Model:      m.ModelId,
		Dimensions: int(m.Dimensions),
	}
	for i, in := range m.Inputs {
		switch p := in.GetInput().(type) {
		case *v1.EmbedInput_Text:
			o.Inputs = append(o.Inputs, Prompt{Kind: PromptText, Text: p.Text})
		case *v1.EmbedInput_TokenIds:
			o.Inputs = append(o.Inputs, Prompt{Kind: PromptIDs, IDs: p.TokenIds.GetIds()})
		default:
			return nil, invalid("embed: input %d carries neither text nor token ids", i)
		}
	}
	res, err := s.B.Embed(ctx, o)
	if err != nil {
		return nil, connectErr(err)
	}
	out := &v1.EmbedResponse{
		ModelId:           res.ModelID,
		Pooling:           res.Pooling,
		TotalTokens:       int32(res.TotalTokens()),
		QueuedMillis:      millis(res.QueuedFor),
		QueueDepthOnEntry: res.QueueDepth,
		EmbedMillis:       millis(res.Took),
	}
	for i, v := range res.Vectors {
		out.Embeddings = append(out.Embeddings, &v1.Embedding{Vector: v, Tokens: int32(res.Tokens[i])})
		out.Dimensions = int32(len(v))
	}
	return connect.NewResponse(out), nil
}
