package server

import (
	"context"
	"net/http"
	"sync"
	"time"
)

// Backend is everything the two HTTP compatibility shims are allowed to use.
// Keeping it this narrow keeps them adapters (they cannot reach a
// model.State), and lets tests script a token stream to gate SSE framing,
// stop strings and flushing without loading a model.
type Backend interface {
	// Generate is the one generate path. Both shims call this and nothing else
	// to produce tokens.
	Generate(ctx context.Context, o GenerateOptions, emit func(Event) error) error

	// ListLoaded is what /v1/models reports.
	ListLoaded() []ModelSummary

	// BindTarget resolves a wire "model" field, or a session id, onto the
	// engine's own ids, filling in o.ModelID or o.SessionID.
	BindTarget(o *GenerateOptions, sessionID, modelName string) error

	// NextID mints a response id.
	NextID(prefix string) string

	// CancelSession stops a generation in flight. It reports whether one was
	// running, which is a different answer from "the session does not exist".
	CancelSession(id string) (bool, error)

	// Embed is the one embedding path; /v1/embeddings calls this and nothing
	// else to produce vectors.
	Embed(ctx context.Context, o EmbedOptions) (*EmbedResult, error)

	// Decide is the one decision path; /v1/systemone calls this and nothing
	// else to answer.
	Decide(ctx context.Context, o DecideOptions) (*DecideResult, error)
}

// ModelSummary is the little a model list needs.
type ModelSummary struct {
	ID       string
	Name     string
	LoadedAt time.Time
	// MaxModelLen is the KV capacity a session of this model gets when it
	// asks for none: the prompt and the completion together.
	MaxModelLen int
}

// compat carries the Backend into the two shims' handlers.
type compat struct {
	b Backend
	// mux is the routes below, which a batch line runs through.
	mux http.Handler
	// The Files and Batch APIs' store, opened on first use.
	batchOnce sync.Once
	batch     *batches
	batchErr  error
}

// ---- *Engine implements Backend.

func (e *Engine) ListLoaded() []ModelSummary {
	out := []ModelSummary{}
	for _, lm := range e.Models() {
		out = append(out, ModelSummary{ID: lm.id, Name: lm.name, LoadedAt: lm.loadedAt,
			MaxModelLen: e.defaultMaxSeq(lm)})
	}
	return out
}

func (e *Engine) NextID(prefix string) string { return e.nextID(prefix) }

func (e *Engine) CancelSession(id string) (bool, error) {
	s, err := e.Session(id)
	if err != nil {
		return false, err
	}
	return s.cancelGeneration(), nil
}

var _ Backend = (*Engine)(nil)

// CompatHandler mounts the two compatibility APIs on their own mux. It takes
// a Backend so tests can drive it with a scripted token stream.
func CompatHandler(b Backend) http.Handler {
	c := &compat{b: b}
	mux := http.NewServeMux()
	c.mux = mux
	mux.HandleFunc("/v1/files", c.openAIFiles)
	mux.HandleFunc("/v1/files/", c.openAIFiles)
	mux.HandleFunc("/v1/batches", c.openAIBatches)
	mux.HandleFunc("/v1/batches/", c.openAIBatches)
	mux.HandleFunc("/v1/chat/completions", c.openAIChatCompletions)
	mux.HandleFunc("/v1/completions", c.openAICompletions)
	mux.HandleFunc("/v1/models", c.openAIModels)
	mux.HandleFunc("/v1/embeddings", c.openAIEmbeddings)
	mux.HandleFunc("/v1/systemone", c.systemOne)
	mux.HandleFunc("/v1/messages", c.anthropicMessages)
	return mux
}
