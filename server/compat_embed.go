package server

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
)

// OpenAI-compatible POST /v1/embeddings. An adapter like the rest of the shim:
// it builds EmbedOptions and calls Backend.Embed, the same path as the Connect
// InferenceService.Embed.

// oaEmbeddingRequest is the request. `input` is
// `string | []string | []int | [][]int`; every form is honoured, the token
// forms as ids the embedder takes directly.
type oaEmbeddingRequest struct {
	Model          string          `json:"model"`
	Input          json.RawMessage `json:"input"`
	EncodingFormat string          `json:"encoding_format"`
	Dimensions     *int            `json:"dimensions"`
	User           string          `json:"user"`
}

type oaEmbedding struct {
	Object string `json:"object"`
	Index  int    `json:"index"`
	// []float32, or the base64 of their little-endian bytes when the request
	// asked for encoding_format "base64".
	Embedding any `json:"embedding"`
}

type oaEmbeddingUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

type oaEmbeddingResponse struct {
	Object string           `json:"object"`
	Data   []oaEmbedding    `json:"data"`
	Model  string           `json:"model"`
	Usage  oaEmbeddingUsage `json:"usage"`
}

// oaEmbedInputs reads the four shapes `input` takes. An empty string or an
// empty list is refused, as the API refuses it.
func oaEmbedInputs(raw json.RawMessage) ([]Prompt, error) {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		if s == "" {
			return nil, errors.New("input cannot be an empty string")
		}
		return []Prompt{{Kind: PromptText, Text: s}}, nil
	}
	var texts []string
	if json.Unmarshal(raw, &texts) == nil {
		if len(texts) == 0 {
			return nil, errors.New("input cannot be an empty array")
		}
		out := make([]Prompt, len(texts))
		for i, t := range texts {
			if t == "" {
				return nil, fmt.Errorf("input[%d] cannot be an empty string", i)
			}
			out[i] = Prompt{Kind: PromptText, Text: t}
		}
		return out, nil
	}
	var ids []int32
	if json.Unmarshal(raw, &ids) == nil {
		return []Prompt{{Kind: PromptIDs, IDs: ids}}, nil
	}
	var many [][]int32
	if json.Unmarshal(raw, &many) == nil {
		out := make([]Prompt, len(many))
		for i, x := range many {
			out[i] = Prompt{Kind: PromptIDs, IDs: x}
		}
		return out, nil
	}
	return nil, errors.New("input must be a string, an array of strings, an array of token ids, " +
		"or an array of arrays of token ids")
}

func (e *compat) openAIEmbeddings(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		oaFail(w, http.StatusMethodNotAllowed, "POST only", "invalid_request_error")
		return
	}
	var req oaEmbeddingRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		oaFail(w, http.StatusBadRequest, err.Error(), "invalid_request_error")
		return
	}
	inputs, err := oaEmbedInputs(req.Input)
	if err != nil {
		oaFail(w, http.StatusBadRequest, err.Error(), "invalid_request_error")
		return
	}
	b64 := false
	switch req.EncodingFormat {
	case "", "float":
	case "base64":
		b64 = true
	default:
		oaFail(w, http.StatusBadRequest, fmt.Sprintf("encoding_format %q is not one of \"float\", \"base64\"",
			req.EncodingFormat), "invalid_request_error")
		return
	}
	dims := 0
	if req.Dimensions != nil {
		if *req.Dimensions < 1 {
			oaFail(w, http.StatusBadRequest, "dimensions must be at least 1", "invalid_request_error")
			return
		}
		dims = *req.Dimensions
	}
	res, err := e.b.Embed(r.Context(), EmbedOptions{Model: req.Model, Inputs: inputs, Dimensions: dims})
	if err != nil {
		oaFailErr(w, err)
		return
	}
	out := oaEmbeddingResponse{Object: "list", Data: make([]oaEmbedding, len(res.Vectors)), Model: req.Model}
	for i, v := range res.Vectors {
		d := oaEmbedding{Object: "embedding", Index: i, Embedding: v}
		if b64 {
			buf := make([]byte, 4*len(v))
			for j, x := range v {
				binary.LittleEndian.PutUint32(buf[4*j:], math.Float32bits(x))
			}
			d.Embedding = base64.StdEncoding.EncodeToString(buf)
		}
		out.Data[i] = d
	}
	n := res.TotalTokens()
	out.Usage = oaEmbeddingUsage{PromptTokens: n, TotalTokens: n}
	writeJSON(w, http.StatusOK, out)
}
