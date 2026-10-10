package server

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"connectrpc.com/connect"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/format/jlm"
	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// ModelService implements jitllm.v1.ModelService.
type ModelService struct{ E *Engine }

// ListModels lists the loaded models and, unless LoadedOnly is set, the model
// files in the directory, each container checked by opening it.
func (s *ModelService) ListModels(ctx context.Context, req *connect.Request[v1.ListModelsRequest]) (*connect.Response[v1.ListModelsResponse], error) {
	out := &v1.ListModelsResponse{}
	for _, lm := range s.E.Models() {
		out.Loaded = append(out.Loaded, s.E.pbModelInfo(lm))
	}
	if !req.Msg.LoadedOnly {
		files, dir, err := s.E.ScanModels(req.Msg.Directory)
		if err != nil && !os.IsNotExist(err) {
			return nil, connectErr(err)
		}
		out.Directory = dir
		for _, f := range files {
			mf := &v1.ModelFile{
				Path:           f.Path,
				Name:           f.Name,
				Size:           pbBytes(f.Size),
				IsContainer:    f.IsContainer,
				ConvertCommand: f.ConvertCommand,
			}
			if f.IsContainer {
				// jlm.Open refuses any version but this build's, so a
				// successful open proves the version. A container that fails
				// is still listed, with the re-convert fix.
				if h, err := jlm.Open(f.Path); err == nil {
					mf.ContainerVersion = uint32(jlm.Version)
					h.Close()
				} else {
					mf.ConvertCommand = fmt.Sprintf(
						"jitllm convert <source> %s   # %v", f.Path, err)
				}
			}
			out.Files = append(out.Files, mf)
		}
	}
	return connect.NewResponse(out), nil
}

// GetModel describes one loaded model.
func (s *ModelService) GetModel(ctx context.Context, req *connect.Request[v1.GetModelRequest]) (*connect.Response[v1.GetModelResponse], error) {
	lm, err := s.E.Model(req.Msg.ModelId)
	if err != nil {
		return nil, connectErr(err)
	}
	return connect.NewResponse(&v1.GetModelResponse{Model: s.E.pbModelInfo(lm)}), nil
}

// LoadModel opens a container and places it as the request asks, reporting how
// long the load took.
func (s *ModelService) LoadModel(ctx context.Context, req *connect.Request[v1.LoadModelRequest]) (*connect.Response[v1.LoadModelResponse], error) {
	m := req.Msg
	if m.Path == "" {
		return nil, invalid("load: path is required")
	}
	o := LoadOptions{
		Path:            m.Path,
		ModelID:         m.ModelId,
		PageBudgetBytes: m.PageBudgetBytes,
		DeviceIDs:       m.DeviceIds,
		MaxDeviceBlocks: -1,
	}
	if m.MaxDeviceBlocks != nil {
		o.MaxDeviceBlocks = int(*m.MaxDeviceBlocks)
	}
	if m.KvF16 != nil {
		v := *m.KvF16
		o.KVF16 = &v
	}
	start := time.Now()
	lm, err := s.E.LoadModel(o)
	if err != nil {
		return nil, connectErr(err)
	}
	return connect.NewResponse(&v1.LoadModelResponse{
		Model:      s.E.pbModelInfo(lm),
		LoadMillis: millis(time.Since(start)),
	}), nil
}

// UnloadModel closes a model and reports how many sessions it closed: a
// model with open sessions is refused unless Force is set.
func (s *ModelService) UnloadModel(ctx context.Context, req *connect.Request[v1.UnloadModelRequest]) (*connect.Response[v1.UnloadModelResponse], error) {
	n, err := s.E.UnloadModel(req.Msg.ModelId, req.Msg.Force)
	if err != nil {
		return nil, connectErr(err)
	}
	return connect.NewResponse(&v1.UnloadModelResponse{SessionsClosed: int32(n)}), nil
}

// Convert streams a GGUF -> .jlm conversion. convert.FromGGUFs reports no
// progress, so the stream carries the phases the server knows (started,
// writing, done) and a fraction that is only ever 0 or 1.
func (s *ModelService) Convert(ctx context.Context, req *connect.Request[v1.ConvertRequest], st *connect.ServerStream[v1.ConvertResponse]) error {
	m := req.Msg
	if m.SourcePath == "" {
		return invalid("convert: source_path is required")
	}
	src := s.E.ResolvePath(m.SourcePath)
	if _, err := os.Stat(src); err != nil {
		return connectErr(err)
	}
	dst := m.OutputPath
	if dst == "" {
		dst = convert.DestFor(src, filepath.Dir(src))
	} else {
		dst = s.E.ResolvePath(dst)
	}
	if !m.Overwrite {
		if _, err := os.Stat(dst); err == nil {
			return connect.NewError(connect.CodeAlreadyExists,
				fmt.Errorf("convert: %s already exists; set overwrite to replace it", dst))
		}
	}
	mmproj := ""
	if m.MmprojPath != "" {
		mmproj = s.E.ResolvePath(m.MmprojPath)
	}

	if err := st.Send(&v1.ConvertResponse{
		Phase:      "started",
		Detail:     fmt.Sprintf("%s -> %s", src, dst),
		OutputPath: dst,
	}); err != nil {
		return err
	}

	type result struct {
		h   *jlm.Header
		err error
	}
	done := make(chan result, 1)
	go func() {
		fp := jlm.Fingerprint{Writer: jlm.WriterID()}
		var h *jlm.Header
		var err error
		if convert.IsSafetensors(src) {
			if mmproj != "" {
				err = fmt.Errorf("convert: an mmproj second file is a GGUF-only contract")
			} else {
				h, err = convert.FromSafetensors(src, dst, fp)
			}
		} else {
			h, err = convert.FromGGUFs(src, mmproj, dst, fp)
		}
		done <- result{h, err}
	}()

	// A heartbeat, so a client on a 46 GiB model can tell a live conversion
	// from a dead connection. It carries no fraction because there is none.
	tick := time.NewTicker(5 * time.Second)
	defer tick.Stop()
	for {
		select {
		case <-ctx.Done():
			// Giving up on the stream does not cancel the conversion:
			// FromGGUFs takes no context, so the goroutine finishes its
			// write-beside-and-rename on its own.
			return ctx.Err()
		case <-tick.C:
			if err := st.Send(&v1.ConvertResponse{Phase: "writing", OutputPath: dst}); err != nil {
				return err
			}
		case r := <-done:
			if r.err != nil {
				return connectErr(r.err)
			}
			var size uint64
			if fi, err := os.Stat(dst); err == nil {
				size = uint64(fi.Size())
			}
			return st.Send(&v1.ConvertResponse{
				Phase:      "done",
				Fraction:   1,
				Done:       true,
				OutputPath: dst,
				OutputSize: pbBytes(size),
				Detail: fmt.Sprintf("container v%d, %d block(s), page %s",
					jlm.Version, r.h.NBlocks, humanBytes(r.h.PageSize)),
			})
		}
	}
}

// Tokenize encodes text with the model's tokenizer and returns the ids with
// each one's piece.
func (s *ModelService) Tokenize(ctx context.Context, req *connect.Request[v1.TokenizeRequest]) (*connect.Response[v1.TokenizeResponse], error) {
	lm, err := s.E.Model(req.Msg.ModelId)
	if err != nil {
		return nil, connectErr(err)
	}
	if lm.m.Vocab == nil {
		return nil, connectErr(fmt.Errorf("server: model %q has no tokenizer: %v", lm.id, lm.m.TokErr))
	}
	ids := lm.m.Vocab.Encode(req.Msg.Text, req.Msg.AddSpecial)
	pieces := make([]string, len(ids))
	for i, id := range ids {
		pieces[i] = lm.m.Vocab.Text(id)
	}
	return connect.NewResponse(&v1.TokenizeResponse{TokenIds: ids, Pieces: pieces}), nil
}

// Detokenize decodes token ids to text, as a generate's stream decodes them.
func (s *ModelService) Detokenize(ctx context.Context, req *connect.Request[v1.DetokenizeRequest]) (*connect.Response[v1.DetokenizeResponse], error) {
	lm, err := s.E.Model(req.Msg.ModelId)
	if err != nil {
		return nil, connectErr(err)
	}
	if lm.m.Vocab == nil {
		return nil, connectErr(fmt.Errorf("server: model %q has no tokenizer: %v", lm.id, lm.m.TokErr))
	}
	return connect.NewResponse(&v1.DetokenizeResponse{
		Text: lm.m.Vocab.DecodeChat(req.Msg.TokenIds),
	}), nil
}

// ApplyChatTemplate renders messages through the model's chat template and
// returns the prompt text and its token ids. A model with no template is
// refused rather than run as a raw completion.
func (s *ModelService) ApplyChatTemplate(ctx context.Context, req *connect.Request[v1.ApplyChatTemplateRequest]) (*connect.Response[v1.ApplyChatTemplateResponse], error) {
	lm, err := s.E.Model(req.Msg.ModelId)
	if err != nil {
		return nil, connectErr(err)
	}
	if !lm.m.ChatCapable() {
		// Refused rather than run as a raw completion, which would answer
		// fluently and be a different thing from a chat turn.
		return nil, connect.NewError(connect.CodeFailedPrecondition,
			fmt.Errorf("model %q carries no chat template", lm.id))
	}
	msgs := make([]model.ChatMessage, 0, len(req.Msg.Messages))
	for _, m := range req.Msg.Messages {
		msgs = append(msgs, model.ChatMessage{Role: m.Role, Content: m.Content})
	}
	prompt, err := lm.m.ChatPrompt(msgs, req.Msg.AddGenerationPrompt)
	if err != nil {
		return nil, connectErr(err)
	}
	ids, err := lm.m.ChatIDs(msgs, req.Msg.AddGenerationPrompt)
	if err != nil {
		return nil, connectErr(err)
	}
	return connect.NewResponse(&v1.ApplyChatTemplateResponse{Prompt: prompt, TokenIds: ids}), nil
}
