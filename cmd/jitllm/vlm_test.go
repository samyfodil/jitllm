package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/samyfodil/jitllm/convert"
	"github.com/samyfodil/jitllm/engine/model"
	"github.com/samyfodil/jitllm/format/jlm"
	"github.com/samyfodil/jitllm/internal/testmodels"
)

// jlmOfPair converts a text model and its vision tower into one container
// beside the source and returns it, because the engine reads nothing else; a
// missing .jlm is built rather than skipped (RULE 11). The two-file mmproj
// contract is llama.cpp's and dies at conversion.
func jlmOfPair(t *testing.T, path, mmproj string) string {
	t.Helper()
	dst := strings.TrimSuffix(path, filepath.Ext(path)) + jlm.Ext
	if mmproj != "" {
		dst = strings.TrimSuffix(path, filepath.Ext(path)) + "-vlm" + jlm.Ext
	}
	si, err := os.Stat(path)
	_, merr := os.Stat(mmproj)
	if err != nil || mmproj != "" && merr != nil {
		// The container is the model and its source is disposable; see
		// containerAlone in engine/model/jlm_helper_test.go, which this mirrors.
		c, oerr := jlm.Open(dst)
		if oerr != nil {
			t.Skipf("MODEL MISSING: %s, and its container %s does not open (%v) (set JITLLM_MODELS to the model directory) -- this "+
				"gate proved nothing", path, dst, oerr)
		}
		c.Close()
		return dst
	}
	// Newer than its source and readable by this build: a version bump leaves
	// the first true and the second false.
	if di, err := os.Stat(dst); err == nil && di.ModTime().After(si.ModTime()) {
		if c, err := jlm.Open(dst); err == nil {
			c.Close()
			return dst
		}
	}
	if _, err := convert.FromGGUFs(path, mmproj, dst, jlm.Fingerprint{Host: "test"}); err != nil {
		t.Fatalf("convert %s: %v", path, err)
	}
	return dst
}

// TestVLMTowerSizesTheState: the state must be sized for the spliced prompt,
// not the text one. The marker ids are held by
// model.TestChatSpansFollowTheTemplate.
func TestVLMTowerSizesTheState(t *testing.T) {
	text := testmodels.Path("SmolVLM-256M-Instruct-Q8_0.gguf")
	m2, err := model.Open(jlmOfPair(t, text, testmodels.Path("mmproj-SmolVLM-256M-Instruct-Q8_0.gguf")))
	if err != nil {
		t.Skipf("MODEL MISSING: no VLM at %s (set JITLLM_MODELS to the model directory) -- this gate proved nothing", text)
	}
	defer m2.Close()
	tw := m2.Tower()
	if tw == nil {
		t.Fatal("the merged container carries no tower")
	}
	if got, want := tw.Cfg.Tokens(), 64; got != want {
		t.Errorf("tower emits %d embeddings, want %d", got, want)
	}
}
