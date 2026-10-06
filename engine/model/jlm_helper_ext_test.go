package model_test

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/samyfodil/jitllm/convert/gguf"
	"github.com/samyfodil/jitllm/internal/testmodels"

	"github.com/samyfodil/jitllm/convert"
	"github.com/samyfodil/jitllm/engine/model"
	"github.com/samyfodil/jitllm/format/jlm"
)

var convertOnceExt sync.Map

// convertResultExt is what the cache holds; see convertResult.
type convertResultExt struct {
	once sync.Once
	err  error
}

// jlmOf is the external test package's twin of the internal jlmOf (see
// jlm_helper_test.go): convert a GGUF to a container beside it, failing
// rather than skipping when conversion fails.
func jlmOf(t testing.TB, path string) string {
	t.Helper()
	if strings.HasSuffix(path, jlm.Ext) {
		return path
	}
	dst := strings.TrimSuffix(path, filepath.Ext(path)) + jlm.Ext
	if _, err := os.Stat(path); err != nil {
		// The container is the model and its source is disposable; see
		// containerAlone in jlm_helper_test.go, which this mirrors.
		c, err := jlm.Open(dst)
		if err != nil {
			t.Skipf("MODEL MISSING: %s, and its container %s does not open (%v) (set JITLLM_MODELS to the model directory) -- this "+
				"gate proved nothing", path, dst, err)
		}
		c.Close()
		return dst
	}
	v, _ := convertOnceExt.LoadOrStore(dst, &convertResultExt{})
	r := v.(*convertResultExt)
	r.once.Do(func() {
		si, err := os.Stat(path)
		if err != nil {
			r.err = err
			return
		}
		// Cached, unless this build cannot read the container or an earlier
		// build wrote it; see the same check in jlmOfPair.
		if di, err := os.Stat(dst); err == nil && di.ModTime().After(si.ModTime()) {
			if c, err := jlm.Open(dst); err == nil {
				stale := c.FP.Writer != model.BuildIDForTest()
				c.Close()
				if !stale {
					return
				}
			}
		}
		if _, err := convert.FromGGUF(path, dst, jlm.Fingerprint{Host: "test", Writer: model.BuildIDForTest()}); err != nil {
			r.err = err
		}
	})
	if r.err != nil && !errors.Is(r.err, convert.ErrNotImplemented) {
		t.Fatalf("convert %s: %v", path, r.err)
	}
	if r.err != nil {
		t.Skipf("NOT IMPLEMENTED: %s -- %v (RULE 7 scope)", filepath.Base(path), r.err)
	}
	return dst
}

// languageModels is the internal helper's twin for the external test package:
// the language models available, projectors left out by
// general.architecture.
func languageModels(t testing.TB) []string {
	t.Helper()
	paths := testmodels.Glob("*.gguf")
	if len(paths) == 0 {
		// No GGUF is not "no models": a host may hold only containers.
		return containerModels(t)
	}
	out := make([]string, 0, len(paths))
	for _, p := range paths {
		f, err := gguf.Open(p)
		if err != nil {
			continue
		}
		arch, _ := f.KV["general.architecture"].String()
		f.Close()
		if arch == "clip" {
			continue
		}
		out = append(out, p)
	}
	if len(out) == 0 {
		t.Fatal("every GGUF on this box is a projector; this sweep would prove nothing")
	}
	return out
}

// containerModels is languageModels' arm for a host that holds .jlm and no GGUF.
// The paths it returns are containers, which jlmOf passes straight through.
func containerModels(t testing.TB) []string {
	t.Helper()
	paths := testmodels.Glob("*.jlm")
	if len(paths) == 0 {
		testmodels.Missing(t, "%s", "no models under "+testmodels.Dir()+" (set JITLLM_MODELS to the model directory) -- RULE 11: fetch some, do not record the absence")
	}
	out := make([]string, 0, len(paths))
	for _, p := range paths {
		f, err := jlm.Open(p)
		if err != nil {
			// A container this build cannot read is stale; say so.
			t.Logf("skipping %s: %v", filepath.Base(p), err)
			continue
		}
		c := f.Config()
		lm := c != nil && c.Arch != jlm.ArchNone && c.NLayer > 0
		f.Close()
		if lm {
			out = append(out, p)
		}
	}
	if len(out) == 0 {
		t.Fatal("every container on this box is a projector or unreadable; this sweep would prove nothing")
	}
	return out
}
