//go:build linux

package engine

import (
	"os"
	"strings"
	"testing"
	"time"

	"github.com/samyfodil/jitllm/internal/testmodels"
)

// A file the engine cannot open must reach the session as a problem with its
// fix, through the real engine -- not as a raw error in a two-line dialog.
//
// explain is gated on its own; this gates the wiring. Against the violation
// (an Alert and a status line) no problem is ever reported.
func TestAFileTheEngineCannotOpenBecomesAProblemWithItsFix(t *testing.T) {
	cases := []struct{ path, title, action string }{
		{testmodels.Path("gemma-2b-v13.jlm"), "older jitllm", "Reconvert"},
		{testmodels.Path("stories260K.gguf"), "converting first", "Convert…"},
	}
	for _, c := range cases {
		if _, err := os.Stat(c.path); err != nil {
			testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- RULE 11, fetch it; this gate proves nothing without it", err)
		}
		sh := newTestShell()
		e := New(sh, sh.state())
		e.Load(c.path)
		pump(t, sh, 30*time.Second, "the problem to be reported", func() bool {
			return sh.Store.Problem.Get().Seq != 0 && !sh.Store.Busy.Get()
		})
		p := sh.Store.Problem.Get()
		if !strings.Contains(p.Title, c.title) || p.Action != c.action || p.Do == nil {
			t.Errorf("%s: reported %q with action %q (Do set %v), want %q / %q",
				c.path, p.Title, p.Action, p.Do != nil, c.title, c.action)
		}
		if sh.Store.Loaded.Get() {
			t.Errorf("%s: the app says a model is loaded", c.path)
		}
		e.Close()
	}
}
