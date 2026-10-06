//go:build linux

package engine

import (
	"os"
	"strings"
	"testing"
	"time"

	"github.com/samyfodil/jitllm/common/session"
	"github.com/samyfodil/jitllm/internal/testmodels"
)

// loadFor opens modelPath on a fresh engine with the given setup applied first.
func loadFor(t *testing.T, setup func(*testStore)) (*Engine, *testShell) {
	t.Helper()
	if _, err := os.Stat(modelPath); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proves nothing without it", err)
	}
	sh := newTestShell()
	if setup != nil {
		setup(sh.Store)
	}
	e := New(sh, sh.state())
	t.Cleanup(e.Close)
	e.Load(modelPath)
	pump(t, sh, 90*time.Second, "the load to finish", func() bool { return !sh.Store.Busy.Get() })
	return e, sh
}

// The header must say when the context was cut to the model's own.
//
// activate clamps the request before the session is built, so the note has
// to compare against what was asked, not the session's own figures.
func TestTheHeaderSaysWhenTheContextWasClamped(t *testing.T) {
	_, sh := loadFor(t, func(st *testStore) { st.MaxSeq.Set(1 << 20); st.DeviceSpec.Set("cpu") })
	if got := sh.Store.ModelSummary.Get(); !strings.Contains(got, "(the most it supports)") {
		t.Errorf("a clamped context is not said: %q", got)
	}
}

// Relocation follows the device choice, as the CLI's -relocate does: on where
// the host may run blocks, off where the choice names devices alone -- moving a
// block onto a CPU the person kept out of it would undo their choice.
func TestRelocationFollowsTheDeviceChoice(t *testing.T) {
	for _, tc := range []struct {
		spec string
		want bool
	}{
		{"auto", true},
		{"cpu", true},
		{"gpu", false},
	} {
		e, sh := loadFor(t, func(st *testStore) { st.DeviceSpec.Set(tc.spec) })
		if !sh.Store.Loaded.Get() {
			if tc.spec == "gpu" {
				t.Logf("-devices gpu: no GPU opened here, so the devices-only arm did not run: %s", sh.Store.Problem.Get().Title)
				continue
			}
			t.Fatalf("%s: the model did not load: %s", tc.spec, sh.Store.Problem.Get().Title)
		}
		reply := sh.Store.AppendTurn(session.Turn{Role: session.RoleAssistant})
		e.Send(session.ChatRequest{Prompt: "Once upon a time", Reply: reply, MaxTokens: 4})
		pump(t, sh, 90*time.Second, "the turn", func() bool { return !sh.Store.Busy.Get() })
		if got := e.sess.Relocating(); got != tc.want {
			t.Errorf("devices %q: relocation %v after a turn, want %v", tc.spec, got, tc.want)
		}
	}
}

// A device that cannot be opened must leave nothing that looks loaded.
//
// activate closes the running session before it opens the device, so a
// failure there must not leave Loaded true under the previous header.
func TestAMissingDeviceLeavesNothingLooksLoaded(t *testing.T) {
	e, sh := loadFor(t, func(st *testStore) { st.DeviceSpec.Set("cpu") })
	if !sh.Store.Loaded.Get() {
		t.Fatal("setup: the model did not load on the CPU")
	}
	// Re-loading an open model re-activates it, on the device asked for.
	sh.Store.DeviceSpec.Set("cuda:9")
	e.Load(modelPath)
	pump(t, sh, 90*time.Second, "the switch to fail", func() bool {
		return !sh.Store.Busy.Get() && sh.Store.Problem.Get().Seq != 0
	})
	if sh.Store.Loaded.Get() {
		t.Error("the app says a model is loaded after its device failed to open")
	}
	if got := sh.Store.ModelSummary.Get(); got != "" {
		t.Errorf("the header still describes the closed session: %q", got)
	}
	if p := sh.Store.Problem.Get(); !strings.Contains(p.Title, "device") {
		t.Errorf("the failure was not reported as a device problem: %+v", p)
	}
	if len(sh.Store.Models.Get()) != 1 || sh.Store.Active.Get() != "" {
		t.Errorf("the opened model should be listed and inactive: %+v active %q", sh.Store.Models.Get(), sh.Store.Active.Get())
	}
}

// A model that loads must take down the banner of the problem it answers.
func TestALoadThatWorksClearsTheProblem(t *testing.T) {
	_, sh := loadFor(t, func(st *testStore) {
		st.DeviceSpec.Set("cpu")
		st.Problem.Set(session.Problem{Seq: 1, Title: "The device setting doesn't fit this machine"})
	})
	if !sh.Store.Loaded.Get() {
		t.Fatal("setup: the model did not load")
	}
	if p := sh.Store.Problem.Get(); p.Seq != 0 {
		t.Errorf("a successful load left the banner up: %q", p.Title)
	}
}
