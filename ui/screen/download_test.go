package screen

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/samyfodil/jitllm/common/convertjob"
	"github.com/samyfodil/jitllm/common/discover"
	"github.com/samyfodil/jitllm/convert/hf"
	"github.com/samyfodil/jitllm/convert/library"
	"github.com/samyfodil/jitllm/internal/testmodels"
	"github.com/samyfodil/jitllm/ui/app"
)

// firstText is the first library entry with no vision tower.
func firstText(t *testing.T) (library.Model, int) {
	for i, m := range library.Models {
		if !m.Vision() {
			return m, i
		}
	}
	t.Fatal("the library has no text model")
	return library.Model{}, -1
}

// waitIdle drains the UI queue until the download screen is no longer busy.
func waitIdle(t *testing.T, sh *app.Shell, d *downloadScreen) {
	t.Helper()
	deadline := time.Now().Add(60 * time.Second)
	for time.Now().Before(deadline) {
		sh.DrainQueue(0)
		if !d.busy.Get() {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("still busy after a minute; note %q", d.note.Get())
}

// Discover lists every library model exactly once, whatever the balance.
func TestDiscoverListsTheWholeLibrary(t *testing.T) {
	d := downloadsFor(dropShell(t))
	for pref := range discover.BalanceSteps {
		d.pref.Set(pref)
		seen := map[int]bool{}
		for _, i := range d.order().Get() {
			if seen[i] {
				t.Fatalf("balance %d lists %s twice", pref, library.Models[i].Title)
			}
			seen[i] = true
		}
		if len(seen) != len(library.Models) {
			t.Fatalf("balance %d lists %d of %d models", pref, len(seen), len(library.Models))
		}
	}
}

// The disk is checked before a byte moves.
func TestADownloadThatWouldFillTheDiskDoesNotStart(t *testing.T) {
	sh := dropShell(t)
	d := downloadsFor(sh)
	m, i := firstText(t)
	d.sel.Set(i)
	fetched := false
	defer func(f func(*hf.Client, hf.Ref, string) (string, error), g func(string) (int64, bool)) {
		downloadFetch, downloadFree = f, g
	}(downloadFetch, downloadFree)
	downloadFetch = func(*hf.Client, hf.Ref, string) (string, error) { fetched = true; return "", nil }
	downloadFree = func(string) (int64, bool) { return m.Bytes, true } // half of what it needs

	d.act()
	if fetched || d.busy.Get() {
		t.Fatal("a download started with too little disk")
	}
	if !strings.Contains(d.note.Get(), "Not enough disk space") {
		t.Errorf("the note does not say why: %q", d.note.Get())
	}
}

// A gated repository says what to do, not what HTTP said.
func TestAGatedModelSaysHowToGetIt(t *testing.T) {
	sh := dropShell(t)
	d := downloadsFor(sh)
	m, i := firstText(t)
	d.sel.Set(i)
	defer func(f func(*hf.Client, hf.Ref, string) (string, error), g func(string) (int64, bool)) {
		downloadFetch, downloadFree = f, g
	}(downloadFetch, downloadFree)
	downloadFree = func(string) (int64, bool) { return 1 << 50, true }
	downloadFetch = func(*hf.Client, hf.Ref, string) (string, error) {
		return "", fmt.Errorf("hf://x: HTTP 401: %w", hf.ErrAccess)
	}

	d.act()
	waitIdle(t, sh, d)
	note := d.note.Get()
	for _, want := range []string{"gated or private", "huggingface.co/" + m.Repo, "HF_TOKEN"} {
		if !strings.Contains(note, want) {
			t.Errorf("the note %q does not say %q", note, want)
		}
	}
}

// A download ends as a container on the Models tab, with the GGUF it fetched
// gone and the row marked on disk.
func TestADownloadEndsAsAContainer(t *testing.T) {
	src := testmodels.Path("stories260K.gguf")
	b, err := os.ReadFile(src)
	if err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proves nothing without it", err)
	}
	sh := dropShell(t)
	queueRunsNow(sh)
	d := downloadsFor(sh)
	m, i := firstText(t)
	d.sel.Set(i)
	defer func(f func(*hf.Client, hf.Ref, string) (string, error), g func(string) (int64, bool)) {
		downloadFetch, downloadFree = f, g
	}(downloadFetch, downloadFree)
	downloadFree = func(string) (int64, bool) { return 1 << 50, true }
	var landed string
	downloadFetch = func(cl *hf.Client, r hf.Ref, dir string) (string, error) {
		landed = filepath.Join(dir, filepath.Base(r.File))
		cl.OnProgress(int64(len(b))/2, int64(len(b)))
		return landed, os.WriteFile(landed, b, 0o644)
	}

	d.act()
	waitIdle(t, sh, d)
	if !convertjob.LoadableAt(d.containerFor(m)) {
		t.Fatalf("no container at %s; note %q", d.containerFor(m), d.note.Get())
	}
	if _, err := os.Stat(landed); err == nil {
		t.Error("the GGUF this download fetched is still on disk beside its container")
	}
	if !d.onDisk(i) || d.statusOf(i) != "On this machine" {
		t.Error("the row does not say the model is on disk")
	}
	if !strings.Contains(d.note.Get(), "is ready") {
		t.Errorf("the note does not say it is ready: %q", d.note.Get())
	}
}

// queueRunsNow installs a runner that runs each job at once, as the engine's
// queue would with nothing ahead of it.
func queueRunsNow(sh *app.Shell) {
	Attach(sh, Deps{Engine: &fakeEngine{run: func(job func()) bool { job(); return true }}})
}
