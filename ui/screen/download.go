package screen

import (
	"errors"
	"os"
	"sync"

	"github.com/gogpu/ui/state"

	"github.com/samyfodil/jitllm/common/discover"
	"github.com/samyfodil/jitllm/convert/hf"
	"github.com/samyfodil/jitllm/convert/library"
	"github.com/samyfodil/jitllm/ui/app"
)

// The Download screen: the models jitllm fetches by name, one press from the
// Hub to a container on the Models screen. What is offered and how it is
// fetched is [discover]'s.

// downloadScreen is this screen's state, keyed by shell like modelsScreen's.
type downloadScreen struct {
	sh *app.Shell

	sel      state.Signal[int]
	rows     state.Signal[int]
	have     state.Signal[[]bool] // a current container for each row is on disk
	busy     state.Signal[bool]   // one download at a time
	progress state.Signal[float64]
	note     state.Signal[string]
	pref     state.Signal[int] // the balance, an index into discover.BalanceSteps
	picked   bool              // a person chose a model; stop following the top
	ord      state.ReadonlySignal[[]int]
}

var (
	downloadsMu sync.Mutex
	downloads   = map[*app.Shell]*downloadScreen{}
)

func downloadsFor(sh *app.Shell) *downloadScreen {
	downloadsMu.Lock()
	defer downloadsMu.Unlock()
	d, ok := downloads[sh]
	if !ok {
		d = &downloadScreen{
			sh:       sh,
			sel:      state.NewSignal(-1),
			rows:     state.NewSignal(len(library.Models)),
			have:     state.NewSignal([]bool(nil)),
			busy:     state.NewSignal(false),
			progress: state.NewSignal(0.0),
			note:     state.NewSignal(""),
			pref:     state.NewSignal(discover.DefaultBalance),
		}
		d.refresh()
		downloads[sh] = d
	}
	return d
}

// dir is where downloads land; see [discover.Dir].
func (d *downloadScreen) dir() string { return discover.Dir(d.sh.Cfg.ModelDirs) }

// containerFor is where a library model's container is written.
func (d *downloadScreen) containerFor(m library.Model) string {
	return discover.ContainerFor(d.dir(), m)
}

// refresh re-reads which models are already on disk.
func (d *downloadScreen) refresh() { d.have.Set(discover.OnDisk(d.dir())) }

func (d *downloadScreen) selected() (library.Model, int, bool) {
	i := d.sel.Get()
	if i < 0 || i >= len(library.Models) {
		return library.Model{}, i, false
	}
	return library.Models[i], i, true
}

func (d *downloadScreen) onDisk(i int) bool {
	have := d.have.Get()
	return i >= 0 && i < len(have) && have[i]
}

// --- the action ----------------------------------------------------------------

// Seams for the tests: the Hub and the disk.
var (
	downloadFetch discover.FetchFunc = discover.HubFetch
	downloadFree  discover.FreeFunc  = discover.FreeBytes
)

// act is the button: Load a model that is on disk, fetch one that is not.
func (d *downloadScreen) act() {
	m, i, ok := d.selected()
	if !ok || d.busy.Get() {
		return
	}
	if d.onDisk(i) {
		if eng := depsOf(d.sh).Engine; eng != nil {
			eng.Load(d.containerFor(m))
			d.sh.GoSession()
		}
		return
	}
	d.fetch(m)
}

// fetch downloads m (and its vision tower), converts it on the engine's
// queue, removes the GGUF it downloaded, and says where the model is. The
// steps are [discover]'s; this puts them on the screen.
func (d *downloadScreen) fetch(m library.Model) {
	dir := d.dir()
	dst := d.containerFor(m)
	if note := discover.SpaceNote(m, dir, downloadFree); note != "" {
		d.note.Set(note)
		return
	}

	d.busy.Set(true)
	d.progress.Set(0)
	d.note.Set("Downloading " + m.Title + "...")
	d.sh.SetStatus("downloading " + m.Name)

	go func() {
		got, err := discover.Download(m, dir, hf.FindToken(os.Getenv, os.UserHomeDir), downloadFetch,
			func(f float64) { d.sh.Post(func() { d.progress.Set(f) }) })
		if err != nil {
			d.fail(m, err)
			return
		}
		d.sh.Post(func() { d.note.Set("Converting " + m.Title + " to a .jlm (the bar is an estimate)...") })
		took := modelsFor(d.sh).runner()(func() {
			if err := RunConvert(d.sh, got.Request(dst)); err != nil {
				d.fail(m, err)
				return
			}
			got.Cleanup()
			d.sh.Post(func() {
				d.refresh()
				modelsFor(d.sh).rescan()
				d.busy.Set(false)
				d.note.Set(m.Title + " is ready. Press Load, or find it on the Models tab.")
				d.sh.SetStatus("downloaded " + m.Name)
			})
		})
		if !took {
			d.fail(m, errors.New("the engine is busy"))
		}
	}()
}

// fail says what went wrong in words a person can act on.
func (d *downloadScreen) fail(m library.Model, err error) {
	msg := discover.FailMessage(m, err)
	d.sh.Post(func() {
		d.busy.Set(false)
		d.note.Set(msg)
		d.sh.SetStatus("download failed: " + m.Name)
	})
}

// ShowDiscover switches to Discover with the named model selected. Call it
// on the UI goroutine.
func ShowDiscover(sh *app.Shell, name string) {
	d := downloadsFor(sh)
	for i, m := range library.Models {
		if m.Name == name {
			d.picked = true
			d.sel.Set(i)
		}
	}
	sh.SelectTab(app.TabDownload)
}
