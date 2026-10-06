package engine

import (
	"fmt"

	"github.com/samyfodil/jitllm/common/session"
	"github.com/samyfodil/jitllm/engine/model"
	"github.com/samyfodil/jitllm/engine/sched"
)

// entry is one open model.
//
// Only the active entry has a session, a device and a worker pool. Every
// model.State builds its own pool of sched.DecodeCores() workers, and a second
// live pool is a contended core by construction (every region ends at a
// barrier). A tier is also one model: its layer map is keyed by block index,
// so two models offering block 0 overwrite each other. And the JIT's codegen
// knobs are process-wide, last writer wins.
//
// So several models are open and one runs: weights and pager stay alive for
// every entry; the session, device and pool belong to the one in use.
type entry struct {
	m    *model.Model
	path string
	// grant is the host page budget this model currently holds, and pin the
	// cap the user set for it (0 = divide automatically).
	grant, pin uint64
}

// find returns the open entry for path, or nil.
func (e *Engine) find(path string) *entry {
	for _, en := range e.models {
		if en.path == path {
			return en
		}
	}
	return nil
}

// paths is the open models in load order, which is what first-come-first-served
// means.
func (e *Engine) paths() []string {
	out := make([]string, 0, len(e.models))
	for _, en := range e.models {
		out = append(out, en.path)
	}
	return out
}

// rebudget re-divides the host budget and applies it to every open model.
//
// Model.SetPageBudget re-caps residency in bytes and can widen as well as
// narrow. It is called on every load, unload and switch, because a share is a
// function of the whole set.
func (e *Engine) rebudget() {
	if len(e.models) == 0 {
		return
	}
	active := ""
	if e.active != nil {
		active = e.active.path
	}
	pins := make(map[string]uint64, len(e.models))
	for _, en := range e.models {
		if en.pin > 0 {
			pins[en.path] = en.pin
		}
	}
	for p, n := range shares(sched.MemBudget(), e.paths(), active, e.priority, pins) {
		en := e.find(p)
		if en == nil || en.m == nil {
			continue
		}
		en.grant = n
		en.m.SetPageBudget(n)
	}
}

// publishModels mirrors the registry into the store.
func (e *Engine) publishModels() {
	out := make([]session.LoadedModel, 0, len(e.models))
	for i, en := range e.models {
		lm := session.LoadedModel{
			Path:   en.path,
			Name:   short(en.path),
			Colour: i,
			Grant:  en.grant,
			Pinned: en.pin > 0,
			Active: e.active == en,
		}
		if en.m != nil {
			lm.Arch = en.m.Cfg.Arch
			lm.Blocks = en.m.Cfg.NLayer
		}
		out = append(out, lm)
	}
	active := ""
	if e.active != nil {
		active = e.active.path
	}
	e.sh.Post(func() {
		e.st.Models.Set(out)
		e.st.Active.Set(active)
	})
}

// Use makes an already-open model the active one. Loading it is Load's job.
func (e *Engine) Use(path string) {
	if _, ok := session.LoadingOf(e.st.Loads, path); ok {
		return
	}
	e.busyBegin()
	session.QueueLoad(e.st.Loads, session.Loading{Path: path, Stage: "Switching models"})
	ok := e.post("use", func() {
		defer e.busyEnd()
		defer e.loading(path)()
		en := e.find(path)
		if en == nil {
			e.status("%s is not open", short(path))
			return
		}
		if e.active == en {
			return
		}
		e.status("switching to %s", short(path))
		if err := e.activate(en); err != nil {
			e.report(stageDevice, path, err)
		}
	})
	if !ok {
		e.busyEnd()
		session.EndLoad(e.st.Loads, path)
	}
}

// Unload closes one model. Closing the active one leaves nothing running.
func (e *Engine) Unload(path string) {
	e.post("unload", func() {
		en := e.find(path)
		if en == nil {
			return
		}
		if e.active == en {
			e.releaseSession()
			e.active = nil
		}
		en.m.Close()
		for i, x := range e.models {
			if x == en {
				e.models = append(e.models[:i], e.models[i+1:]...)
				break
			}
		}
		// Give the bytes back: the remaining models grow into what this one held.
		e.rebudget()
		e.publishModels()
		if e.active == nil {
			e.sh.Post(func() { e.st.Loaded.Set(false); e.st.Vision.Set(false) })
		}
		e.status("closed %s", short(path))
	})
}

// Pin caps one model's host budget, or releases the cap at 0.
func (e *Engine) Pin(path string, bytes uint64) {
	e.post("pin", func() {
		en := e.find(path)
		if en == nil {
			return
		}
		en.pin = bytes
		e.rebudget()
		e.publishModels()
		if bytes == 0 {
			e.status("%s: budget divided automatically again", short(path))
			return
		}
		e.status("%s: capped at %s", short(path), session.Bytes(bytes))
	})
}

// SetPriority toggles whether the active model gets the lion's share.
func (e *Engine) SetPriority(on bool) {
	e.post("priority", func() {
		e.priority = on
		e.rebudget()
		e.publishModels()
		if on {
			e.status("the model in use gets the memory; switching moves it")
			return
		}
		e.status("memory divided in load order")
	})
}

// errNoModel is returned when a switch is asked for with nothing to switch to.
var errNoModel = fmt.Errorf("no model is open")

// activeName is the running model's display name and palette index, for the
// turn it is about to produce.
func (e *Engine) activeName() (string, int) {
	if e.active == nil {
		return "", 0
	}
	for i, en := range e.models {
		if en == e.active {
			return short(en.path), i
		}
	}
	return short(e.active.path), 0
}
