package screen

import (
	"path/filepath"
	"strings"

	"github.com/samyfodil/jitllm/common/catalog"
	"github.com/samyfodil/jitllm/ui/app"
)

// Drag and drop. The handler lives in this package because a drop fills
// screen state (the convert form, the attachments). Shell.OnFilesDropped
// holds one handler, so it classifies and routes: pictures to the session,
// models to the catalog, and only a model drop switches tabs.

// Drop is what a set of dropped paths means to this app. At most one file of
// each role is acted on (the first), since four GGUFs cannot fill one form.
type Drop struct {
	// Container is a .jlm to reveal in the table.
	Container string
	// Source is a .gguf or .safetensors to convert.
	Source string
	// Images are pictures to attach to the next message.
	Images []string
	// Ignored is the base name of every dropped path this app does not read.
	Ignored []string
}

// Empty reports whether nothing actionable was dropped.
func (d Drop) Empty() bool { return d.Container == "" && d.Source == "" && len(d.Images) == 0 }

// ClassifyDrop sorts dropped paths by what the app can do with each. It is a
// pure function of the names, classifying through [catalog.KindOf] like the
// directory scan, so the table and the window never disagree.
func ClassifyDrop(paths []string) Drop {
	var d Drop
	for _, p := range paths {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		if IsImage(p) {
			d.Images = append(d.Images, p)
			continue
		}
		switch catalog.KindOf(p) {
		case catalog.KindContainer:
			if d.Container == "" {
				d.Container = p
			}
		case catalog.KindGGUF, catalog.KindSafetensors:
			if d.Source == "" {
				d.Source = p
			}
		default:
			d.Ignored = append(d.Ignored, filepath.Base(p))
		}
	}
	return d
}

// DropFiles is the window's drag-and-drop handler. Wire it once:
//
//	sh.OnFilesDropped(func(paths []string) { screen.DropFiles(sh, paths) })
//
// It runs on the UI goroutine, which is what [modelsScreen.rescan] requires.
func DropFiles(sh *app.Shell, paths []string) {
	d := ClassifyDrop(paths)
	if d.Empty() {
		if len(d.Ignored) > 0 {
			sh.SetStatus("not a file this app reads: " + strings.Join(d.Ignored, ", ") +
				" (models: .jlm, .gguf, .safetensors; pictures: .png, .jpg)")
		}
		return
	}
	if len(d.Images) > 0 {
		AttachImages(sh, d.Images)
		sh.GoSession()
	}
	if d.Source == "" && d.Container == "" {
		return
	}

	// A container lands on Models and a source on Convert; with both, the
	// source is the action and Convert is where the drop leaves you.
	if d.Container != "" {
		sh.GoModels()
		modelsFor(sh).reveal(d.Container)
	}
	if d.Source != "" {
		c := convertFor(sh)
		sh.GoConvert()
		c.selectSource(d.Source)
		c.fillSource(d.Source)
	}
}

// reveal selects a container in the table, pulling its folder into the
// catalog first if the scan has not seen the file. It does not load: a drop
// is "look at this".
func (m *modelsScreen) reveal(path string) {
	if m.selectPath(path) {
		m.sh.SetStatus("selected " + filepath.Base(path))
		return
	}

	dir := filepath.Dir(path)
	if m.sh.AddModelDir(dir) {
		m.sh.SetStatus("added " + dir + " to the catalog; rescanning for " + filepath.Base(path))
	} else {
		m.sh.SetStatus("rescanning " + dir + " for " + filepath.Base(path))
	}
	m.rescan()
}
