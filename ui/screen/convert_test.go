package screen

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/common/catalog"
	"github.com/jitllm/jitllm/common/convertjob"
	"github.com/jitllm/jitllm/ui/app"
)

// convertRows is one of each kind of row the Convert table shows.
func convertRows(dir string) (fresh, done, old, tower catalog.Entry) {
	mk := func(name string, v uint32, tw bool) catalog.Entry {
		p := filepath.Join(dir, name)
		return catalog.Entry{Path: p, Name: name, Kind: catalog.KindGGUF,
			Target: strings.TrimSuffix(p, ".gguf") + ".jlm", TargetVersion: v, Tower: tw}
	}
	return mk("fresh.gguf", 0, false), mk("done.gguf", catalog.CurrentVersion, false),
		mk("old.gguf", 17, false), mk("mmproj-x.gguf", 0, true)
}

// Picking a row puts it in the field it belongs in: a model as the source,
// with its destination derived, and a vision tower as the tower -- which does
// not throw away the source already chosen.
func TestPickingARowFillsTheFieldItBelongsIn(t *testing.T) {
	fresh, _, _, tower := convertRows(t.TempDir())
	sh := dropShell(t)
	waitForScan(t, modelsFor(sh))
	sh.Store.SetSources([]catalog.Entry{fresh, tower})
	c := convertFor(sh)

	c.pick(0)
	if c.src.Get() != fresh.Path || c.dst.Get() != DestFor(fresh.Path, primaryDir(sh.Cfg.ModelDirs)) {
		t.Fatalf("picking a model filled src %q dst %q", c.src.Get(), c.dst.Get())
	}
	c.pick(1)
	if c.mmproj.Get() != tower.Path {
		t.Fatalf("picking a vision tower filled mmproj %q", c.mmproj.Get())
	}
	if c.src.Get() != fresh.Path {
		t.Fatalf("picking the tower replaced the source with %q", c.src.Get())
	}

	// Nothing moves under a running conversion.
	c.converting.Set(true)
	c.pick(0)
	if c.mmproj.Get() != tower.Path {
		t.Fatal("a pick during a conversion changed the form")
	}
}

// A rescan republishes the rows and must not touch the form: only a person's
// selection fills it.
func TestARepublishLeavesTheFormAlone(t *testing.T) {
	fresh, done, _, _ := convertRows(t.TempDir())
	sh := dropShell(t)
	waitForScan(t, modelsFor(sh))
	c := convertFor(sh)
	c.src.Set("/typed/by/hand.gguf")
	publishSources(sh.Store, []catalog.Entry{fresh, done})
	sh.Store.SourcesSel.Set(rowOf(sh.Store.Sources.Get(), fresh.Path))
	// One more file, sorting first, moves fresh down a row.
	early := catalog.Entry{Path: "/m/a.gguf", Name: "a.gguf", Kind: catalog.KindGGUF}
	publishSources(sh.Store, []catalog.Entry{done, fresh, early})
	if c.src.Get() != "/typed/by/hand.gguf" {
		t.Fatalf("a republish changed the form's source to %q", c.src.Get())
	}
	if e, ok := sh.Store.SelectedSource(); !ok || e.Path != fresh.Path {
		t.Fatalf("the selection did not follow the file across a republish: %+v", e)
	}
}

// A row says what converting it would do, and every word fits the row's
// right-hand column.
func TestARowSaysWhatConvertingWouldDo(t *testing.T) {
	size := app.NewShell(nil, nil).P.Type.BodySmall.FontSize
	fresh, done, old, tower := convertRows(t.TempDir())
	newer := done
	newer.TargetVersion = catalog.CurrentVersion + 1
	for _, c := range []struct {
		e    catalog.Entry
		want string
	}{{fresh, "none"}, {done, "converted"}, {old, "outdated"}, {newer, "newer"}, {tower, "--"}} {
		if got := convertjob.SourceContainer(c.e); got != c.want {
			t.Errorf("%s: %q, want %q", c.e.Name, got, c.want)
		}
		if w := realTextWidth(t, convertjob.SourceState(c.e), size); w > rowSideWidth {
			t.Errorf("%q is %.0f px in a %d px column", convertjob.SourceState(c.e), w, rowSideWidth)
		}
	}
}

// Filling a source whose container already exists says so, before a Convert
// replaces it.
func TestFillingASourceSaysWhatItWouldReplace(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "m.gguf")
	writeZeros(t, src, 64)
	sh := dropShell(t)
	c := convertFor(sh)

	c.fillSource(src)
	if c.note.Get() != "" {
		t.Fatalf("a source with no container got a note: %q", c.note.Get())
	}
	writeStaleContainer(t, DestFor(src, primaryDir(sh.Cfg.ModelDirs)), catalog.CurrentVersion)
	c.fillSource(src)
	if !strings.Contains(c.note.Get(), "already exists and loads") {
		t.Errorf("a current container is not mentioned: %q", c.note.Get())
	}
	writeStaleContainer(t, DestFor(src, primaryDir(sh.Cfg.ModelDirs)), 17)
	c.fillSource(src)
	if !strings.Contains(c.note.Get(), "will not load") {
		t.Errorf("an outdated container is not mentioned: %q", c.note.Get())
	}
}

// The hardware probe landing must not empty the Convert list. Republishing the
// Models rows on it -- so their fit words updated -- once handed publish the
// containers alone, and publish took that for the whole scan.
func TestAProbeLandingKeepsTheSources(t *testing.T) {
	fresh, done, _, _ := convertRows(t.TempDir())
	sh := dropShell(t)
	m := modelsFor(sh)
	waitForScan(t, m)
	sh.DrainQueue(0) // the scan's own results, so they do not land after ours
	m.publish([]catalog.Entry{{Path: "/m/a.jlm", Name: "a.jlm", Kind: catalog.KindContainer}, fresh, done})
	sh.Store.Machine.Set(app.MachineReport{Probed: true})
	sh.DrainQueue(0)
	if n := sh.Store.SourcesRows.Get(); n != 2 {
		t.Fatalf("the probe landing left %d source row(s), want 2", n)
	}
}
