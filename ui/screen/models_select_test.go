package screen

import (
	"testing"
	"time"

	"github.com/jitllm/jitllm/common/catalog"
)

// A scan publishes once per probed container, and a file found on the way
// shifts every row after it. Store.SetCatalog keeps the selected index, so the
// selection has to be carried across a publish by path.
func TestPublishKeepsTheSelectionOnTheFileAndNotTheRow(t *testing.T) {
	sh := dropShell(t)
	m := modelsFor(sh)
	waitForScan(t, m)

	rows := []catalog.Entry{
		{Path: "/m/a.jlm", Name: "a.jlm", Kind: catalog.KindContainer},
		{Path: "/m/b.jlm", Name: "b.jlm", Kind: catalog.KindContainer},
		{Path: "/m/c.jlm", Name: "c.jlm", Kind: catalog.KindContainer},
	}
	m.publish(rows)
	sh.Store.CatalogSel.Set(2) // c.jlm

	// A rescan that found one more file, which sorts first: c.jlm is now row 3.
	m.publish(append([]catalog.Entry{{Path: "/m/0.jlm", Name: "0.jlm", Kind: catalog.KindContainer}}, rows...))

	if got := sh.Store.CatalogSel.Get(); got != 3 {
		t.Fatalf("CatalogSel = %d; c.jlm moved to row 3 and the selection must follow it", got)
	}
	e, ok := sh.Store.SelectedEntry()
	if !ok || e.Path != "/m/c.jlm" {
		t.Fatalf("SelectedEntry = %+v (ok=%v); want c.jlm", e, ok)
	}
}

// The file going away is the one case where losing the selection is right.
// The table stays the same length on purpose: SetCatalog already drops an
// index past the end, so the case covered is a deleted file whose row number
// is still valid.
func TestPublishDropsASelectionWhoseFileIsGone(t *testing.T) {
	sh := dropShell(t)
	m := modelsFor(sh)
	waitForScan(t, m)

	m.publish([]catalog.Entry{
		{Path: "/m/a.jlm", Name: "a.jlm", Kind: catalog.KindContainer},
		{Path: "/m/b.jlm", Name: "b.jlm", Kind: catalog.KindContainer},
		{Path: "/m/c.jlm", Name: "c.jlm", Kind: catalog.KindContainer},
	})
	sh.Store.CatalogSel.Set(1) // b.jlm

	m.publish([]catalog.Entry{
		{Path: "/m/a.jlm", Name: "a.jlm", Kind: catalog.KindContainer},
		{Path: "/m/d.jlm", Name: "d.jlm", Kind: catalog.KindContainer},
		{Path: "/m/c.jlm", Name: "c.jlm", Kind: catalog.KindContainer},
	})

	if got := sh.Store.CatalogSel.Get(); got != -1 {
		e, _ := sh.Store.SelectedEntry()
		t.Fatalf("CatalogSel = %d (%s); b.jlm is gone and row 1 is now a different model",
			got, e.Name)
	}
	if _, ok := sh.Store.SelectedEntry(); ok {
		t.Error("SelectedEntry must report no selection")
	}
}

// waitForScan blocks until the scan that modelsFor kicked off has finished.
//
// Without it the scan's publish races the test's own catalog write. rescan
// marks itself running before it spawns, so this poll cannot miss it.
func waitForScan(t *testing.T, m *modelsScreen) {
	t.Helper()
	for range 1000 {
		m.mu.Lock()
		running := m.scanRunning
		m.mu.Unlock()
		if !running {
			return
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatal("the catalog scan did not finish")
}
