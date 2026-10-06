package srcgate

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/samyfodil/jitllm/internal/modelsdoc"
	"github.com/samyfodil/jitllm/internal/testmodels"
)

// TestModelsDocIsGenerated holds docs/models.md to the converter's tables: it
// renders the page in memory from convert.Architectures, convert.HFClasses,
// convert.Projectors, quant's lists and the catalogue, and fails when the file
// on disk is anything else -- an architecture added without regenerating, or a
// line edited by hand.
func TestModelsDocIsGenerated(t *testing.T) {
	testmodels.SourceTree(t)
	want, err := modelsdoc.Render()
	if err != nil {
		t.Fatal(err)
	}
	have, err := os.ReadFile(filepath.Join("..", "..", filepath.FromSlash(modelsdoc.Path)))
	if err != nil {
		t.Fatalf("%v; run: %s", err, modelsdoc.Command)
	}
	if bytes.Equal(have, want) {
		return
	}
	hl, wl := strings.Split(string(have), "\n"), strings.Split(string(want), "\n")
	for i := 0; i < len(hl) || i < len(wl); i++ {
		var h, w string
		if i < len(hl) {
			h = hl[i]
		}
		if i < len(wl) {
			w = wl[i]
		}
		if h != w {
			t.Fatalf("%s is not what the code generates; run: %s\nfirst difference, line %d:\n  file: %q\n  code: %q",
				modelsdoc.Path, modelsdoc.Command, i+1, h, w)
		}
	}
}
