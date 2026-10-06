package oracle_test

import (
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/samyfodil/jitllm/internal/testmodels"
)

// TestNoProductionCodeImportsTheOracle walks every non-test Go file in the
// module and fails if one imports this package.
//
// It makes "no Go compute on the inference path" a property rather than a
// habit: a missing kernel cannot silently fall back to the oracle.
func TestNoProductionCodeImportsTheOracle(t *testing.T) {
	testmodels.SourceTree(t)
	const me = "github.com/samyfodil/jitllm/internal/oracle"
	root := filepath.Join("..", "..")
	fset := token.NewFileSet()
	checked := 0
	err := filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			// A dot-directory (.git, a local tool's worktrees) is not this
			// tree's source.
			if p != root && strings.HasPrefix(d.Name(), ".") {
				return filepath.SkipDir
			}
			switch d.Name() {
			case "gen", "vendor", "node_modules", "testdata":
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(p, ".go") || strings.HasSuffix(p, "_test.go") {
			return nil
		}
		if strings.Contains(filepath.ToSlash(p), "internal/oracle/") {
			return nil
		}
		f, err := parser.ParseFile(fset, p, nil, parser.ImportsOnly|parser.ParseComments)
		if err != nil {
			return err
		}
		// A file built only under the jitllmtest tag is test code: the Go
		// quantizer arm of a measurement (cpu.quantGoLoop) takes its arithmetic
		// from here, as the rule says it must.
		if testOnly(f) {
			return nil
		}
		checked++
		for _, im := range f.Imports {
			if strings.Trim(im.Path.Value, `"`) == me {
				t.Errorf("%s imports the oracle: production code must run generated kernels", p)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if checked < 100 {
		t.Fatalf("walked only %d files from %s -- the walk is not seeing the module", checked, root)
	}
	if _, err := os.Stat(filepath.Join(root, "go.mod")); err != nil {
		t.Fatalf("%s is not the module root: %v", root, err)
	}
	t.Logf("%d production files, none importing the oracle", checked)
}
