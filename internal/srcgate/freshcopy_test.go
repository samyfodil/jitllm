// Package srcgate holds gates over the source tree itself: properties a
// compiler does not check and a reviewer does not reliably see.
package srcgate

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"path/filepath"
	"strings"
	"testing"

	"github.com/samyfodil/jitllm/internal/testmodels"
)

// TestNoCopyOfAFreshSlice fails on a defensive copy of a slice its callee
// already built for the caller: append([]T(nil), f()...) or slices.Clone(f())
// where every function named f returns a slice it made itself. The copy costs
// an allocation and, worse, claims the callee hands out its own memory when it
// does not -- the next reader either trusts the claim or has to go and check.
//
// A copy of something the callee keeps (a field, a parameter, a slice of
// either) is the right call and is not flagged: the UI copies a Signal's slice
// before appending to it for exactly that reason. Resolution is by name and
// conservative -- a name with any non-fresh definition is never flagged -- so
// the gate can miss a copy but does not flag a needed one.
func TestNoCopyOfAFreshSlice(t *testing.T) {
	testmodels.SourceTree(t)
	root := filepath.Join("..", "..")
	files, n := parseTree(t, root)
	if n < 100 {
		t.Fatalf("walked %d non-test Go files under %s: the tree was not found, so this gate proved nothing", n, root)
	}
	for _, f := range freshCopies(files) {
		t.Errorf("%s: copies the result of %s(), which returns a slice it built for the caller -- drop the copy", f.pos, f.callee)
	}
}

// The gate is run against a violation and a needed copy, so a green line
// means it looked.
func TestTheFreshCopyGateDiscriminates(t *testing.T) {
	files, _ := parseTree(t, "testdata")
	got := map[string]bool{}
	for _, f := range freshCopies(files) {
		got[f.callee] = true
	}
	for _, want := range []string{"runs", "built", "viaCall", "cloned"} {
		if !got[want] {
			t.Errorf("a copy of %s() was not flagged: it returns a fresh slice", want)
		}
	}
	for _, not := range []string{"Get", "param", "mixed", "sliced"} {
		if got[not] {
			t.Errorf("a copy of %s() was flagged: it returns memory the callee keeps, so the copy is needed", not)
		}
	}
}

type finding struct {
	pos    token.Position
	callee string
}

type tree struct {
	fset  *token.FileSet
	files []*ast.File
}

// parseTree parses every non-test Go file under root, skipping the
// directories no gate in this tree reads.
func parseTree(t *testing.T, root string) (tree, int) {
	t.Helper()
	tr := tree{fset: token.NewFileSet()}
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
			case "gen", "vendor", "node_modules":
				return filepath.SkipDir
			case "testdata":
				if p != root {
					return filepath.SkipDir
				}
			}
			return nil
		}
		if !strings.HasSuffix(p, ".go") || strings.HasSuffix(p, "_test.go") {
			return nil
		}
		f, err := parser.ParseFile(tr.fset, p, nil, 0)
		if err != nil {
			return err
		}
		tr.files = append(tr.files, f)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return tr, len(tr.files)
}

// freshCopies finds every copy of a call's result whose callee, under every
// definition of its name, returns a slice it built.
func freshCopies(tr tree) []finding {
	defs := map[string][]*ast.FuncDecl{}
	for _, f := range tr.files {
		for _, d := range f.Decls {
			if fd, ok := d.(*ast.FuncDecl); ok && fd.Body != nil {
				defs[fd.Name.Name] = append(defs[fd.Name.Name], fd)
			}
		}
	}
	fr := freshness{defs: defs, memo: map[string]bool{}, busy: map[string]bool{}}
	var out []finding
	for _, f := range tr.files {
		ast.Inspect(f, func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			src := copiedCall(call)
			if src == nil {
				return true
			}
			if name := calleeName(src); name != "" && fr.name(name) {
				out = append(out, finding{tr.fset.Position(call.Pos()), name})
			}
			return true
		})
	}
	return out
}

// copiedCall returns the call whose result n copies, or nil:
// append([]T(nil), f()...), append([]T{}, f()...) and slices.Clone(f()).
func copiedCall(n *ast.CallExpr) *ast.CallExpr {
	switch fn := n.Fun.(type) {
	case *ast.Ident:
		if fn.Name != "append" || len(n.Args) != 2 || !n.Ellipsis.IsValid() || !emptySlice(n.Args[0]) {
			return nil
		}
		c, _ := n.Args[1].(*ast.CallExpr)
		return c
	case *ast.SelectorExpr:
		if pkg, ok := fn.X.(*ast.Ident); ok && pkg.Name == "slices" && fn.Sel.Name == "Clone" && len(n.Args) == 1 {
			c, _ := n.Args[0].(*ast.CallExpr)
			return c
		}
	}
	return nil
}

// emptySlice is []T(nil) or []T{}.
func emptySlice(e ast.Expr) bool {
	switch x := e.(type) {
	case *ast.CallExpr:
		_, arr := x.Fun.(*ast.ArrayType)
		id, isNil := unparen(x.Args...).(*ast.Ident)
		return arr && isNil && id.Name == "nil"
	case *ast.CompositeLit:
		_, arr := x.Type.(*ast.ArrayType)
		return arr && len(x.Elts) == 0
	}
	return false
}

func unparen(args ...ast.Expr) ast.Expr {
	if len(args) != 1 {
		return nil
	}
	return ast.Unparen(args[0])
}

func calleeName(c *ast.CallExpr) string {
	switch fn := c.Fun.(type) {
	case *ast.Ident:
		return fn.Name
	case *ast.SelectorExpr:
		return fn.Sel.Name
	}
	return ""
}

// freshness decides whether a function name returns a slice its body built.
type freshness struct {
	defs map[string][]*ast.FuncDecl
	memo map[string]bool
	busy map[string]bool
}

// name is fresh when it has at least one definition and every one of them has
// a single result that every return builds. A recursive name is not fresh.
func (fr *freshness) name(n string) bool {
	if v, ok := fr.memo[n]; ok {
		return v
	}
	ds := fr.defs[n]
	if len(ds) == 0 || fr.busy[n] {
		return false
	}
	fr.busy[n] = true
	v := true
	for _, d := range ds {
		if !fr.decl(d) {
			v = false
			break
		}
	}
	delete(fr.busy, n)
	fr.memo[n] = v
	return v
}

func (fr *freshness) decl(d *ast.FuncDecl) bool {
	res := d.Type.Results
	if res == nil || res.NumFields() != 1 {
		return false
	}
	if _, ok := res.List[0].Type.(*ast.ArrayType); !ok {
		return false
	}
	locals := fr.locals(d.Body)
	ok, returns := true, 0
	ast.Inspect(d.Body, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.FuncLit:
			return false
		case *ast.ReturnStmt:
			returns++
			if len(x.Results) != 1 || !fr.expr(x.Results[0], locals) {
				ok = false
			}
		}
		return ok
	})
	return ok && returns > 0
}

// locals are the body's variables every assignment to which is a built
// slice or an append to the variable itself.
func (fr *freshness) locals(body *ast.BlockStmt) map[string]bool {
	declared, spoiled := map[string]bool{}, map[string]bool{}
	assign := func(name string, rhs ast.Expr) {
		if rhs == nil {
			return
		}
		if call, ok := rhs.(*ast.CallExpr); ok {
			if id, ok := call.Fun.(*ast.Ident); ok && id.Name == "append" && len(call.Args) > 0 {
				if self, ok := call.Args[0].(*ast.Ident); ok && self.Name == name {
					return
				}
			}
		}
		if !fr.expr(rhs, nil) {
			spoiled[name] = true
		}
	}
	ast.Inspect(body, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.FuncLit:
			return false
		case *ast.ValueSpec:
			for i, id := range x.Names {
				declared[id.Name] = true
				if i < len(x.Values) {
					assign(id.Name, x.Values[i])
				} else if len(x.Values) > 0 {
					spoiled[id.Name] = true
				}
			}
		case *ast.AssignStmt:
			for i, l := range x.Lhs {
				id, ok := l.(*ast.Ident)
				if !ok {
					continue
				}
				if x.Tok == token.DEFINE {
					declared[id.Name] = true
				}
				if len(x.Lhs) != len(x.Rhs) {
					spoiled[id.Name] = true
					continue
				}
				assign(id.Name, x.Rhs[i])
			}
		case *ast.RangeStmt:
			for _, e := range []ast.Expr{x.Key, x.Value} {
				if id, ok := e.(*ast.Ident); ok {
					spoiled[id.Name] = true
				}
			}
		}
		return true
	})
	out := map[string]bool{}
	for n := range declared {
		if !spoiled[n] {
			out[n] = true
		}
	}
	return out
}

// expr is a slice built here: nil, a literal, make, an append onto a built
// slice, a fresh local, or a call to a fresh name.
func (fr *freshness) expr(e ast.Expr, locals map[string]bool) bool {
	switch x := ast.Unparen(e).(type) {
	case *ast.Ident:
		return x.Name == "nil" || locals[x.Name]
	case *ast.CompositeLit:
		_, arr := x.Type.(*ast.ArrayType)
		return arr
	case *ast.CallExpr:
		if id, ok := x.Fun.(*ast.Ident); ok {
			switch id.Name {
			case "make":
				return true
			case "append":
				return len(x.Args) > 0 && fr.expr(x.Args[0], locals)
			}
		}
		if _, conv := x.Fun.(*ast.ArrayType); conv {
			return len(x.Args) == 1 && fr.expr(x.Args[0], locals)
		}
		n := calleeName(x)
		return n != "" && fr.name(n)
	}
	return false
}
