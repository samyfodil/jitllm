// Package testmodels is where a test finds a model file.
//
// One variable, JITLLM_MODELS, says where. Unset, the directory is models/ at
// the repository root, found by walking up to the go.mod that declares this
// module, so it is the same answer from every package, including the nested
// ui/ and server/ modules.
//
// The per-test variables (JITLLM_MODEL, JITLLM_DEV_MODEL, ...) say which model;
// this one says where. docs/testing.md has the table.
//
// A model a test needs and does not find is reported through Missing, which
// fails the test (RULE 11) unless JITLLM_MODEL_FREE=1 says this is CI's
// model-free run, where it skips with the same message.
//
// Tests only: TestNoProductionCodeImportsTestModels refuses a production import.
package testmodels

import (
	"bufio"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Shipped names the variable a cross-compiled test binary runs under on
// another host, away from the tree it was built from.
const Shipped = "JITLLM_SHIPPED_BINARY"

// SourceTree skips t, loudly, in a shipped test binary: a gate that reads the
// module's source has none beside it there, and the build tree runs it. A
// binary with no go.mod declaring this module above its working directory is
// shipped whether or not Shipped was set: a gate reading the source there
// would fail on the absence of the tree, not on anything the tree holds.
func SourceTree(t testing.TB) {
	t.Helper()
	if os.Getenv(Shipped) != "" || root() == "" {
		t.Skip("LOUD SKIP -- NOT A PASS: this reads the module's source, and this is a shipped test " +
			"binary running away from the tree it was built from. That tree runs this check.")
	}
}

// Env names the model directory.
const Env = "JITLLM_MODELS"

// FreeEnv is the switch for CI's model-free run. Only
// .github/scripts/test-model-free.sh sets it; a run on a box that is meant to
// hold the models leaves it unset, so an absent model fails there.
const FreeEnv = "JITLLM_MODEL_FREE"

// Free reports whether this is the model-free run (FreeEnv is "1").
func Free() bool { return os.Getenv(FreeEnv) == "1" }

// Missing reports a model, fixture or reference a test needs and does not
// find. It FAILS the test: a missing artefact is a task (RULE 11), and a gate
// that skipped proved nothing. In the model-free run (Free) it skips with the
// same message instead, and that run's job summary lists every skip with it.
//
// A GPU that is not there is not a missing artefact: a device arm skips on it
// whatever the switch says.
func Missing(t testing.TB, format string, args ...any) {
	t.Helper()
	if Free() {
		t.Skipf(format, args...)
	}
	t.Fatalf(format, args...)
}

const module = "github.com/jitllm/jitllm"

// Dir is the model directory: $JITLLM_MODELS, or models/ at the repository
// root. With neither a variable nor a root in reach it is "models", which a
// skip then names -- a path that does not exist is the loud answer.
func Dir() string {
	if d := strings.TrimSpace(os.Getenv(Env)); d != "" {
		return d
	}
	if r := root(); r != "" {
		return filepath.Join(r, "models")
	}
	return "models"
}

// Path is name inside Dir. An absolute name is returned unchanged.
func Path(name string) string {
	if filepath.IsAbs(name) {
		return name
	}
	p := filepath.Join(Dir(), name)
	if _, err := os.Stat(p); err == nil {
		return p
	}
	// Hugging Face names the same file q4_k_m in one repo and Q4_K_M in
	// another, and a box keeps whichever it downloaded; a gate that misses on
	// the case skips and proves nothing.
	if es, err := os.ReadDir(Dir()); err == nil {
		for _, e := range es {
			if strings.EqualFold(e.Name(), name) {
				return filepath.Join(Dir(), e.Name())
			}
		}
	}
	return p
}

// Glob matches pattern inside Dir.
func Glob(pattern string) []string {
	m, _ := filepath.Glob(filepath.Join(Dir(), pattern))
	return m
}

// Resolve reads an override variable's value: a bare name is a file in Dir, and
// anything with a directory in it is a path the caller wrote and is kept as is.
func Resolve(v string) string {
	if v == "" || strings.ContainsRune(v, '/') || strings.ContainsRune(v, filepath.Separator) {
		return v
	}
	return Path(v)
}

// root walks up from the working directory to the go.mod declaring module --
// not the first go.mod, because ui/ and server/ are modules of their own.
func root() string {
	d, err := os.Getwd()
	if err != nil {
		return ""
	}
	for {
		if declares(filepath.Join(d, "go.mod")) {
			return d
		}
		up := filepath.Dir(d)
		if up == d {
			return ""
		}
		d = up
	}
}

func declares(gomod string) bool {
	f, err := os.Open(gomod)
	if err != nil {
		return false
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	for s.Scan() {
		if w := strings.Fields(s.Text()); len(w) == 2 && w[0] == "module" {
			return w[1] == module
		}
	}
	return false
}
