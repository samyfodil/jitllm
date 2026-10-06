package main

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/samyfodil/jitllm/convert/hf"
)

// The command half of package hf: where a download lands, and which token pays
// for it. Both are environment questions, so both live here -- the Scope
// section's rule is that a library takes options and cmd/ reads the names.

// hfClient is the configured fetcher. Progress goes to stderr beside every other
// line convert prints, so a redirected stdout still shows the download.
func hfClient() *hf.Client {
	return &hf.Client{
		Endpoint: strings.TrimSpace(os.Getenv("HF_ENDPOINT")),
		Token:    hfToken(),
		Progress: os.Stderr,
	}
}

// hfToken is hf.FindToken over this process's environment.
//
// There is deliberately no -token flag: a flag value is in the process title,
// which every other user of the machine can read.
func hfToken() string { return hf.FindToken(os.Getenv, os.UserHomeDir) }

// localise turns a HuggingFace reference into a local path, fetching it if it is
// not here yet. A local path passes straight through, untouched and unchecked --
// the converter opens it a moment later and says a better thing about it than
// this could.
func localise(cl *hf.Client, arg, dir string) (string, error) {
	r, remote, err := hf.Parse(arg)
	if err != nil || !remote {
		return arg, err
	}
	if dir == "" {
		dir = defaultModelDir()
	}
	return cl.Fetch(r, dir)
}

// defaultModelDir is where a download lands when -o did not say.
func defaultModelDir() string {
	if d := strings.TrimSpace(os.Getenv("JITLLM_MODELS")); d != "" {
		return d
	}
	return modelDirNear("models")
}

// modelDirNear picks the directory a download should land in, given the
// conventional one.
//
// A download lands beside the models already here, read off the disk rather
// than compiled in: `models/` may be a directory of symlinks into a model disk,
// and following them answers with the disk the models are on. With no links it
// answers ./models, which is what internal/testmodels/fetch.sh uses.
//
// JITLLM_MODELS overrides it, and -o DIR overrides that.
func modelDirNear(dir string) string {
	fi, err := os.Lstat(dir)
	if err != nil {
		return dir // nothing there yet: it gets created where it was named
	}
	// The whole directory being a link is the simple case and needs no vote.
	if fi.Mode()&os.ModeSymlink != 0 {
		if t, err := filepath.EvalSymlinks(dir); err == nil {
			if st, err := os.Stat(t); err == nil && st.IsDir() {
				return t
			}
		}
		return dir
	}
	if !fi.IsDir() {
		return dir // a FILE called models: not ours to interpret
	}
	ents, err := os.ReadDir(dir)
	if err != nil {
		return dir
	}
	votes := map[string]int{}
	for _, e := range ents {
		if e.Type()&os.ModeSymlink == 0 {
			continue
		}
		t, err := filepath.EvalSymlinks(filepath.Join(dir, e.Name()))
		if err != nil {
			continue // a dangling link votes for nothing
		}
		votes[filepath.Dir(t)]++
	}
	// Sorted, so a tie resolves to the same directory every run: a default that
	// moves between invocations is a download that happens twice.
	keys := make([]string, 0, len(votes))
	for k := range votes {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	best, n := dir, 0
	for _, k := range keys {
		if votes[k] > n {
			best, n = k, votes[k]
		}
	}
	return best
}

// convertArgs splits convert's positionals into the three things they can be.
//
// A remote reference counts as a .gguf: the tower is recognised by its suffix
// (`convert model.gguf mmproj.gguf out.jlm`), and `hf://o/r` has none, so it
// would otherwise be taken for the output path.
func convertArgs(args []string) (src, mmproj, dst string, err error) {
	if len(args) == 0 {
		return "", "", "", fmt.Errorf("usage: jitllm convert [-o DIR] <model.gguf|hf://OWNER/REPO/FILE.gguf> " +
			"[mmproj.gguf] [out.jlm]")
	}
	src = args[0]
	// The model itself is parsed here rather than left to the fetch, so a
	// mistyped reference is refused before anything is opened or created.
	if _, _, perr := hf.Parse(src); perr != nil {
		return "", "", "", perr
	}
	for _, a := range args[1:] {
		_, remote, perr := hf.Parse(a)
		if perr != nil {
			return "", "", "", perr
		}
		switch {
		case mmproj == "" && dst == "" && (remote || strings.HasSuffix(a, ".gguf")):
			mmproj = a
		case dst == "":
			dst = a
		default:
			return "", "", "", fmt.Errorf("jitllm convert: %q is a fourth argument; "+
				"the form is <model> [mmproj] [out.jlm]", a)
		}
	}
	return src, mmproj, dst, nil
}
