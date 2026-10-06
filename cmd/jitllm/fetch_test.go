package main

import (
	"os"
	"path/filepath"
	"testing"
)

// TestConvertArgsSplitsThePositionals.
//
// The case that matters is a repository reference, which has no ".gguf"
// suffix and must not be taken for the output path.
func TestConvertArgsSplitsThePositionals(t *testing.T) {
	for _, c := range []struct {
		argv             []string
		src, mmproj, dst string
	}{
		// What shipped, unchanged.
		{[]string{"m.gguf"}, "m.gguf", "", ""},
		{[]string{"m.gguf", "out.jlm"}, "m.gguf", "", "out.jlm"},
		{[]string{"m.gguf", "mmproj.gguf"}, "m.gguf", "mmproj.gguf", ""},
		{[]string{"m.gguf", "mmproj.gguf", "out.jlm"}, "m.gguf", "mmproj.gguf", "out.jlm"},
		// A remote model, a local output.
		{[]string{"hf://o/r/m.gguf"}, "hf://o/r/m.gguf", "", ""},
		{[]string{"hf://o/r/m.gguf", "out.jlm"}, "hf://o/r/m.gguf", "", "out.jlm"},
		// A repository reference has no suffix and is still not the output.
		{[]string{"hf://o/r", "out.jlm"}, "hf://o/r", "", "out.jlm"},
		{[]string{"hf://o/r", "hf://o/v"}, "hf://o/r", "hf://o/v", ""},
		{[]string{"hf://o/r", "hf://o/v", "out.jlm"}, "hf://o/r", "hf://o/v", "out.jlm"},
		// Mixed, which is the VLM case: a text model here, a tower on the Hub.
		{[]string{"m.gguf", "https://huggingface.co/o/r/resolve/main/mmproj.gguf", "out.jlm"},
			"m.gguf", "https://huggingface.co/o/r/resolve/main/mmproj.gguf", "out.jlm"},
	} {
		src, mmproj, dst, err := convertArgs(c.argv)
		if err != nil {
			t.Errorf("convertArgs(%q): %v", c.argv, err)
			continue
		}
		if src != c.src || mmproj != c.mmproj || dst != c.dst {
			t.Errorf("convertArgs(%q) = (%q, %q, %q), want (%q, %q, %q)",
				c.argv, src, mmproj, dst, c.src, c.mmproj, c.dst)
		}
	}
	for _, argv := range [][]string{
		{},                             // nothing to convert
		{"m.gguf", "out.jlm", "extra"}, // a fourth thing
		{"hf://o", "out.jlm"},          // a malformed reference is not a filename
		{"https://example.com/m.gguf", "out.jlm"}, // and neither is a URL to somewhere else
	} {
		if _, _, _, err := convertArgs(argv); err == nil {
			t.Errorf("convertArgs(%q) was accepted", argv)
		}
	}
}

// TestModelDirFollowsTheModelsAlreadyHere.
//
// "./models" is right on a fresh checkout, but where ./models holds symlinks
// onto a model disk the download must follow them there.
func TestModelDirFollowsTheModelsAlreadyHere(t *testing.T) {
	// A fresh checkout: models/ holds real files, or does not exist.
	t.Run("no links", func(t *testing.T) {
		root := t.TempDir()
		dir := filepath.Join(root, "models")
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "a.gguf"), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
		if got := modelDirNear(dir); got != dir {
			t.Errorf("got %q, want the directory itself %q", got, dir)
		}
		missing := filepath.Join(root, "nothing-here")
		if got := modelDirNear(missing); got != missing {
			t.Errorf("got %q for a directory that does not exist, want %q", got, missing)
		}
	})

	// A checkout where models/ is a directory of symlinks onto the model disk.
	t.Run("a farm of links", func(t *testing.T) {
		root := t.TempDir()
		disk := filepath.Join(root, "extend", "jpt-models")
		other := filepath.Join(root, "elsewhere")
		for _, d := range []string{disk, other} {
			if err := os.MkdirAll(d, 0o755); err != nil {
				t.Fatal(err)
			}
		}
		dir := filepath.Join(root, "models")
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		link := func(target, name string) {
			t.Helper()
			if err := os.WriteFile(target, []byte("x"), 0o644); err != nil {
				t.Fatal(err)
			}
			if err := os.Symlink(target, filepath.Join(dir, name)); err != nil {
				t.Fatal(err)
			}
		}
		link(filepath.Join(disk, "a.gguf"), "a.gguf")
		link(filepath.Join(disk, "b.gguf"), "b.gguf")
		link(filepath.Join(other, "c.gguf"), "c.gguf") // a minority elsewhere
		// A dangling link votes for nothing rather than for its own directory.
		if err := os.Symlink(filepath.Join(disk, "gone.gguf"), filepath.Join(dir, "gone.gguf")); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "real.txt"), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
		want, err := filepath.EvalSymlinks(disk)
		if err != nil {
			t.Fatal(err)
		}
		if got := modelDirNear(dir); got != want {
			t.Errorf("got %q, want the disk the models are on, %q", got, want)
		}
	})

	// models/ is itself one link: no vote needed.
	t.Run("the directory is the link", func(t *testing.T) {
		root := t.TempDir()
		disk := filepath.Join(root, "jpt-models")
		if err := os.MkdirAll(disk, 0o755); err != nil {
			t.Fatal(err)
		}
		dir := filepath.Join(root, "models")
		if err := os.Symlink(disk, dir); err != nil {
			t.Fatal(err)
		}
		want, err := filepath.EvalSymlinks(disk)
		if err != nil {
			t.Fatal(err)
		}
		if got := modelDirNear(dir); got != want {
			t.Errorf("got %q, want %q", got, want)
		}
	})
}

// TestTheTokenComesFromTheEnvironmentAndTheLoginFile. A gated repo answers 401
// either way, so "the token was not found" and "the token was refused" are the
// same message unless this is checked directly.
func TestTheTokenComesFromTheEnvironmentAndTheLoginFile(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HF_HOME", home)
	for _, k := range []string{"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"} {
		t.Setenv(k, "")
	}
	if got := hfToken(); got != "" {
		t.Errorf("a token appeared from nowhere: %q", got)
	}
	// huggingface-cli login writes this and exports nothing.
	if err := os.WriteFile(filepath.Join(home, "token"), []byte("from-the-file\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if got := hfToken(); got != "from-the-file" {
		t.Errorf("hfToken() = %q, want the login file's token (newline trimmed)", got)
	}
	t.Setenv("HUGGING_FACE_HUB_TOKEN", "from-the-env")
	if got := hfToken(); got != "from-the-env" {
		t.Errorf("hfToken() = %q, want the environment to win over the file", got)
	}
	t.Setenv("HF_TOKEN", "first")
	if got := hfToken(); got != "first" {
		t.Errorf("hfToken() = %q, want HF_TOKEN to be read first", got)
	}
}

// TestLocaliseLeavesALocalPathAlone: the whole remote path must be invisible to
// a caller converting a file that is already here, including for a path that
// does not exist -- the converter's own error about it is better than this
// package's.
func TestLocaliseLeavesALocalPathAlone(t *testing.T) {
	// A client that would fail loudly if it were ever used.
	cl := hfClient()
	cl.Endpoint = "http://127.0.0.1:1"
	for _, p := range []string{"", "models/gemma-2b.gguf", "/nope/x.gguf", "./out.jlm"} {
		got, err := localise(cl, p, "")
		if err != nil {
			t.Errorf("localise(%q): %v", p, err)
		}
		if got != p {
			t.Errorf("localise(%q) = %q; a local path passes through untouched", p, got)
		}
	}
	if _, err := localise(cl, "https://example.com/m.gguf", ""); err == nil {
		t.Error("a URL to somewhere else was treated as a local path")
	}
}
