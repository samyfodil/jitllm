package convertjob

import (
	"bytes"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

func writeZeros(t *testing.T, path string, n int) {
	t.Helper()
	if err := os.WriteFile(path, make([]byte, n), 0o600); err != nil {
		t.Fatal(err)
	}
}

// A container goes to the model directory, not beside its source; with no
// model directory it stays beside.
func TestDestForMatchesTheCLIsDerivation(t *testing.T) {
	cases := []struct{ in, dir, want string }{
		{"/m/llama.gguf", "", "/m/llama.jlm"},
		{"/m/llama.safetensors", "", "/m/llama.jlm"},
		{"/m/qwen/", "", "/m/qwen.jlm"},
		{"/m/qwen", "", "/m/qwen.jlm"},
		{"  /m/a.gguf  ", "", "/m/a.jlm"},
		{"", "/models", ""},
		{"/home/u/.ollama/models/blobs/sha256:e6a6", "/models", "/models/sha256:e6a6.jlm"},
		{"/tmp/x/llama.gguf", "/models", "/models/llama.jlm"},
		{"/home/u/Downloads/qwen/", "/models", "/models/qwen.jlm"},
		{"/models/tiny.gguf", "/models", "/models/tiny.jlm"},
	}
	for _, c := range cases {
		// ToSlash: the join into dir uses the OS's separator.
		if got := filepath.ToSlash(DestFor(c.in, c.dir)); got != c.want {
			t.Errorf("DestFor(%q, %q) = %q, want %q", c.in, c.dir, got, c.want)
		}
	}
}

// The destination is derived at plan time, so the form's field may be left
// empty.
func TestPlanDerivesTheDestination(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "tinyllama.gguf")
	writeZeros(t, src, 16)

	got, err := Plan(Request{Src: "  " + src + "  "})
	if err != nil {
		t.Fatal(err)
	}
	if want := filepath.Join(dir, "tinyllama.jlm"); got.Dst != want {
		t.Fatalf("Dst = %q, want %q", got.Dst, want)
	}
	if got.Src != src {
		t.Fatalf("Src = %q, want it trimmed to %q", got.Src, src)
	}
}

func TestPlanRefusesAnEmptySource(t *testing.T) {
	if _, err := Plan(Request{}); err == nil {
		t.Fatal("an empty source must be refused with a sentence, not converted")
	}
}

func TestPlanRefusesAMissingSource(t *testing.T) {
	_, err := Plan(Request{Src: filepath.Join(t.TempDir(), "nope.gguf")})
	if err == nil {
		t.Fatal("a source that is not on disk must be refused before any work starts")
	}
}

// A conversion must not write over its own input: a GGUF is not recoverable
// from the container it produced.
func TestPlanRefusesWritingOverTheSource(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "m.gguf")
	writeZeros(t, src, 8)
	if _, err := Plan(Request{Src: src, Dst: src}); err == nil {
		t.Fatal("writing the container over its own source must be refused")
	}
}

func TestPlanRefusesWritingOverTheVisionTower(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "m.gguf")
	mm := filepath.Join(dir, "mmproj.gguf")
	writeZeros(t, src, 8)
	writeZeros(t, mm, 8)
	if _, err := Plan(Request{Src: src, MMProj: mm, Dst: mm}); err == nil {
		t.Fatal("writing the container over the mmproj must be refused")
	}
}

// An mmproj beside a HuggingFace model is refused by name rather than
// dropped, since such a model carries its tower in its own directory.
func TestPlanRefusesAnMMProjBesideSafetensors(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "model.safetensors")
	mm := filepath.Join(dir, "mmproj.gguf")
	writeZeros(t, src, 8)
	writeZeros(t, mm, 8)

	_, err := Plan(Request{Src: src, MMProj: mm})
	if err == nil {
		t.Fatal("a safetensors model plus an mmproj must be refused, not half-converted")
	}
}

func TestPlanRefusesAMissingVisionTower(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "m.gguf")
	writeZeros(t, src, 8)
	_, err := Plan(Request{Src: src, MMProj: filepath.Join(dir, "gone.gguf")})
	if err == nil {
		t.Fatal("a vision tower that is not on disk must be refused")
	}
}

// A directory's size is the sum of its shards, not the directory entry's.
func TestSourceBytesSumsTheShardsOfADirectory(t *testing.T) {
	dir := t.TempDir()
	writeZeros(t, filepath.Join(dir, "model-00001.safetensors"), 1000)
	writeZeros(t, filepath.Join(dir, "model-00002.safetensors"), 2000)
	writeZeros(t, filepath.Join(dir, "config.json"), 50)

	n, err := SourceBytes(dir)
	if err != nil {
		t.Fatal(err)
	}
	if n != 3000 {
		t.Fatalf("SourceBytes = %d, want 3000 (the shards, not the JSON beside them)", n)
	}
}

func TestSourceBytesIsTheFileSizeForOneFile(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "m.gguf")
	writeZeros(t, p, 4096)
	n, err := SourceBytes(p)
	if err != nil {
		t.Fatal(err)
	}
	if n != 4096 {
		t.Fatalf("SourceBytes = %d, want 4096", n)
	}
}

func TestSourceBytesRefusesADirectoryWithNoWeights(t *testing.T) {
	if _, err := SourceBytes(t.TempDir()); err == nil {
		t.Fatal("a directory holding no weight file must be refused")
	}
}

// The bar cannot reach 1: a container can be bigger than its GGUF, so a
// ratio above 1 can be a finished conversion.
func TestFractionClampsBelowOne(t *testing.T) {
	cases := []struct {
		dst, src int64
		want     float64
	}{
		{0, 100, 0},
		{50, 100, 0.5},
		{100, 100, 0.99},
		{135, 100, 0.99},
		{50, 0, 0},
		{-1, 100, 0},
	}
	for _, c := range cases {
		if got := Fraction(c.dst, c.src); got != c.want {
			t.Errorf("Fraction(%d, %d) = %v, want %v", c.dst, c.src, got, c.want)
		}
	}
}

// A conversion that fails must leave the file it would have replaced
// (jlm.Write writes a part file and renames it).
func TestAFailedConversionKeepsTheFileItWasReplacing(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "broken.gguf")
	dst := filepath.Join(dir, "broken.jlm")
	writeZeros(t, src, 64) // not a GGUF: the converter refuses it
	old := []byte("the container that was here")
	if err := os.WriteFile(dst, old, 0o600); err != nil {
		t.Fatal(err)
	}

	var last Update
	if _, err := Run(Request{Src: src, Dst: dst}, func(u Update) { last = u }); err == nil {
		t.Fatal("setup: a file of zeros converted")
	}
	got, err := os.ReadFile(dst)
	if err != nil {
		t.Fatalf("a failed conversion deleted %s: %v", filepath.Base(dst), err)
	}
	if !bytes.Equal(got, old) {
		t.Fatalf("a failed conversion changed %s", filepath.Base(dst))
	}
	if last.Stage != Failed || last.Err == nil || last.Label != "" {
		t.Fatalf("the last report was %+v, want Failed with its error and no label", last)
	}
}

// The bar must follow the file being written (PartPath(dst)), not the
// destination, which on a reconvert is the old container.
func TestTheBarFollowsTheFileBeingWritten(t *testing.T) {
	const total = 4 << 20
	dir := t.TempDir()
	dst := filepath.Join(dir, "m.jlm")
	writeZeros(t, dst, total) // the old container, as big as the source
	// The part file as jlm.Write makes it: presized to the end, a quarter
	// written, so a bar that read the size would fail.
	f, err := os.Create(PartPath(dst))
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Truncate(total); err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteAt(bytes.Repeat([]byte{1}, total/4), 0); err != nil {
		t.Fatal(err)
	}
	f.Close()
	// The bar counts allocated blocks, which only tells written from presized
	// on a filesystem that keeps the presized tail sparse.
	if fi, err := os.Stat(PartPath(dst)); err == nil && written(fi) >= total {
		t.Skipf("%s allocated the presized part file in full, so written bytes cannot be told from its size here", dir)
	}
	var mu sync.Mutex
	var frac float64
	stop, done := make(chan struct{}), make(chan struct{})
	go poll(func(u Update) {
		mu.Lock()
		frac = u.Fraction
		mu.Unlock()
	}, dst, total, stop, done)
	got := func() float64 {
		mu.Lock()
		defer mu.Unlock()
		return frac
	}
	deadline := time.Now().Add(3 * time.Second)
	for got() == 0 && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	close(stop)
	<-done

	if got := got(); got < 0.2 || got > 0.3 {
		t.Fatalf("progress %.2f, want about 0.25 -- the part file, not the old container", got)
	}
}

// A request that names no destination converts into the model directory it
// names, not beside the source: the form passes the configured folder.
func TestPlanPutsAnUnnamedDestinationInTheModelDir(t *testing.T) {
	src := filepath.Join(t.TempDir(), "m.gguf")
	if err := os.WriteFile(src, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	plan, err := Plan(Request{Src: src, Dir: dir})
	if err != nil {
		t.Fatal(err)
	}
	if want := DestFor(src, dir); plan.Dst != want || filepath.Dir(plan.Dst) != dir {
		t.Errorf("Dst = %s, want %s in %s", plan.Dst, want, dir)
	}
}
