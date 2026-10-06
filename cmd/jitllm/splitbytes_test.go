package main

import (
	"flag"
	"os"
	"path/filepath"
	"testing"
)

// TestSourceBytesCountsEverySplitPart: the convert report divides the container
// by the SOURCE, and a split GGUF's source is all of its parts. Counting part 1
// alone overstated the container's growth over its source.
func TestSourceBytesCountsEverySplitPart(t *testing.T) {
	dir := t.TempDir()
	parts := []struct {
		name string
		n    int
	}{{"m-00001-of-00003.gguf", 700}, {"m-00002-of-00003.gguf", 500}, {"m-00003-of-00003.gguf", 300}}
	for _, p := range parts {
		if err := os.WriteFile(filepath.Join(dir, p.name), make([]byte, p.n), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	got, err := sourceBytes(filepath.Join(dir, parts[0].name))
	if err != nil {
		t.Fatal(err)
	}
	if got != 1500 {
		t.Errorf("a three-part GGUF counts %d bytes, want 1500", got)
	}
	// A lone file is its own size, and a missing part is an error rather than
	// a smaller number.
	one := filepath.Join(dir, "solo.gguf")
	if err := os.WriteFile(one, make([]byte, 42), 0o644); err != nil {
		t.Fatal(err)
	}
	if got, err := sourceBytes(one); err != nil || got != 42 {
		t.Errorf("a single GGUF counts %d (%v), want 42", got, err)
	}
	if err := os.Remove(filepath.Join(dir, parts[2].name)); err != nil {
		t.Fatal(err)
	}
	if _, err := sourceBytes(filepath.Join(dir, parts[0].name)); err == nil {
		t.Error("a split GGUF with a missing part counted anyway")
	}
}

// TestByteFlagsTakeUnits: the size flags read 3G as three GiB, keep a plain
// byte count, keep 0 as "work it out", and keep their default when unset.
func TestByteFlagsTakeUnits(t *testing.T) {
	fs := flag.NewFlagSet("t", flag.ContinueOnError)
	mm := bytesFlag(fs, "maxmem", 0, "")
	kv := bytesFlag(fs, "kv-cache-max", 8<<30, "")
	vr := bytesFlag(fs, "vram", 0, "")
	if err := fs.Parse([]string{"-maxmem", "3G", "-vram", "800000000"}); err != nil {
		t.Fatal(err)
	}
	if *mm != 3<<30 || *vr != 800000000 || *kv != 8<<30 {
		t.Fatalf("maxmem %d, vram %d, kv-cache-max %d; want %d, 800000000, %d", *mm, *vr, *kv, uint64(3<<30), uint64(8<<30))
	}
	if err := fs.Parse([]string{"-maxmem", "0"}); err != nil || *mm != 0 {
		t.Fatalf("-maxmem 0 read as %d (%v), want 0", *mm, err)
	}
}
