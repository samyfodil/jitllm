package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/samyfodil/jitllm/tok/pretok"
)

// TestDumpRawIsReadableTestdata: -dump produces offline testdata, so
// pretok.ParsePreTokenizer must be able to read what it wrote.
func TestDumpRawIsReadableTestdata(t *testing.T) {
	const raw = `{"type":"Sequence","pretokenizers":[{"type":"Digits","individual_digits":true},` +
		`{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true}]}`
	dir := t.TempDir()
	if err := dumpRaw(dir, "smollm", []byte(raw)); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(filepath.Join(dir, "smollm.json"))
	if err != nil {
		t.Fatal(err)
	}
	// The object itself is verbatim -- a re-indent would make a diff against the
	// upstream file unreadable, which is the whole use of keeping it.
	if !strings.Contains(string(b), raw) {
		t.Fatalf("the dumped object was reformatted:\n%s", b)
	}
	ops, err := pretok.ParsePreTokenizer(strings.NewReader(string(b)))
	if err != nil {
		t.Fatalf("the dump cannot be read back by the parser it is testdata for: %v", err)
	}
	if len(ops) != 2 {
		t.Fatalf("read back %+v, want two stages", ops)
	}
	// A pre name is a bare file name, never a path.
	if err := dumpRaw(dir, "../escape", []byte(raw)); err == nil {
		t.Error("dumpRaw accepted a path separator in a pre name")
	}
}
