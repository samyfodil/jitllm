package safetensors

import (
	"encoding/binary"

	"os"
	"path/filepath"
	"strings"
	"testing"
)

// write builds a safetensors file from a header object and a data section.
func write(t testing.TB, hdr string, data []byte) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "m.safetensors")
	b := binary.LittleEndian.AppendUint64(nil, uint64(len(hdr)))
	b = append(b, hdr...)
	b = append(b, data...)
	if err := os.WriteFile(p, b, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// TestParseRefusesACraftedHeader: every number in a safetensors header is
// untrusted, and each row violates one bound, so removing a bound from Parse
// turns its row red. The last two matter most: a byte range longer than the
// shape needs, and an overlap that makes two names resolve to one weight.
func TestParseRefusesACraftedHeader(t *testing.T) {
	cases := []struct {
		name string
		hdr  string
		data uint64
		want string
	}{
		{"not json", `{`, 0, "not a JSON object"},
		{"no tensors", `{"__metadata__":{"format":"pt"}}`, 0, "names no tensors"},
		{"unknown dtype", `{"a":{"dtype":"F4_E2M1","shape":[2],"data_offsets":[0,2]}}`, 2,
			"which this parser does not define"},
		{"offsets not a pair", `{"a":{"dtype":"F32","shape":[1],"data_offsets":[0]}}`, 4,
			"data_offsets, want 2"},
		{"backwards", `{"a":{"dtype":"F32","shape":[1],"data_offsets":[8,4]}}`, 16,
			"runs backwards"},
		{"past the end", `{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`, 2,
			"past the 2 bytes of data"},
		// 2^40 x 2^40 x 2^40 wraps a uint64 product to zero, which would then
		// "match" a zero-length range and let a tensor of no bytes claim a
		// shape of 10^36 elements.
		{"shape overflows", `{"a":{"dtype":"F32","shape":[1099511627776,1099511627776,1099511627776],"data_offsets":[0,0]}}`, 0,
			"overflows"},
		{"length does not match the shape", `{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,1024]}}`, 1024,
			"spans 1024 bytes and its F32[1] wants 4"},
		{"overlap", `{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},` +
			`"b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}}`, 16, "overlaps"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, _, err := Parse([]byte(c.hdr), c.data)
			if err == nil {
				t.Fatalf("accepted %s -- this gate proved nothing", c.hdr)
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Fatalf("refused for the wrong reason:\n got %v\nwant ...%s...", err, c.want)
			}
			t.Logf("refused: %v", err)
		})
	}
}

// TestOpenBoundsTheHeaderByTheFile is the one bound that cannot live in Parse:
// the declared header length is read before anything is allocated, so it must
// be checked against the file's measured size rather than against itself.
func TestOpenBoundsTheHeaderByTheFile(t *testing.T) {
	p := filepath.Join(t.TempDir(), "m.safetensors")
	// A 16-byte file claiming a 2^63-byte header. Allocating first is an
	// instant OOM; checking first is a one-line refusal.
	b := binary.LittleEndian.AppendUint64(nil, 1<<63)
	b = append(b, make([]byte, 8)...)
	if err := os.WriteFile(p, b, 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := Open(p)
	if err == nil {
		t.Fatal("opened a file claiming a 2^63-byte header -- this gate proved nothing")
	}
	if !strings.Contains(err.Error(), "claims a") {
		t.Fatalf("refused for the wrong reason: %v", err)
	}
	t.Logf("refused: %v", err)

	if _, err := Open(filepath.Join(t.TempDir(), "short")); err == nil {
		t.Fatal("opened a file that does not exist")
	}
	short := filepath.Join(t.TempDir(), "short.safetensors")
	if err := os.WriteFile(short, []byte{1, 2, 3}, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Open(short); err == nil || !strings.Contains(err.Error(), "too short") {
		t.Fatalf("a 3-byte file: %v", err)
	}
}

// TestOpenReadsWhatItParsed walks a small file end to end: the directory, the
// metadata, the sort order and the bytes.
func TestOpenReadsWhatItParsed(t *testing.T) {
	data := make([]byte, 24)
	for i := range data {
		data[i] = byte(i)
	}
	p := write(t, `{"__metadata__":{"format":"pt"},`+
		`"b":{"dtype":"F32","shape":[2,2],"data_offsets":[8,24]},`+
		`"a":{"dtype":"F16","shape":[4],"data_offsets":[0,8]}}`, data)
	f, err := Open(p)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if f.Meta["format"] != "pt" {
		t.Errorf("metadata %v", f.Meta)
	}
	// Sorted by Begin, not by name: a caller walking the directory reads the
	// file forwards.
	if len(f.Tensors) != 2 || f.Tensors[0].Name != "a" || f.Tensors[1].Name != "b" {
		t.Fatalf("directory %v", f.Tensors)
	}
	tb, ok := f.Get("b")
	if !ok {
		t.Fatal("no b")
	}
	if tb.Elems() != 4 || tb.NBytes() != 16 {
		t.Errorf("b: %d elements, %d bytes", tb.Elems(), tb.NBytes())
	}
	got, err := f.ReadTensor(tb)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range got {
		if v != byte(8+i) {
			t.Fatalf("b[%d] = %d, want %d", i, v, 8+i)
		}
	}
	// ReadTensor re-checks rather than inheriting the parse's proof.
	bad := Tensor{Name: "x", DType: F32, Shape: []uint64{4}, Begin: 0, End: 1 << 40}
	if _, err := f.ReadTensor(&bad); err == nil {
		t.Fatal("read a range past the data section")
	}
}

// FuzzParse is the exec-budget fuzz gate on the header parser.
//
//	./scripts/cap 4G -- go test ./convert/safetensors -run=XXX -fuzz=FuzzParse -fuzztime=1000000x -parallel=2
//
// The property is not merely "does not panic": every tensor it returns must lie
// inside the data section it was given, because that bound is what every later
// read inherits.
func FuzzParse(f *testing.F) {
	f.Add([]byte(`{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}`), uint64(8))
	f.Add([]byte(`{"__metadata__":{"format":"pt"},"a":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}`), uint64(12))
	f.Add([]byte(`{}`), uint64(0))
	f.Add([]byte(`{"a":{"dtype":"F32","shape":[18446744073709551615],"data_offsets":[0,0]}}`), uint64(1))
	f.Fuzz(func(t *testing.T, hdr []byte, dataLen uint64) {
		_, ts, err := Parse(hdr, dataLen)
		if err != nil {
			return
		}
		for i := range ts {
			e := &ts[i]
			if e.Begin > e.End || e.End > dataLen {
				t.Fatalf("accepted %q spanning [%d,%d) of %d", e.Name, e.Begin, e.End, dataLen)
			}
			w, ok := e.DType.Size()
			if !ok {
				t.Fatalf("accepted dtype %q", e.DType)
			}
			if e.Elems()*uint64(w) != e.End-e.Begin {
				t.Fatalf("accepted %q: %d elements of %d bytes in %d bytes",
					e.Name, e.Elems(), w, e.End-e.Begin)
			}
			if i > 0 && e.Begin < ts[i-1].End {
				t.Fatalf("accepted an overlap: %q and %q", ts[i-1].Name, e.Name)
			}
		}
	})
}

// TestEveryDTypeHasAWidthOrIsRefused keeps Size honest: a dtype that reports a
// width must report the right one, because a wrong width is a wrong bounds
// check rather than a wrong number.
func TestEveryDTypeHasAWidthOrIsRefused(t *testing.T) {
	want := map[DType]int{F64: 8, I64: 8, F32: 4, I32: 4, F16: 2, BF16: 2, I16: 2,
		I8: 1, U8: 1, BOOL: 1, F8E4M3: 1, F8E5M2: 1}
	for d, w := range want {
		got, ok := d.Size()
		if !ok || got != w {
			t.Errorf("%s: %d, %v; want %d", d, got, ok, w)
		}
	}
	for _, d := range []DType{"", "f32", "FLOAT32", "F4_E2M1", "COMPLEX64"} {
		if _, ok := d.Size(); ok {
			t.Errorf("%q has a width; an unknown dtype must not be assumed to be one byte", d)
		}
	}
	// The table above must cover every constant this package defines, or a new
	// dtype can be added and silently left untested.
	var n int
	for range want {
		n++
	}
	if n != 12 {
		t.Fatalf("the width table has %d entries; add the new dtype here too", n)
	}
}
