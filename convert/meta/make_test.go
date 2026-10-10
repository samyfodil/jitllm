package meta_test

import (
	"testing"

	"github.com/jitllm/jitllm/convert/meta"
)

// TestMakeRoundTrips builds every value kind from ordinary Go values and reads
// them back with the accessors the engine uses. It is the only caller of Make*
// (see convert/meta/value.go).
func TestMakeRoundTrips(t *testing.T) {
	// Narrowing is deliberate: a count that fits a byte must not round-trip as
	// 64 bits, or every converted model grows a metadata section for nothing.
	for _, c := range []struct {
		n    uint64
		want meta.ValueType
	}{{7, meta.Uint8}, {300, meta.Uint16}, {70000, meta.Uint32}, {1 << 33, meta.Uint64}} {
		v := meta.MakeUint(c.n)
		if v.Type != c.want {
			t.Errorf("MakeUint(%d) is %v, want %v", c.n, v.Type, c.want)
		}
		if got, ok := v.Uint(); !ok || got != c.n {
			t.Errorf("MakeUint(%d) read back %d, %v", c.n, got, ok)
		}
	}
	if got, ok := meta.MakeInt(-3).Int(); !ok || got != -3 {
		t.Errorf("MakeInt(-3) read back %d, %v", got, ok)
	}
	if got, ok := meta.MakeFloat(1.5).Float(); !ok || got != 1.5 {
		t.Errorf("MakeFloat(1.5) read back %v, %v", got, ok)
	}
	for _, want := range []bool{true, false} {
		v := meta.MakeBool(want)
		got, ok := v.Int()
		if !ok || (got != 0) != want {
			t.Errorf("MakeBool(%v) read back %d, %v", want, got, ok)
		}
	}
	if got, ok := meta.MakeString("héllo").String(); !ok || got != "héllo" {
		t.Errorf("MakeString read back %q, %v", got, ok)
	}

	want := []string{"", "a", "▁the", "🙂", "with\x00nul"}
	v := meta.MakeStrings(want)
	if v.Len() != len(want) {
		t.Fatalf("MakeStrings Len %d, want %d", v.Len(), len(want))
	}
	var got []string
	if err := v.EachString(func(i int, s []byte) bool { got = append(got, string(s)); return true }); err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("read back %d strings, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("string %d is %q, want %q", i, got[i], want[i])
		}
	}

	fs := []float32{-1, 0, 0.5, 1e9}
	gf, err := meta.MakeFloat32s(fs).Float32s()
	if err != nil {
		t.Fatal(err)
	}
	for i := range fs {
		if gf[i] != fs[i] {
			t.Errorf("float %d is %v, want %v", i, gf[i], fs[i])
		}
	}
	is := []int32{-2, 0, 1, 1 << 30}
	gi, err := meta.MakeInt32s(is).Int32s()
	if err != nil {
		t.Fatal(err)
	}
	for i := range is {
		if gi[i] != is[i] {
			t.Errorf("int %d is %v, want %v", i, gi[i], is[i])
		}
	}
}
