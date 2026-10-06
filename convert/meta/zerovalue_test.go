package meta

import "testing"

// TestZeroValueNeverPanics: a missing key is the zero Value (`f.KV["absent"]`),
// whose ValueType is Uint8; its accessors must report false rather than index
// a nil Raw.
func TestZeroValueNeverPanics(t *testing.T) {
	var z Value // exactly what a missing map key returns

	if n, ok := z.Int(); ok || n != 0 {
		t.Errorf("zero Value Int() = %d, %v; want 0, false", n, ok)
	}
	if n, ok := z.Uint(); ok || n != 0 {
		t.Errorf("zero Value Uint() = %d, %v; want 0, false", n, ok)
	}
	if f, ok := z.Float(); ok || f != 0 {
		t.Errorf("zero Value Float() = %v, %v; want 0, false", f, ok)
	}
	if s, ok := z.String(); ok || s != "" {
		t.Errorf("zero Value String() = %q, %v; want \"\", false", s, ok)
	}
	if n := z.Len(); n != 0 {
		t.Errorf("zero Value Len() = %d; want 0", n)
	}

	// And a value whose Raw is short for its declared type, which is what a
	// truncated or hand-built Value looks like. Every one must be comma-ok
	// false rather than a slice bounds panic.
	for _, tc := range []struct {
		name string
		v    Value
	}{
		{"Uint8 empty", Value{Type: Uint8}},
		{"Int8 empty", Value{Type: Int8}},
		{"Bool empty", Value{Type: Bool}},
		{"Uint16 short", Value{Type: Uint16, Raw: []byte{1}}},
		{"Int16 short", Value{Type: Int16, Raw: []byte{1}}},
		{"Uint32 short", Value{Type: Uint32, Raw: []byte{1, 2, 3}}},
		{"Int32 short", Value{Type: Int32, Raw: []byte{1, 2, 3}}},
		{"Uint64 short", Value{Type: Uint64, Raw: []byte{1, 2, 3, 4, 5, 6, 7}}},
		{"Int64 short", Value{Type: Int64, Raw: []byte{1}}},
		{"Float32 short", Value{Type: Float32, Raw: []byte{1, 2, 3}}},
		{"Float64 short", Value{Type: Float64, Raw: []byte{1}}},
	} {
		if n, ok := tc.v.Int(); ok {
			t.Errorf("%s: Int() = %d, true; want false", tc.name, n)
		}
		if f, ok := tc.v.Float(); ok {
			t.Errorf("%s: Float() = %v, true; want false", tc.name, f)
		}
	}
}
