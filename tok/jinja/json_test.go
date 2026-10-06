package jinja

import "testing"

// TestToJSONKeepsKeyOrder: a tool schema decoded with FromJSON renders through
// tojson -- compact and indented -- in the order it was written, exactly as
// Python's json.dumps writes it. Llama-3.x templates use the indented form
// for every tool.
func TestToJSONKeepsKeyOrder(t *testing.T) {
	v, err := FromJSON([]byte(`{"b": 1, "a": [1, 2.5, "x<y"], "c": {}, "d": []}`))
	if err != nil {
		t.Fatal(err)
	}
	for src, want := range map[string]string{
		"{{ x | tojson }}":           `{"b": 1, "a": [1, 2.5, "x<y"], "c": {}, "d": []}`,
		"{{ x | tojson(indent=2) }}": "{\n  \"b\": 1,\n  \"a\": [\n    1,\n    2.5,\n    \"x<y\"\n  ],\n  \"c\": {},\n  \"d\": []\n}",
		"{{ x.a[0] + 1 }}":           "2",
	} {
		tpl, err := Compile(src)
		if err != nil {
			t.Fatal(err)
		}
		got, err := tpl.Render(map[string]any{"x": v})
		if err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Errorf("%s:\n got  %q\n want %q", src, got, want)
		}
	}
	if _, err := FromJSON([]byte(`{"a": 1} x`)); err == nil {
		t.Error("trailing data was accepted")
	}
}
