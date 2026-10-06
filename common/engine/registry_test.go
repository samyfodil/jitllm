package engine

import "testing"

// Opening a model that is already open is a switch, not a second copy: two
// entries over one file would be two pagers over the same bytes, neither
// seeing the other's residency.
func TestAnOpenModelIsFoundRatherThanOpenedAgain(t *testing.T) {
	e := &Engine{models: []*entry{{path: "/m/a.jlm"}, {path: "/m/b.jlm"}}}

	if got := e.find("/m/a.jlm"); got == nil || got.path != "/m/a.jlm" {
		t.Fatalf("an open model was not found: %v", got)
	}
	if got := e.find("/m/c.jlm"); got != nil {
		t.Errorf("a model that is not open was found: %v", got)
	}
}

// The registry keeps load order, which the budget division and the colour
// palette index.
func TestTheRegistryKeepsLoadOrder(t *testing.T) {
	e := &Engine{models: []*entry{{path: "a"}, {path: "b"}, {path: "c"}}}
	want := []string{"a", "b", "c"}
	got := e.paths()
	if len(got) != len(want) {
		t.Fatalf("got %d paths, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("path %d is %q, want %q", i, got[i], want[i])
		}
	}

	// Dropping the middle one keeps the rest in order.
	e.drop(e.models[1])
	if got := e.paths(); len(got) != 2 || got[0] != "a" || got[1] != "c" {
		t.Errorf("after dropping the middle entry: %v", got)
	}
}

// Exactly one model is active, and its name and colour follow its position.
//
// The colour is load order, not a hash of the path, so colours are stable
// within a session and start at the beginning of the palette.
func TestTheActiveModelNamesItselfByPosition(t *testing.T) {
	a, b := &entry{path: "/m/first.jlm"}, &entry{path: "/m/second.jlm"}
	e := &Engine{models: []*entry{a, b}, active: b}

	name, colour := e.activeName()
	if name != "second.jlm" {
		t.Errorf("active name %q, want %q", name, "second.jlm")
	}
	if colour != 1 {
		t.Errorf("active colour %d, want 1 (its load position)", colour)
	}

	e.active = nil
	if name, colour := e.activeName(); name != "" || colour != 0 {
		t.Errorf("with nothing active: %q / %d", name, colour)
	}
}
