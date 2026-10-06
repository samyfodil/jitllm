package nn

import (
	"testing"

	"github.com/samyfodil/jitllm/format/quant"
)

// TestTuneKeyDistinguishes: the key must move when anything the tuned answer
// depends on moves, and must not move for two lookups of the same host and
// model.
func TestTuneKeyDistinguishes(t *testing.T) {
	a := []shapeKey{{t: 0, nb: 64}}
	b := []shapeKey{{t: 0, nb: 128}}
	if tuneKey("pack", a, 6) != tuneKey("pack", a, 6) {
		t.Fatal("the key is not stable for one host and model")
	}
	for _, c := range []struct {
		name string
		x, y string
	}{
		{"shape set", tuneKey("pack", a, 6), tuneKey("pack", b, 6)},
		{"core count", tuneKey("pack", a, 6), tuneKey("pack", a, 4)},
		{"parameter", tuneKey("pack", a, 6), tuneKey("chunk", a, 6)},
	} {
		if c.x == c.y {
			t.Errorf("the key does not separate a different %s", c.name)
		}
	}
	if hostSig() == "" {
		t.Fatal("hostSig is empty; the key carries no host identity at all")
	}
	t.Logf("hostSig = %s", hostSig())
	t.Logf("key     = %s", tuneKey("pack", a, 6))
}

// TestContainerShapesKeyTheTuners opens two JITs on two container models that
// differ only in k and demands their tuner keys differ.
//
// The row-major family's shape list is empty for a container, so a key built
// from it would make every container model on a host share one set of answers.
func TestContainerShapesKeyTheTuners(t *testing.T) {
	key := func(k int) string {
		j := NewJIT(k, 4096, []quant.Type{quant.Q4_K})
		if j == nil {
			t.Skip("no generated tier")
		}
		defer j.Close()
		j.AddShape(quant.Q4_K, k)
		if len(j.keyShapes) == 0 {
			t.Fatal("AddShape of a packed shape left the tuner key's shape set empty")
		}
		return tuneKey("chunk", j.keyShapes, j.pool.Max())
	}
	if a, b := key(2048), key(4096); a == b {
		t.Fatalf("two models with different shapes share the tuner key %s", a)
	}
}
