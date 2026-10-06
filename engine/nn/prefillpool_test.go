//go:build amd64 || arm64

package nn

import (
	"runtime"
	"testing"

	"github.com/samyfodil/jitllm/engine/sched"
	"github.com/samyfodil/jitllm/format/quant"
)

// TestPrefillPoolIsSwappedInAndOut: BeginPrefill runs on the prefill pool and
// EndPrefill returns to the decode pool, and "p" builds no second pool. The
// selection is the whole feature -- a prefill that stayed on the decode pool
// computes the same numbers, so no numeric gate can tell the two apart.
func TestPrefillPoolIsSwappedInAndOut(t *testing.T) {
	pe := len(sched.CoreSet("pe"))
	if pe <= len(sched.DecodeCores()) {
		t.Skipf("no core set here is wider than the decode pool's %d; the swap has nothing to select", pe)
	}
	// cmd/jitllm runs at the decode width; the prefill phase must raise
	// GOMAXPROCS to its pool and put back exactly what it found.
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(0))
	decode := len(sched.DecodeCores())
	for _, c := range []struct {
		set  string
		wide bool
	}{{"pe", true}, {"p", false}} {
		runtime.GOMAXPROCS(decode)
		f := NewJIT(256, 256, []quant.Type{quant.Q4_K}, WithPrefillCoreSet(c.set))
		n := f.Workers()
		f.BeginPrefill()
		w, procs := f.Workers(), runtime.GOMAXPROCS(0)
		f.EndPrefill()
		back, after := f.Workers(), runtime.GOMAXPROCS(0)
		f.Close()
		t.Logf("%s: decode %d workers, prefill %d at GOMAXPROCS %d, after %d at %d",
			c.set, n, w, procs, back, after)
		if back != n || after != decode {
			t.Fatalf("%s: EndPrefill left %d workers at GOMAXPROCS %d, want %d at %d", c.set, back, after, n, decode)
		}
		switch {
		case c.wide && (w != pe || procs < pe):
			t.Fatalf("%s: prefill ran on %d workers at GOMAXPROCS %d, want the %d of the pe set", c.set, w, procs, pe)
		case !c.wide && (w != n || procs != decode):
			t.Fatalf("%s: prefill ran on %d workers at GOMAXPROCS %d, want the decode pool's %d at %d", c.set, w, procs, n, decode)
		}
	}
}
