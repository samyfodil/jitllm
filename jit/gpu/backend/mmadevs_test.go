package backend_test

import (
	"testing"

	"github.com/samyfodil/jitllm/jit/gpu/backend"
	"github.com/samyfodil/jitllm/jit/gpu/ir"
)

// mmaDevices is the devices that can compile an ir.OpMMA, asked once each, so a
// backend without MMA does not produce a skip for every cell of a sweep.
// lowertest's TestMMACoversExactlyWhatItClaims asserts which backends lower MMA
// and why the others decline.
//
// The probe is one tile with the results stored, so nothing is dead-coded away,
// and it compiles the same shape the sweeps use.
func mmaDevices(t *testing.T, devs []backend.Device, sh ir.MMAShape) []backend.Device {
	t.Helper()
	b := ir.New("mmacap", [3]int{32, 1, 1})
	pA, pB := b.Param("pA", ir.U32), b.Param("pB", ir.U32)
	pOut := b.Param("pOut", ir.I32)
	na, nb, nc := sh.Frags()
	a := make([]ir.Value, na)
	for i := range a {
		a[i] = b.Load(ir.U32, pA, b.Const(ir.U32, int64(i)), 0)
	}
	bb := make([]ir.Value, nb)
	for i := range bb {
		bb[i] = b.Load(ir.U32, pB, b.Const(ir.U32, int64(i)), 0)
	}
	c := make([]ir.Value, nc)
	for i := range c {
		c[i] = b.Const(sh.Elem(), 0)
	}
	for i, v := range b.MMA(sh, a, bb, c) {
		b.Store(pOut, b.Const(ir.U32, int64(i)), v, 0)
	}
	probe := b.Done()

	out := make([]backend.Device, 0, len(devs))
	for _, d := range devs {
		k, err := d.Compile(probe)
		if err != nil {
			t.Logf("%s: no MMA lowering, not swept: %v", d.API(), err)
			continue
		}
		k.Close()
		out = append(out, d)
	}
	return out
}
