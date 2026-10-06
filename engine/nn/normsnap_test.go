package nn

import (
	"math"
	"sync"
	"testing"
)

// TestNormDispatchIsConcurrentSafe hammers both norm dispatchers from many
// goroutines across several widths, which is what engine/model/prefill.go does when it
// calls s.rmsnorm from inside Parallel(ntok, ...).
//
// The gate is -race: readers load an immutable snapshot without a lock, so the
// failure to catch is a concurrent map access. Against normFill mutating the
// live snapshot in place, this fails under -race; without -race it still
// checks the arithmetic.
func TestNormDispatchIsConcurrentSafe(t *testing.T) {
	// Several widths, so the snapshot is genuinely rewritten while other
	// goroutines are reading it. A single width would fill on the first call
	// and never exercise copy-on-write again.
	widths := []int{256, 512, 1024, 2048, 4096}
	var wg sync.WaitGroup
	errs := make(chan string, 64)
	for g := 0; g < 16; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for rep := 0; rep < 40; rep++ {
				n := widths[(g+rep)%len(widths)]
				x := make([]float32, n)
				w := make([]float32, n)
				y := make([]float32, n)
				for i := range x {
					x[i] = float32((i%23)-11) / 8
					w[i] = 1 + float32(i%3)/4
				}
				eps := 1e-5 + float64(g%2)*1e-6 // two eps, so konst fills too
				RMSNorm32JIT(y, x, w, eps)
				// Oracle: the float64 reference.
				var ss float64
				for _, v := range x {
					ss += float64(v) * float64(v)
				}
				scale := 1 / math.Sqrt(ss/float64(n)+eps)
				for i := range y {
					want := float32(float64(x[i]) * scale * float64(w[i]))
					if d := math.Abs(float64(y[i] - want)); d > 1e-4 {
						errs <- "rmsnorm mismatch"
						return
					}
				}
				b := make([]float32, n)
				LayerNorm32JIT(y, x, w, b, eps)
			}
		}(g)
	}
	wg.Wait()
	close(errs)
	for e := range errs {
		t.Fatal(e)
	}
}
