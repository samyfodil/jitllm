// Command fmtbench prices each quant format's CPU decode matvec against a read
// wall measured in the same process, every shape sized DRAM-resident.
//
// Holding the memory system fixed makes it a controlled experiment: a format
// that reads fewer bytes per element and goes slower can only be losing to
// arithmetic, which says which kernel is worth opening.
package main

import (
	"fmt"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"strconv"
	"time"

	"github.com/samyfodil/jitllm/format/quant"

	"github.com/samyfodil/jitllm/dev/bench"
	"github.com/samyfodil/jitllm/engine/nn"
	"github.com/samyfodil/jitllm/engine/sched"
)

// cores reads JITLLM_CORES here, in the command, because sched and nn read no
// environment: the number has to reach BOTH the wall probe and the JIT's pool,
// and passing it to only one of them would compare a rate against a wall taken
// at a different thread count.
func cores() int {
	if v := os.Getenv("JITLLM_CORES"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return runtime.NumCPU()
}

func main() {
	wall := bench.MemWall(cores()) / 1e9
	fmt.Printf("read wall (Go probe, reads LOW -- RULE 8b): %.1f GB/s\n"+
		"every shape is sized to ~280 MB so NOTHING is measured out of the 24 MB L3\n\n", wall)
	fmt.Printf("%-24s %8s %7s %6s %8s %8s %8s\n", "shape", "rows", "k", "type", "MB", "GB/s", "of wall")

	for _, s := range []struct {
		name    string
		rows, k int
		t       quant.Type
	}{
		// One row per format, same k, rows chosen so every set is DRAM-resident.
		// The name is the format because that is the variable: the shapes are
		// held fixed so a difference can only be the unpack and the tail.
		{"Q8_0", 1, 2048, quant.Q8_0},
		{"Q4_0", 1, 2048, quant.Q4_0},
		{"Q4_K", 1, 2048, quant.Q4_K},
		{"Q5_K", 1, 2048, quant.Q5_K},
		{"Q6_K", 1, 2048, quant.Q6_K},
		{"Q3_K", 1, 2048, quant.Q3_K},
		// The same formats at the sizes a real model uses. If these are much
		// slower per byte than the DRAM-sized rows, the gap is per-matvec
		// overhead and not the kernel.
		{"gemma gate/up REAL", 16384, 2048, quant.Q4_0},
		{"gemma q/o REAL", 2048, 2048, quant.Q4_0},
		{"gemma ffn_down REAL", 2048, 16384, quant.Q4_0},
		{"gemma lm_head REAL", 256128, 2048, quant.Q8_0},
		{"tinyllama gate/up REAL", 5632, 2048, quant.Q3_K},
	} {
		// Rows are scaled so every shape is DRAM-resident: a working set under
		// the last-level cache reports cache as DRAM. The k is the model's;
		// only the row count moves.
		const want = 280 << 20
		nb := s.k / int(s.t.BlockElems())
		rowB := nb * int(s.t.BlockBytes())
		if s.rows == 1 {
			s.rows = want / rowB
		}
		w := make([]byte, s.rows*rowB)
		for i := range w {
			w[i] = byte(rand.Intn(256))
		}
		f := nn.NewJIT(s.k, s.rows, []quant.Type{s.t}, nn.WithSched(sched.WithCores(cores())))
		if f == nil {
			fmt.Printf("%-24s no fast tier\n", s.name)
			continue
		}
		f.AddShape(s.t, s.k)
		x := make([]float32, s.k)
		for i := range x {
			x[i] = float32(rand.NormFloat64())
		}
		out := make([]float32, s.rows)
		f.NewInput()
		f.MatVec(out, s.t, w, x, s.rows, s.k)
		// Soak, then take the median, not best-of-N: the clock settles well
		// below burst within seconds, and best-of-N would measure the boost
		// while a real decode runs at steady state.
		deadline := time.Now().Add(1500 * time.Millisecond)
		for time.Now().Before(deadline) {
			f.NewInput()
			f.MatVec(out, s.t, w, x, s.rows, s.k)
		}
		// NewInput is not called in the timed loop: it forces a re-quantization
		// of the activation that the engine does once per shared input, not
		// per matvec.
		ds := make([]time.Duration, 0, 15)
		for r := 0; r < 15; r++ {
			t0 := time.Now()
			f.MatVec(out, s.t, w, x, s.rows, s.k)
			ds = append(ds, time.Since(t0))
		}
		sort.Slice(ds, func(i, j int) bool { return ds[i] < ds[j] })
		best := ds[len(ds)/2]
		gbs := float64(len(w)) / best.Seconds() / 1e9
		// A working set under the last-level cache is not a DRAM measurement:
		// the real-shape rows fit in cache, so their "% of wall" is meaningless.
		// Read the DRAM-sized rows.
		note := ""
		if len(w) < 32<<20 {
			note = "   <- under L3: NOT a DRAM number"
		}
		fmt.Printf("%-24s %8d %7d %6s %8.1f %8.1f %7.0f%%%s\n",
			s.name, s.rows, s.k, s.t, float64(len(w))/(1<<20), gbs, gbs/wall*100, note)
		f.Close()
	}
}
