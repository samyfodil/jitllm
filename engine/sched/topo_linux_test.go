//go:build linux

package sched

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"syscall"
	"testing"
	"unsafe"
)

// TestTopologyClassifiesTheMachinesThisEngineRunsOn reads synthetic sysfs
// trees for the four machine shapes this engine has, plus the hybrid corner,
// and checks PCores, ECores and SMTSiblings on each.
//
// The Atom row matters: a Goldmont Atom has four single-threaded cores and no
// SMT, and "single-threaded => E-core" would leave PCores() empty.
func TestTopologyClassifiesTheMachinesThisEngineRunsOn(t *testing.T) {
	ok := allowed()
	keep := func(cpus []int) []int {
		var out []int
		for _, c := range cpus {
			if len(ok) == 0 || ok[c] {
				out = append(out, c)
			}
		}
		return out
	}
	// A P-core is taken at the first of its threads the mask allows (see
	// TestPCoresTakesTheThreadTheMaskAllows), so under a taskset that drops a
	// core's first thread its expected CPU is the sibling the mask keeps, not
	// nothing: keepCores maps each listed P-core to that thread.
	keepCores := func(sibs []string, p []int) []int {
		isP := map[string]bool{}
		for _, c := range p {
			isP[sibs[c]] = true
		}
		var out []int
		seen := map[string]bool{}
		for cpu, s := range sibs {
			if len(ok) > 0 && !ok[cpu] || seen[s] {
				continue
			}
			seen[s] = true
			if isP[s] {
				out = append(out, cpu)
			}
		}
		return out
	}
	pairs := func(n int) []string { // cpu 2i and 2i+1 are one core, Alder Lake style
		var s []string
		for i := 0; i < n; i++ {
			s = append(s, fmt.Sprintf("%d-%d", i&^1, i|1))
		}
		return s
	}
	singles := func(from, n int) []string {
		var s []string
		for i := from; i < from+n; i++ {
			s = append(s, fmt.Sprint(i))
		}
		return s
	}
	cases := []struct {
		name      string
		sibs      []string
		atoms     string // "" = the kernel publishes no hybrid split
		p, e, smt []int
	}{
		{"alder lake, 6P+8E, kernel publishes cpu_atom",
			append(pairs(12), singles(12, 8)...), "12-19",
			[]int{0, 2, 4, 6, 8, 10}, []int{12, 13, 14, 15, 16, 17, 18, 19},
			[]int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}},
		{"alder lake, older kernel without cpu_atom",
			append(pairs(12), singles(12, 8)...), "",
			[]int{0, 2, 4, 6, 8, 10}, []int{12, 13, 14, 15, 16, 17, 18, 19},
			[]int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}},
		{"goldmont atom C3558, 4 cores, no SMT",
			singles(0, 4), "",
			[]int{0, 1, 2, 3}, nil, []int{0, 1, 2, 3}},
		{"xeon D-2123IT, 4 cores x 2 threads, siblings 0/4",
			[]string{"0,4", "1,5", "2,6", "3,7", "0,4", "1,5", "2,6", "3,7"}, "",
			[]int{0, 1, 2, 3}, nil, []int{0, 4, 1, 5, 2, 6, 3, 7}},
		{"hybrid with SMT disabled, kernel publishes cpu_atom",
			singles(0, 10), "6-9",
			[]int{0, 1, 2, 3, 4, 5}, []int{6, 7, 8, 9}, []int{0, 1, 2, 3, 4, 5}},
	}
	old := sysRoot
	defer func() { sysRoot = old }()
	for _, c := range cases {
		root := t.TempDir()
		for cpu, s := range c.sibs {
			d := filepath.Join(root, "devices/system/cpu", fmt.Sprintf("cpu%d", cpu), "topology")
			if err := os.MkdirAll(d, 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(d, "thread_siblings_list"), []byte(s+"\n"), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		if c.atoms != "" {
			d := filepath.Join(root, "devices/cpu_atom")
			if err := os.MkdirAll(d, 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(d, "cpus"), []byte(c.atoms+"\n"), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		sysRoot = root
		if got, want := PCores(), keepCores(c.sibs, c.p); !reflect.DeepEqual(got, want) {
			t.Errorf("%s: PCores = %v, want %v", c.name, got, want)
		}
		if got, want := ECores(), keep(c.e); !reflect.DeepEqual(got, want) {
			t.Errorf("%s: ECores = %v, want %v", c.name, got, want)
		}
		if got, want := SMTSiblings(), keep(c.smt); !reflect.DeepEqual(got, want) {
			t.Errorf("%s: SMTSiblings = %v, want %v", c.name, got, want)
		}
		if p, want := DecodeCores(), keepCores(c.sibs, c.p); len(want) > 0 && len(p) != len(want) {
			t.Errorf("%s: DecodeCores = %v, want every P-core %v", c.name, p, want)
		}
	}
}

// TestPCoresTakesTheThreadTheMaskAllows: a process confined to the second
// thread of each core still gets one worker per core. PCores once marked a core
// seen at its first thread before asking whether the mask allows it, so a
// taskset of only the hyperthreads found no P-core at all and DecodeCores fell
// back to a one-worker pool outside the mask.
func TestPCoresTakesTheThreadTheMaskAllows(t *testing.T) {
	if runtime.NumCPU() < 8 {
		t.Skipf("%d CPUs: the synthetic xeon's threads 4-7 are not on this machine -- this gate proved nothing", runtime.NumCPU())
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	setMask := func(cpus map[int]bool) {
		t.Helper()
		var mask [16]uint64
		for c := range cpus {
			mask[c/64] |= 1 << uint(c%64)
		}
		if _, _, errno := syscall.RawSyscall(syscall.SYS_SCHED_SETAFFINITY,
			0, uintptr(len(mask)*8), uintptr(unsafe.Pointer(&mask[0]))); errno != 0 {
			t.Fatalf("sched_setaffinity: %v", errno)
		}
	}
	was := allowed()
	if !was[4] || !was[5] || !was[6] || !was[7] {
		t.Skipf("the mask %v does not hold CPUs 4-7 -- this gate proved nothing", was)
	}
	defer setMask(was)
	root := t.TempDir()
	for cpu, s := range []string{"0,4", "1,5", "2,6", "3,7", "0,4", "1,5", "2,6", "3,7"} {
		d := filepath.Join(root, "devices/system/cpu", fmt.Sprintf("cpu%d", cpu), "topology")
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(d, "thread_siblings_list"), []byte(s+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	old := sysRoot
	defer func() { sysRoot = old }()
	sysRoot = root
	setMask(map[int]bool{4: true, 5: true, 6: true, 7: true})
	if got, want := PCores(), []int{4, 5, 6, 7}; !reflect.DeepEqual(got, want) {
		t.Fatalf("confined to the second thread of each core, PCores = %v, want %v", got, want)
	}
	if got := DecodeCores(); len(got) != 4 {
		t.Fatalf("DecodeCores = %v: four cores are in the mask", got)
	}
}
