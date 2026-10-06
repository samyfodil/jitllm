//go:build linux

package sched

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"syscall"
	"unsafe"
)

// allowed is the set of CPUs this process may actually run on, which is not the
// same question as which CPUs the machine has. A pool sized from raw topology
// would start six workers under `taskset -c 0,2`, oversubscribing two CPUs.
// Empty means "could not ask", and every caller then falls back to topology.
func allowed() map[int]bool {
	var mask [16]uint64 // 1024 CPUs, the same width pin() used
	_, _, errno := syscall.RawSyscall(syscall.SYS_SCHED_GETAFFINITY,
		0, uintptr(len(mask)*8), uintptr(unsafe.Pointer(&mask[0])))
	if errno != 0 {
		return nil
	}
	out := map[int]bool{}
	for i, w := range mask {
		for b := 0; b < 64; b++ {
			if w&(1<<uint(b)) != 0 {
				out[i*64+b] = true
			}
		}
	}
	return out
}

// sysRoot is where the topology is read from; a test points it at a synthetic
// tree so the classification can be checked on machines the test runtime does not have access to.
var sysRoot = "/sys"

// topo is the machine's core layout as sysfs states it: one thread_siblings_list
// per logical CPU, from cpu0 up to the first gap, and the kernel's own list of
// efficiency cores where it publishes one.
type topo struct {
	sibs  []string     // index = logical CPU
	atoms map[int]bool // /sys/devices/cpu_atom/cpus; nil when absent
	smt   bool         // some core has more than one thread
}

func readTopo() topo {
	var t topo
	for cpu := 0; cpu < 512; cpu++ {
		b, err := os.ReadFile(fmt.Sprintf("%s/devices/system/cpu/cpu%d/topology/thread_siblings_list", sysRoot, cpu))
		if err != nil {
			break
		}
		s := strings.TrimSpace(string(b))
		t.sibs = append(t.sibs, s)
		t.smt = t.smt || multiThread(s)
	}
	if b, err := os.ReadFile(sysRoot + "/devices/cpu_atom/cpus"); err == nil {
		t.atoms = parseCPUList(strings.TrimSpace(string(b)))
	}
	return t
}

func multiThread(sibs string) bool { return strings.ContainsAny(sibs, ",-") }

// efficiency reports whether a single-threaded logical CPU is an efficiency
// core.
//
// "Single-threaded means E-core" holds only on a machine with SMT somewhere; a
// Goldmont Atom has none and would read as all E-cores. The kernel's own
// hybrid list (cpu_atom/cpus) decides when it exists; otherwise a machine with
// no SMT anywhere is homogeneous. A hybrid part with SMT disabled on a kernel
// without cpu_atom reads as homogeneous, erring toward using a core.
func (t topo) efficiency(cpu int) bool {
	if t.atoms != nil {
		return t.atoms[cpu]
	}
	return t.smt
}

// parseCPUList parses the kernel's cpulist format: "0-3,8,10-11".
func parseCPUList(s string) map[int]bool {
	out := map[int]bool{}
	for _, part := range strings.Split(s, ",") {
		lo, hi, rng := strings.Cut(part, "-")
		a, err := strconv.Atoi(lo)
		if err != nil {
			continue
		}
		b := a
		if rng {
			if b, err = strconv.Atoi(hi); err != nil {
				continue
			}
		}
		for i := a; i <= b; i++ {
			out[i] = true
		}
	}
	return out
}

// PCores returns one logical CPU per physical performance core.
//
// Detection is by SMT topology rather than a hardcoded list: on a hybrid part,
// P-cores are SMT pairs (thread_siblings_list "0-1") and E-cores are
// single-threaded ("12"). Taking the first sibling of each pair yields one CPU
// per distinct P-core and never schedules two workers onto the two
// hyperthreads of one core, which would share the same load ports and measure
// as a barrier straggler.
//
// On a non-hybrid machine every core is an SMT pair and this returns all of
// them; on one with no SMT at all every core is returned (see topo.efficiency).
func PCores() []int {
	t := readTopo()
	if len(t.sibs) == 0 {
		return nil // no topology exposed; caller falls back
	}
	ok := allowed()
	var out []int
	seen := map[string]bool{}
	for cpu, sibs := range t.sibs {
		// The mask first: a core is taken at the first of its threads the
		// process may run on, which under `taskset` of the second threads is
		// not its first thread at all.
		if len(ok) > 0 && !ok[cpu] {
			continue // the process cannot run here; see allowed
		}
		if seen[sibs] {
			continue // already took one thread of this physical core
		}
		seen[sibs] = true
		if !multiThread(sibs) && t.efficiency(cpu) {
			continue // an E-core: no read bandwidth to add
		}
		out = append(out, cpu)
	}
	return out
}

// ECores returns one logical CPU per efficiency core: single-threaded in the
// topology, so no SMT sibling to skip, and an efficiency core by
// topo.efficiency -- so a homogeneous machine without SMT has none.
func ECores() []int {
	t := readTopo()
	ok := allowed()
	var out []int
	for cpu, sibs := range t.sibs {
		if !multiThread(sibs) && t.efficiency(cpu) && (len(ok) == 0 || ok[cpu]) {
			out = append(out, cpu)
		}
	}
	return out
}

// SMTSiblings returns every logical CPU of every performance core, both threads
// of each. Included so the hyperthread question can be measured rather than
// assumed -- two threads sharing one core's load ports is a real experiment,
// not obviously a bad idea. A single-threaded performance core contributes its
// one thread.
func SMTSiblings() []int {
	t := readTopo()
	ok := allowed()
	var out []int
	seen := map[int]bool{}
	for cpu, sibs := range t.sibs {
		if !multiThread(sibs) {
			if !t.efficiency(cpu) && !seen[cpu] && (len(ok) == 0 || ok[cpu]) {
				seen[cpu] = true
				out = append(out, cpu)
			}
			continue
		}
		for _, f := range strings.FieldsFunc(sibs, func(r rune) bool { return r == ',' || r == '-' }) {
			if n, err := strconv.Atoi(f); err == nil && !seen[n] {
				seen[n] = true
				if len(ok) == 0 || ok[n] {
					out = append(out, n)
				}
			}
		}
	}
	return out
}
