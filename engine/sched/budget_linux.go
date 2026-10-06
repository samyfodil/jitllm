//go:build linux

package sched

import (
	"os"
	"strconv"
	"strings"
)

// MemBudget is how many bytes of weights may be resident, when the caller does
// not say.
//
// It reads the cgroup as well as /proc/meminfo: under scripts/cap a process
// with plenty of MemAvailable can be limited far lower, and cgroup v2 charges
// page cache to the cgroup, so the cgroup number decides whether a model
// thrashes. The smaller of the two limits wins, and a slice is left for
// everything that is not weights -- the KV cache, the scratch, the Go heap and
// the runtime itself.
func MemBudget() uint64 {
	avail := meminfoAvailable()
	if c := cgroupLimit(); c > 0 && (avail == 0 || c < avail) {
		avail = c
	}
	if avail == 0 {
		return 0 // unknown: the caller decides what that means
	}
	// Not all of it: the KV cache is unreclaimable, the pool has scratch, and a
	// cgroup at MemoryHigh is throttled (which is what systemd-oomd reacts to).
	// Eight tenths leaves room and keeps the budget dominated by weights.
	b := avail / 10 * 8
	// With the weights on the Go heap, eight tenths can leave the permanently
	// live page frames on the collector's goal, so GCBudgetCap applies; off
	// the heap (the default) it is no cap. See SetGCReserve.
	if cap := gcBudgetCap(); cap > 0 && cap < b {
		b = cap
	}
	return b
}

// MemLimit is the hard ceiling this process must stay under, or 0 when there is
// none: the cgroup's, not a share of it.
//
// It exists for Go's garbage collector, which does not know about cgroups: at
// GOGC=100 a heap holding 18 GiB of page frames plans for ~36 GiB, which under
// memory.high means permanent direct reclaim. The caller passes it to
// debug.SetMemoryLimit, which is process-wide and so not a library's decision.
// MemBudget is a share of this, for weights.
func MemLimit() uint64 { return cgroupLimit() }

// cgroupLimit is this process's cgroup-v2 memory ceiling, or 0 if there is none.
// memory.high is used when it is lower than memory.max, because crossing HIGH is
// what throttles -- reaching MAX only kills, and by then it is too late.
func cgroupLimit() uint64 {
	rel := ""
	if b, err := os.ReadFile("/proc/self/cgroup"); err == nil {
		for _, ln := range strings.Split(strings.TrimSpace(string(b)), "\n") {
			// cgroup v2 is the single "0::<path>" line.
			if strings.HasPrefix(ln, "0::") {
				rel = strings.TrimPrefix(ln, "0::")
			}
		}
	}
	if rel == "" {
		return 0
	}
	var lim uint64
	for _, f := range []string{"memory.high", "memory.max"} {
		v := readLimit("/sys/fs/cgroup" + rel + "/" + f)
		if v > 0 && (lim == 0 || v < lim) {
			lim = v
		}
	}
	return lim
}

// readLimit reads a cgroup limit file. "max" means unlimited, which is 0 here.
func readLimit(path string) uint64 {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0
	}
	s := strings.TrimSpace(string(b))
	if s == "max" {
		return 0
	}
	v, err := strconv.ParseUint(s, 10, 64)
	if err != nil {
		return 0
	}
	return v
}

func meminfoAvailable() uint64 {
	b, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0
	}
	for _, ln := range strings.Split(string(b), "\n") {
		if !strings.HasPrefix(ln, "MemAvailable:") {
			continue
		}
		f := strings.Fields(ln)
		if len(f) < 2 {
			return 0
		}
		kb, err := strconv.ParseUint(f[1], 10, 64)
		if err != nil {
			return 0
		}
		return kb * 1024
	}
	return 0
}
