#!/usr/bin/env python3
"""Print the busy percentage of a comma-separated CPU list, sampled over 1s.

★ IT EXISTS BECAUSE PSI IS BLIND TO THE TENANT THAT ACTUALLY RUINS A ROW.
`/proc/pressure/cpu` "some" measures time tasks spend STALLED waiting for a CPU,
so on a box with spare CPUs a large foreign job stalls almost nothing. Measured
on the 20-CPU laptop with another project's test binary at 913% and a Go compile
at 386% -- 1372% of 2000% capacity -- PSI read 7.52% and vs-llamacpp.sh's guard
said the box was quiet enough to measure. That tenant was running on psr 0,
which is the FIRST core in RULE 5's `taskset -c 0,2,4,6,8,10`.

The right question is not "is the machine stalled" but "are the cores I am about
to pin to already busy", and /proc/stat answers it directly. No process
attribution and no guessing: it sees a tenant whether pinned or unpinned,
another project's test or another session's compile.

    scripts/corebusy.py 0,2,4,6,8,10   ->  45.9

Prints nothing and exits 1 where /proc/stat is unavailable (darwin), which is
the caller's cue to skip rather than to refuse.
"""

import sys
import time


def snapshot(want):
    d = {}
    try:
        f = open("/proc/stat")
    except OSError:
        return None
    with f:
        for ln in f:
            parts = ln.split()
            if parts and parts[0] in want:
                v = [int(x) for x in parts[1:]]
                # user nice system idle iowait irq softirq ...
                idle = v[3] + (v[4] if len(v) > 4 else 0)
                d[parts[0]] = (sum(v), idle)
    return d


def main():
    if len(sys.argv) != 2:
        print("usage: corebusy.py <cpu list, e.g. 0,2,4>", file=sys.stderr)
        return 2
    want = {"cpu" + c.strip() for c in sys.argv[1].split(",") if c.strip() != ""}
    if not want:
        return 2
    a = snapshot(want)
    if not a:
        return 1
    time.sleep(1.0)
    b = snapshot(want)
    if not b:
        return 1
    # ★ ONLY CPUS PRESENT IN BOTH SNAPSHOTS COUNT. A cpu named on the command
    # line that /proc/stat does not carry is silently absent from both dicts;
    # summing over `a` alone would KeyError on it, which would read as a broken
    # probe rather than a bad argument.
    keys = [k for k in a if k in b]
    if not keys:
        return 1
    dt = sum(b[k][0] - a[k][0] for k in keys)
    di = sum(b[k][1] - a[k][1] for k in keys)
    print("%.1f" % (100.0 * (dt - di) / dt if dt > 0 else 0.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
