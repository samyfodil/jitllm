#!/usr/bin/env python3
"""Pick nodes out of a llama-mtmd-debug -p encode dump into a JSON golden.

    python3 scripts/mtmddump.py DUMP OUT.json NODE...

llama-mtmd-debug prints, per graph node, the first and last three rows of its
first two axes -- three leading and three trailing values of each -- and the
sum of the whole tensor. That is what is kept: for each node, {"rows": [[first
three values, last three values] for rows 0,1,2,-3,-2,-1], "sum", "shape"}.
A node name printed twice (a name ggml reuses across layers, like norm_w)
keeps its LAST printing; name nodes that are unique.
"""
import json, re, sys

HEAD = re.compile(r"common_debug_cb_eval:\s+(.*?) = \((f32|f16)\).*= \{(\d+), (\d+), (\d+), (\d+)\}")
ROW = re.compile(r"\[\s*([-\d.e+]+),\s*([-\d.e+]+),\s*([-\d.e+]+),\s*\.\.\.,\s*([-\d.e+]+),\s*([-\d.e+]+),\s*([-\d.e+]+)\s*\]")
SUM = re.compile(r"sum = ([-\d.e+naninf]+)")


def main(dump, out, names):
    nodes, cur = {}, None
    for line in open(dump, errors="replace"):
        m = HEAD.search(line)
        if m:
            cur = {"name": m.group(1).strip(), "shape": [int(m.group(i)) for i in range(3, 7)], "rows": []}
            continue
        if cur is None:
            continue
        r = ROW.search(line)
        if r:
            cur["rows"].append([float(x) for x in r.groups()])
            continue
        s = SUM.search(line)
        if s:
            cur["sum"] = float(s.group(1))
            nodes[cur["name"]] = cur
            cur = None
    keep = {}
    for n in names:
        if n not in nodes:
            sys.exit(f"no node {n!r} in {dump}")
        keep[n] = {k: nodes[n][k] for k in ("shape", "rows", "sum")}
    json.dump(keep, open(out, "w"), indent=1)
    print("wrote", out, ", ".join(f"{n} sum {keep[n]['sum']}" for n in names))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
