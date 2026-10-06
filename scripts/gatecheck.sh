#!/usr/bin/env bash
# Run the BOARD'S OWN GATES against violations. RULE 10: a gate that has never
# fired is not a gate, and every gate in vs-llamacpp.sh and clockadj.py decides
# whether a perf row is quotable -- so an unverified one silently promotes noise
# into a standing.
#
# ★ IT TESTS THE GATES, NOT THE ENGINE, SO IT NEEDS NO QUIET BOX. That is the
# point: the rows those gates guard need a rested machine and cannot be taken on
# demand, and the gates themselves are pure functions of a sample list. This runs
# in under a second on a box at 100 C with three other tenants.
#
# ★ AND IT FOUND ONE. The clamp gate treated a sample whose frequency read came
# back 0 as POWER-CLAMPED -- dropping it and counting it in "N sample(s)
# discarded: CPU power-clamped". That count is what gets read as "the box is
# thermally sick", so a failed read was being promoted into thermal evidence.
# Case 8 is that violation and it is why this file exists.
#
#   scripts/gatecheck.sh
set -uo pipefail
cd "$(dirname "$0")/.."
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
fail=0; ran=0

# The gate is extracted from the script rather than copied, so it cannot drift.
# Anchored on the block's OWN marker, not a line number: the first version used
# /python3 -c/ and silently extracted an earlier, unrelated block, which failed
# loudly only because the result was not valid Python. A quieter mismatch would
# have tested the wrong code and passed.
awk '/CLAMP_FRAC=.*python3 -c/{f=1;next} f&&/^'"'"'$/{exit} f{print}' scripts/vs-llamacpp.sh > "$TMP/gate.py"
[ -s "$TMP/gate.py" ] || { echo "FATAL: could not extract the gate from vs-llamacpp.sh"; exit 1; }

# want <name> <expected substring> <samples on stdin>
want() {
  local name=$1 expect=$2 out
  ran=$((ran+1))
  out=$(CLAMP_FRAC=0.75 COHERE=1.05 python3 "$TMP/gate.py" 2>&1)
  if [[ "$out" == *"$expect"* ]]; then
    printf '  ok    %s\n' "$name"
  else
    printf '  FAIL  %s\n        want substring: %s\n        got: %s\n' "$name" "$expect" "$out"
    fail=$((fail+1))
  fi
}

rep() { local n=$1; shift; for _ in $(seq "$n"); do echo "$@"; done; }

echo "vs-llamacpp.sh sample gates:"
want "clean row passes"            "n=8"                          <<< "$(rep 8 52.0 50.0 1800 1800)"
want "dispersed row REJECTED"      "REJECTED: IQR/median"         <<< "$(printf '%s\n' "80 50 1800 1800" "30 50 1800 1800" "75 50 1800 1800" "33 50 1800 1800" "70 50 1800 1800" "35 50 1800 1800" "65 50 1800 1800" "40 50 1800 1800")"
want "thin row REJECTED"           "only 4 pair(s) survived"      <<< "$(rep 4 52.0 50.0 1800 1800)"
want "power-clamped discarded"     "discarded: CPU power-clamped" <<< "$(rep 6 52.0 50.0 1800 1800; rep 3 20.0 50.0 900 900)"
want "incoherent pairs discarded"  "arms ran at different clocks" <<< "$(rep 6 52.0 50.0 1800 1800; rep 3 52.0 50.0 1800 2700)"
want "all-incoherent refused"      "cannot produce a row"         <<< "$(rep 8 52.0 50.0 1200 2400)"
# The two that motivated the fix.
want "darwin no-freq is a no-op"   "n=8"                          <<< "$(rep 8 52.0 50.0 0 0)"
want "no-freq KEPT, not clamped"   "KEPT UNJUDGED"                <<< "$(rep 6 52.0 50.0 1800 1800; rep 3 52.0 50.0 0 0)"
want "no-freq is not counted clamped" "n=9"                       <<< "$(rep 6 52.0 50.0 1800 1800; rep 3 52.0 50.0 0 0)"

# clockadj.py: rounds carry TWO pairs each, printed as [MHz f1 f2 g1 g2].
rounds() { python3 -c '
import sys
pts=eval(sys.argv[1])
rows=[(50*y,50.0,x,x) for x,y in pts]
for i in range(0,len(rows)-1,2):
    p,q=rows[i],rows[i+1]
    print("  round %d: jitllm %.2f %.2f   llama.cpp %.2f %.2f   [MHz %d %d %d %d]"%(
        i//2+1,p[0],q[0],p[1],q[1],p[2],q[2],p[3],q[3]))' "$1"; }

adj() {
  local name=$1 expect=$2 pts=$3; shift 3
  local out; ran=$((ran+1))
  out=$(rounds "$pts" | python3 scripts/clockadj.py "$@" 2>&1)
  if [[ "$out" == *"$expect"* ]]; then printf '  ok    %s\n' "$name"
  else printf '  FAIL  %s\n        want: %s\n        got: %s\n' "$name" "$expect" "$out"; fail=$((fail+1)); fi
}

echo "clockadj.py refusals:"
FLAT='[(1600+(i*100)%800, 1.04+(0.02 if i%2 else -0.02)) for i in range(16)]'
TREND='[(1600+(i*100)%800, 0.9+(1600+(i*100)%800)/10000.0) for i in range(16)]'
# High r driven by two outliers; subtracting the line they define spreads the
# tight cluster. This is the shape that printed "GATES ... (was 0.013)" over a
# spread that had gone 0.013 -> 0.064.
OUTLIER='[(1700+20*i,1.04) for i in range(14)]+[(2300,1.30),(2300,1.30)]'
adj "no trend refuses to adjust"   "NO TREND"                  "$FLAT"
adj "real trend fits and tightens" "GATES once the clock term" "$TREND"
adj "refuses to extrapolate"       "refusing to extrapolate"   "$TREND" -ref 3500
adj "thin input refuses to fit"    "nothing to fit"            '[(1700,1.04),(1800,1.05),(1900,1.06),(2000,1.07)]'
adj "worse-after is reported"      "THE ADJUSTMENT MADE IT WORSE" "$OUTLIER"


# ★ THE CORE-BUSY PROBE IS A LIVE SYSTEM GATE, NOT A SAMPLE GATE, SO IT IS
# TESTED BY PUTTING REAL LOAD ON A REAL CPU. It exists because PSI is blind to a
# tenant that fits in the box's spare CPUs -- measured on this laptop at 1372%
# of 2000% foreign load with PSI reading 7.52% and the old guard saying "quiet".
if [ -r /proc/stat ] && [ "$(uname -s)" = "Linux" ]; then
  echo "core-busy probe:"
  # The probe is a FILE that vs-llamacpp.sh calls, not a here-doc it embeds, so
  # there is nothing to extract and nothing to drift. The first version of this
  # check sed'd the code out of the shell script and silently produced an empty
  # file -- a test that could only ever fail, which is RULE 10 from the useless
  # end rather than the dangerous one.
  C=$(( $(nproc) - 1 ))   # last CPU: least likely to be the one running this
  ran=$((ran+1))
  taskset -c "$C" bash -c 'while :; do :; done' & spin=$!
  sleep 0.2
  load=$(scripts/corebusy.py "$C" 2>/dev/null)
  kill "$spin" 2>/dev/null; wait "$spin" 2>/dev/null
  if [ -n "$load" ] && [ "$(echo "$load > 50" | bc)" = "1" ]; then
    printf '  ok    sees a pinned spinner on cpu%s (%s%% busy)\n' "$C" "$load"
  else
    printf '  FAIL  probe missed a pinned spinner on cpu%s: read %s%%\n' "$C" "${load:-<none>}"
    fail=$((fail+1))
  fi
  # And it must not cry wolf: a CPU nobody is on has to read low. Sampled after
  # the spinner is reaped, on the same CPU, so the two differ only in the load.
  ran=$((ran+1))
  quiet=$(scripts/corebusy.py "$C" 2>/dev/null)
  if [ -n "$quiet" ] && [ "$(echo "$quiet < 50" | bc)" = "1" ]; then
    printf '  ok    reads cpu%s low once the spinner is gone (%s%%)\n' "$C" "$quiet"
  else
    printf '  FAIL  probe still reads cpu%s busy with nothing on it: %s%%\n' "$C" "${quiet:-<none>}"
    fail=$((fail+1))
  fi
fi


# ★ THE CONTAINER-VERSION CHECK, WHICH REPLACED AN mtime PROXY THAT WAS WRONG
# WHENEVER ANOTHER BINARY WROTE THE FILE. A v6 tree converting into the shared
# models directory leaves a v6 container with a NEWER mtime than the v15 binary,
# so the old check said "fresh" and the run died mid-soak with an empty error.
echo "container-version probe:"
mkhdr() { printf 'JITLLM\0\0' > "$1"; printf "$(printf '\\x%02x\\x00\\x00\\x00' "$2")" >> "$1"; }
ver_of() { od -A n -t u4 -j 8 -N 4 "$1" 2>/dev/null | tr -d ' '; }
for v in 6 15; do
  mkhdr "$TMP/c$v.jlm" "$v"
  got=$(ver_of "$TMP/c$v.jlm")
  ran=$((ran+1))
  if [ "$got" = "$v" ]; then printf '  ok    reads a v%s header as %s\n' "$v" "$got"
  else printf '  FAIL  v%s header read as %s\n' "$v" "${got:-<none>}"; fail=$((fail+1)); fi
done
# And the script must actually ACT on a mismatch. Compare against the source
# constant the same way vs-llamacpp.sh does.
want=$(sed -n 's/^const Version = \([0-9]*\).*/\1/p' format/jlm/format.go 2>/dev/null)
ran=$((ran+1))
if [ -n "$want" ] && [ "$(ver_of "$TMP/c6.jlm")" != "$want" ]; then
  printf '  ok    a v6 container is stale against format/jlm/format.go (v%s)\n' "$want"
else
  printf '  FAIL  could not read format/jlm/format.go Version, or v6 compared equal to it (got %s)\n' "${want:-<none>}"
  fail=$((fail+1))
fi

echo
if [ "$fail" = 0 ]; then echo "all $ran gate(s) fired correctly"; else echo "$fail of $ran FAILED"; fi
exit $((fail > 0))
