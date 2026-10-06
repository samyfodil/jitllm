#!/usr/bin/env bash
# One paired, interleaved rate for BLOCK PAGING: the same binary, the same
# model, the same device budget, one environment variable apart.
#
#   scripts/paging-rate.sh MODEL DEVSPEC N ROUNDS CAPMEM
#
# The knob defaults to streaming every block against placing nothing. JITLLM_RATE_KNOB
# names another and JITLLM_RATE_BASE is what BOTH arms get, which is how the
# device unpack is priced against the host packer with everything else equal:
#
#   JITLLM_RATE_KNOB=JITLLM_GPU_UNPACK JITLLM_RATE_BASE=... \
#     scripts/paging-rate.sh .../Llama-3.3-70B-Instruct-Q4_K_M.gguf cuda:0=2G 4 2 24G
#
# ★ IT IS CROSS-PROCESS AND IT SAYS SO. RULE 2 wants A/B/A/B inside ONE process,
# and this comparison cannot have that: the two arms differ in where the model IS
# -- paging on puts every block on the card, paging off leaves all but a handful
# on the host -- and placement is decided at load, by PrepLayer. There is no
# field to flip between rounds the way `verify -ab graph` flips a recording. So
# this does what scripts/vs-llamacpp.sh does for the same reason (there the other
# engine is another program): interleave HARD, ABBA rather than ABAB so drift
# cannot accumulate on one side, report the median of per-round ratios with its
# IQR, and gate at IQR/median > 0.10.
#
# ★ THE A/A SELF-CONTROL RUNS FIRST, ALWAYS (RULE 2e). One M4 A/B had a 3.5% slot
# bias that only a same-configuration-in-both-arms run exposed. Here the slot
# bias is not hypothetical: a 39.59 GiB model does not fit in this box's 31 GiB
# of RAM, so every sample re-reads its weights from NVMe and the page cache in
# slot 1 is not the page cache in slot 4. The A/A round is the only thing that
# can see that, and its ratio is printed whatever it says.
#
# ★ AND THE PAGE-IN COUNTERS TRAVEL WITH THE RATES. A rate comparison whose two
# arms are secretly the same configuration is the RULE 10c trap in its purest
# form -- JITLLM_PAGE is one character away from doing nothing -- so every sample
# carries its own page-in count and blocks-run count, and the summary line prints
# the range each arm saw.
#
# ★ THE WHOLE COMPARISON IS ONE CAPPED, EXCLUSIVE JOB (RULE 3, RULE 3d). Capping
# each sample separately would let another job take the measurement lock between
# two samples of one ABBA round, which is the failure RULE 3d was written for.
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "$0")/.." && pwd)
MODEL=${1:?usage: paging-rate.sh MODEL DEVSPEC N ROUNDS CAPMEM}
SPEC=${2:?}
N=${3:-4}
ROUNDS=${4:-3}
CAPMEM=${5:-24G}
# ★ THE SELF-CONTROL'S ROUND COUNT IS ITS OWN KNOB, because its samples are the
# EXPENSIVE arm twice. On the 70B one A/A round is four paged samples at four
# minutes each; asking for three of them costs an hour to answer a question two
# ratios already answer. It is still run FIRST and its ratio is still printed.
AAROUNDS=${JITLLM_AA_ROUNDS:-1}
# ★ THE KNOB IS A PARAMETER BECAUSE THE LEVER MOVED, NOT BECAUSE A HARNESS WANTS
# OPTIONS. The comparison that matters changes as each term stops being the row:
# with the page-in costing a host repack it was paging against no paging, and
# with the repack gone it is the DEVICE UNPACK against the host packer -- same
# binary, same placement, same eighty blocks streaming, one environment
# variable apart. Hard-coding JITLLM_PAGE would have meant a second copy of this
# file with one word changed, which is how two harnesses end up disagreeing
# about what a soak is.
#
# BASE is what BOTH arms get. It is not decoration: an unpack A/B with paging
# unset is two arms that both place two blocks of eighty and never page, which
# is RULE 10c's trap -- the ratio comes back at 1.0 and it is measuring nothing.
# The default knob, "stream", is not a variable: streaming is a placement, so
# arm 1 runs -placement '*=DEV~' on SPEC's first device and arm 0 places
# nothing. Any other knob is an environment variable set to 1 and 0.
KNOB=${JITLLM_RATE_KNOB:-stream}
read -r -a BASE <<<"${JITLLM_RATE_BASE:-}"
# The label stays "paging" for the default knob because scripts/paging-70b.sh
# reads these lines back and asserts on them; a knob that is not the default
# names itself, so a transcript cannot be mistaken for the other comparison.
LBL=paging; [ "$KNOB" = stream ] || LBL=$KNOB
if [ -z "${JITLLM_RATE_LOCKED:-}" ]; then
  export JITLLM_RATE_LOCKED=1 JITLLM_MEASURE=1
  export JITLLM_RATE_KNOB="${JITLLM_RATE_KNOB:-stream}" JITLLM_RATE_BASE="${JITLLM_RATE_BASE:-}"
  exec "$ROOT/scripts/cap" "$CAPMEM" -- "$0" "$MODEL" "$SPEC" "$N" "$ROUNDS" "$CAPMEM"
fi

BIN=${JITLLM_BIN:-$ROOT/jitllm}
PROMPT="The capital of France is"
OUT=$(mktemp -d); trap 'rm -rf "$OUT"' EXIT

# sample <paged 0|1> -> "<tok/s> <median MHz> <page-ins> <blocks run>"
sample() {
  local p=$1 f="$OUT/s" sp r m pi bl
  : > "$OUT/mhz"
  ( while :; do awk '/cpu MHz/{s+=$4;n++} END{if(n) printf "%.0f\n", s/n}' /proc/cpuinfo >> "$OUT/mhz"; sleep 0.5; done ) &
  sp=$!
  local knob=() args=()
  if [ "$KNOB" = stream ]; then
    local first=${SPEC%%,*}
    [ "$p" = 1 ] && args=(-placement "*=${first%%=*}~")
  else
    knob=("$KNOB=$p")
  fi
  env "${BASE[@]}" "${knob[@]}" taskset -c 0,2,4,6,8,10 \
      "$BIN" run -devices "$SPEC" "${args[@]}" -n "$N" -temp 0 "$MODEL" "$PROMPT" >/dev/null 2>"$f"
  kill "$sp" 2>/dev/null; wait "$sp" 2>/dev/null
  # ★ THE RATE IS COMPUTED FROM THE DURATION, NOT READ OFF THE tok/s FIELD, AND
  # THAT IS NOT A STYLE CHOICE. cmd/jitllm prints tok/s as %.2f, which on a 70B
  # streaming through a 4 GiB card reads "0.01" against the host's "0.02" -- one
  # significant figure, 50% quantisation, and the first A/A self-control taken
  # this way came back at exactly 1.0000 with an IQR of exactly 0.000 because
  # every sample had been rounded to the same two digits. RULE 10d's lesson in a
  # harness: a number that cannot vary is not a measurement. The duration beside
  # it is a Go duration rounded to the MILLISECOND, so on a 280-second sample it
  # carries seven digits, and tokens/duration is the same quantity measured
  # properly.
  r=$(sed -n 's/.*decode \([0-9]*\) tok in \([^ ]*\) .*/\1 \2/p' "$f" | tail -1 | awk '{
        n = $1; d = $2; s = 0
        # Go prints a sub-second duration as one unit (372ms) and anything above
        # a second as h/m/s parts (4m42.328s), so the sub-second units have to be
        # matched FIRST -- "372ms" otherwise parses as 372 MINUTES.
        if      (d ~ /[0-9]ms$/) { sub(/ms$/, "", d); s = d / 1e3 }
        else if (d ~ /s$/ && d !~ /[0-9](h|m)/) { sub(/s$/, "", d); s = d + 0 }
        else {
          if (match(d, /[0-9.]+h/)) { s += substr(d, RSTART, RLENGTH-1) * 3600; d = substr(d, RSTART+RLENGTH) }
          if (match(d, /[0-9.]+m/)) { s += substr(d, RSTART, RLENGTH-1) * 60;   d = substr(d, RSTART+RLENGTH) }
          if (match(d, /[0-9.]+s/)) { s += substr(d, RSTART, RLENGTH-1) }
        }
        if (s > 0) printf "%.6f", n/s }')
  m=$(sort -n "$OUT/mhz" | awk '{v[NR]=$1} END{if(NR) print v[int((NR+1)/2)]; else print 0}')
  pi=$(sed -n 's/^paging [0-9]* block slot(s), \([0-9]*\) page-in(s).*/\1/p' "$f" | tail -1)
  bl=$(sed -n 's/^device .*[0-9] block(s) placed, \([0-9][0-9]*\) run, .*/\1/p' "$f" | tail -1)
  [ -n "$r" ] || { echo "  a sample produced no rate; last lines:" >&2; tail -3 "$f" >&2; }
  echo "${r:-0} ${m:-0} ${pi:-0} ${bl:-0}"
}

# rounds <label> <A paged> <B paged> -> writes "$OUT/$label" as "ra rb fa fb pa pb"
rounds() {
  local lbl=$1 ap=$2 bp=$3 nr=$4 r a1 b1 b2 a2
  : > "$OUT/$lbl"
  for r in $(seq 1 "$nr"); do
    a1=$(sample "$ap"); b1=$(sample "$bp"); b2=$(sample "$bp"); a2=$(sample "$ap")   # ABBA
    set -- $a1; local ra1=$1 fa1=$2 pa1=$3 ba1=$4
    set -- $b1; local rb1=$1 fb1=$2 pb1=$3 bb1=$4
    set -- $b2; local rb2=$1 fb2=$2 pb2=$3 bb2=$4
    set -- $a2; local ra2=$1 fa2=$2 pa2=$3 ba2=$4
    printf "  %s round %d: A %s %s tok/s   B %s %s tok/s   [MHz %s %s %s %s]  [page-ins A %s %s B %s %s]  [blocks run A %s %s B %s %s]\n" \
      "$lbl" "$r" "$ra1" "$ra2" "$rb1" "$rb2" "$fa1" "$fa2" "$fb1" "$fb2" \
      "$pa1" "$pa2" "$pb1" "$pb2" "$ba1" "$ba2" "$bb1" "$bb2"
    printf '%s %s %s %s %s %s\n%s %s %s %s %s %s\n' \
      "$ra1" "$rb1" "$fa1" "$fb1" "$pa1" "$pb1" \
      "$ra2" "$rb2" "$fa2" "$fb2" "$pa2" "$pb2" >> "$OUT/$lbl"
  done
}

stat() { # stat <file> <label>
  python3 - "$1" "$2" <<'PY'
import sys, statistics
rows=[]
for ln in open(sys.argv[1]):
    f=ln.split()
    if len(f)==6 and float(f[0])>0 and float(f[1])>0: rows.append([float(x) for x in f])
if not rows:
    print("  %s: no samples" % sys.argv[2]); raise SystemExit(1)
# ★ DISCARD POWER-CLAMPED SAMPLES RATHER THAN AVERAGE THEM (RULE 2d): the clamp
# lands on whichever arm happens to be running, so it does not cancel in an ABBA
# and the IQR gate cannot see it.
#
# ★ AND THE FREQUENCY IS A PROXY FOR THE WRONG RESOURCE IN ONE OF THE TWO ARMS
# HERE, WHICH IS WHY THE DISCARD COUNT IS ALWAYS PRINTED. RULE 2d's clamp is a
# CPU-decode phenomenon -- six cores of AVX2 against the package power limit. A
# 70B streaming through a 4 GiB card runs the host at ~2 of 6 cores and is bound
# by NVMe and the link, so its clocks sit low because there is nothing to do, not
# because the package is clamped. Discarding on that reading can throw away a
# perfectly good sample; the count says when it happened so the reader can judge.
freqs=[f for r in rows for f in (r[2],r[3]) if f>0]
dropped=0
if freqs:
    fmed=statistics.median(freqs)
    keep=[r for r in rows if r[2]>=0.75*fmed and r[3]>=0.75*fmed]
    dropped=len(rows)-len(keep)
    if keep: rows=keep
rs=sorted(r[0]/r[1] for r in rows)
med=statistics.median(rs)
pa=[int(r[4]) for r in rows]; pb=[int(r[5]) for r in rows]
note="" if not dropped else "  [%d discarded: clocks below 0.75 of the session median]"%dropped
# ★ AN IQR OVER FEWER THAN FOUR RATIOS IS NOT A DISPERSION, AND PRINTING ONE IS
# WORSE THAN PRINTING NOTHING. With n=2 the quartiles are the two samples; with
# n=1 they are the SAME sample, so the gate reports "IQR/median 0.000" -- a
# perfect score awarded for having one number. That is RULE 10a arriving in the
# harness: a gate that cannot fail is not a gate. Below four, print the ratios.
if len(rs) < 4:
    print("  %s ratio %.4f  (n=%d -- too few for a dispersion; the ratios are %s)%s"
          % (sys.argv[2], med, len(rs), ", ".join("%.4f" % r for r in rs), note))
else:
    iqr=rs[len(rs)*3//4]-rs[len(rs)//4]
    gate="" if iqr/med<=0.10 else "   REJECTED: IQR/median %.3f > 0.10"%(iqr/med)
    print("  %s ratio %.4f  (IQR/median %.3f, n=%d)%s%s" % (sys.argv[2], med, iqr/med, len(rs), gate, note))
    print("  %s ratios: %s" % (sys.argv[2], ", ".join("%.4f" % r for r in rs)))
print("  %s page-ins: A %d..%d   B %d..%d" % (sys.argv[2], min(pa), max(pa), min(pb), max(pb)))
PY
}

echo "  model $(basename "$MODEL")   devices $SPEC   n=$N   rounds=$ROUNDS x 4 samples (ABBA)"
echo "  knob  $KNOB=1 against $KNOB=0${JITLLM_RATE_BASE:+   both arms: $JITLLM_RATE_BASE}"
# ★ ONE SOAK RUN PER ARM, NOT SIX. On the CPU board six are needed to reach
# thermal steady state; this comparison is bound by NVMe and the link, the
# package runs at ~2 of 6 cores, and six soaks of a 70B cost twelve minutes to
# remove an effect the A/A control measures directly. What the soak buys HERE is
# the JIT tune cache and the first faults, and one run per arm buys both.
echo "  soaking (one run per arm: the tune cache and the first faults)..."
sample 1 >/dev/null; sample 0 >/dev/null

echo "  A/A self-control FIRST -- both arms $KNOB=1, $AAROUNDS round(s):"
rounds "AA" 1 1 "$AAROUNDS"
stat "$OUT/AA" "self-control"
echo "  A/B -- A = $KNOB=1, B = $KNOB=0:"
rounds "AB" 1 0 "$ROUNDS"
stat "$OUT/AB" "$LBL"
