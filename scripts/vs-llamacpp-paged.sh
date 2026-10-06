#!/usr/bin/env bash
# jitllm paging a CONVERTED model against llama.cpp on the same model and box.
#
# ★ THE COMPARISON IS NOT SYMMETRIC AND THAT IS THE POINT, SO IT IS STATED WITH
# THE ROW. On a card too small for the model llama.cpp does not degrade -- it
# offloads the largest -ngl that fits and runs the rest on the CPU, and above
# that it REFUSES TO LOAD. jitllm pages every block through the card. So this is
# not "same workload, who is faster"; it is "what can each engine do with this
# machine", and the -ngl and the block count both belong in the output.
#
# RULE 1: quote the backend with the number. RULE 2: A/B interleaved, median of
# per-round ratios, IQR gate, PSI gate, clamped samples discarded.
set -u
cd "$(dirname "$0")/.."
MODEL=${1:?usage: vs-llamacpp-paged.sh <model.gguf> [vram] [n] [rounds]}
VRAM=${2:-2G}
N=${3:-24}
ROUNDS=${4:-3}
PROMPT="Once upon a time in a distant land there"
JL=${JITLLM_LCPP:?set JITLLM_LCPP to a llama.cpp build directory}
CONV="${MODEL%.gguf}.jlm"
CORES=0,2,4,6,8,10

psi() { awk '/some/{sub("avg10=","",$2);print $2}' /proc/pressure/cpu; }
mhz() { awk '/cpu MHz/{if($4>m)m=$4} END{printf "%.0f", m}' /proc/cpuinfo; }

say() { printf '%s\n' "$*"; }

[ -f "$CONV" ] || { say ">> converting $MODEL"; ./scripts/cap 24G -- ./jitllm convert "$MODEL" "$CONV" || exit 1; }

# ★ llama.cpp's -ngl HAS TO BE FOUND BY HAND, which is why this searches rather
# than assuming: it refuses to load above what fits, so the honest arm is the
# largest value that does. vs-llamacpp.sh says the same thing and for the same
# reason.
NGL=0
for n in 99 64 48 32 24 16 12 10 8 6 4 2; do
  if LD_LIBRARY_PATH=$JL ./scripts/cap 26G -- taskset -c $CORES \
       "$JL/llama-bench" -m "$MODEL" -ngl $n -p 0 -n 2 -r 1 >/dev/null 2>&1; then
    NGL=$n; break
  fi
done
say ">> llama.cpp loads at -ngl $NGL"

jit() { ./scripts/cap 26G -- taskset -c $CORES \
        ./jitllm run -devices "cuda:0=$VRAM" -placement '*=cuda:0~' -n "$N" -temp 0 "$CONV" "$PROMPT" 2>&1 \
        | sed -n 's|.*decode [0-9]* tok in [^(]*(\([0-9.]*\) tok/s).*|\1|p'; }
lcpp() { LD_LIBRARY_PATH=$JL ./scripts/cap 26G -- taskset -c $CORES \
         "$JL/llama-bench" -m "$MODEL" -ngl "$NGL" -p 0 -n "$N" -r 1 2>/dev/null \
         | awk -F'|' '/tg/{gsub(/ /,"",$(NF-1)); split($(NF-1),a,"±"); print a[1]}'; }

say ">> warm-up (discarded)"; jit >/dev/null; lcpp >/dev/null
say ""
say "round   jitllm(paged,ptx)   llama.cpp(cuda,-ngl $NGL)   ratio   MHz   PSI"
RATIOS=()
for r in $(seq 1 "$ROUNDS"); do
  # ABBA within the round so a drift common to one ordering cannot survive.
  a1=$(jit); b1=$(lcpp); b2=$(lcpp); a2=$(jit)
  m=$(mhz); p=$(psi)
  ja=$(python3 -c "print((${a1:-0}+${a2:-0})/2)")
  lb=$(python3 -c "print((${b1:-0}+${b2:-0})/2)")
  ra=$(python3 -c "print(${ja}/${lb} if ${lb}>0 else 0)")
  printf "%5s   %17s   %24s   %5.3f   %5s %5s\n" "$r" "$ja" "$lb" "$ra" "$m" "$p"
  RATIOS+=("$ra")
done
say ""
python3 - "${RATIOS[@]}" <<'PY'
import sys, statistics
r = sorted(float(x) for x in sys.argv[1:] if float(x) > 0)
if not r:
    print("no usable rounds"); sys.exit(1)
med = statistics.median(r)
q1, q3 = (statistics.median(r[:len(r)//2]), statistics.median(r[(len(r)+1)//2:])) if len(r) > 2 else (r[0], r[-1])
iqr = q3 - q1
print(f"jitllm / llama.cpp = {med:.4f}   IQR/median {iqr/med:.3f}   n={len(r)}")
if iqr/med > 0.10:
    print("REJECTED: IQR/median > 0.10 (RULE 2) -- do not quote this number")
elif med > 1:
    print(f"jitllm is {med:.2f}x llama.cpp on this model and box")
else:
    print(f"jitllm is {1/med:.2f}x SLOWER than llama.cpp on this model and box")
PY
