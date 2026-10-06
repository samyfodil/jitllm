#!/usr/bin/env bash
# Prove that a device SMALLER THAN THE MODEL runs EVERY block of it, by swapping
# blocks through the slots its budget leaves -- and that swapping cannot move a
# token id.
#
# ★ THE MEASUREMENT THIS EXISTS FOR. Llama-3.3-70B is 39.59 GiB of weights and
# this box's RTX 3050 Ti has 4, so the tier placed 2 of 80 blocks and the card
# idled through 97.5% of the model that needed it: a block that did not fit was
# declined and fell to the host PERMANENTLY. Streamed (-placement '*=DEV~') a decline becomes
# a swap -- the same PrepLayer seam, the same per-block decision, but no longer
# permanent (RULE 8a: model/ still owns the graph and nothing below it learns
# what a transformer is).
#
# ★ THE TRAP EVERY ASSERTION HERE IS BUILT AROUND IS A GATE THAT PASSES BECAUSE
# NOTHING PAGED. "22/22 blocks placed" is also true of a tier that accepted every
# block and quietly kept six; "the text is English" is true of a run that declined
# everything and used the CPU. So the paged arms assert the PAGE-IN COUNTER, and
# the fitting arm asserts the PAGE-OUT counter is zero -- the two numbers that are
# wrong in exactly the state that looks right.
#
# ★ WHAT IS ASSERTED, AND THE --violate NAME THAT BREAKS EXACTLY ONE (a gate that
# has never been seen to fail is not a gate, RULE 10b):
#
#   every-block  the paged run places ALL of the model's blocks on the device,
#                where the same budget without paging places floor(budget/block).
#                violation: --violate no-paging
#   paged        that run's page-in counter is NON-ZERO, so blocks really left the
#                card and came back. Without it "every block placed" proves
#                nothing.
#                violation: --violate no-paging (the same arm; it reads 0)
#   ids          the paged run's token ids equal `-devices cpu`'s. THE
#                LOAD-BEARING ONE: a pager that uploaded a block's weights shifted
#                by a word still runs, still reports every counter above, and
#                answers fluent nonsense.
#                violation: --violate ids (compare against a doctored reference)
#   fits         a budget that holds the model pages ZERO times and keeps its
#                rate. RULE 6: what fitting buys is that eviction never FIRES,
#                which is a different claim from the machinery being absent.
#                violation: --violate fits-tiny (shrink the budget; it must page)
#   shrink       a budget SHRUNK MID-GENERATION on a model that fits demotes
#                blocks, leaves the placement alone and does not move an id. This
#                is the second-model case and the reason the machinery is primed.
#                violation: --violate no-shrink (do not shrink; nothing demotes)
#   arena        the paged run PACKS EACH BLOCK ONCE: packing is 1.01 GB/s against
#                an upload's 3.19, so a page-in that repacks costs 3.1x the
#                transfer it was scheduling.
#                violation: --violate no-arena (JITLLM_ARENA=0)
#   import       on a device whose memory IS the host's, the same budget holds
#                EVERY block while resident bytes stay at ZERO: the driver is
#                handed the packed arena's own pointer, so the weights are never
#                transferred and never charged. Skipped, loudly, on a host with
#                no unified device.
#                violation: --violate no-import (JITLLM_ARENA=0 -- with nothing
#                retained there is nothing to wrap, and the same device falls
#                back to copying and to holding a fraction of the model)
#
# ★ WHAT IT DOES NOT ASSERT, AND SAYING SO IS THE POINT. Nothing here is a RATE
# claim. Paging moves a block's packed bytes over a link measured at 3.19 GB/s
# where the host reads the same block from DRAM at ~37, so on a model that fits in
# HOST memory paging LOSES and is expected to -- which is why only a placement
# that asks for it streams a block (-placement '*=DEV~'). The rates printed below are single unpaired runs for
# orientation; a claim needs RULE 2's paired interleave and scripts/vs-llamacpp.sh.
#
# usage:  scripts/paging.sh [-m MODEL] [-d DEVSPEC] [-n TOKENS] [--violate NAME]
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/.." && pwd)
MODEL=${JITLLM_PAGE_MODEL:-${JITLLM_MODELS:?set JITLLM_MODELS to the model directory}/tinyllama-1.1b-q3_K_M.gguf}
NTOK=8
PROMPT="The capital of France is"
VIOLATE=""
DEV=""
# SMALL is chosen so the model cannot fit and the slot count still clears the
# floor of two: a budget below two slots is a DECLINE by design, not a bug.
SMALL=200000000
BIG=3000000000
SHRINK=250M
while [ $# -gt 0 ]; do
  case "$1" in
    -m) MODEL=$2; shift 2 ;;
    -d) DEV=$2; shift 2 ;;
    -n) NTOK=$2; shift 2 ;;
    -p) PROMPT=$2; shift 2 ;;
    --violate) VIOLATE=$2; shift 2 ;;
    -h|--help) sed -n '1,/^# usage:/p' "$0"; exit 0 ;;
    *) echo "paging: unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -r "$MODEL" ] || { echo "paging: no model at $MODEL (-m MODEL, or JITLLM_PAGE_MODEL)" >&2; exit 2; }

CAP="$ROOT/scripts/cap 8G -- taskset -c 0,2,4,6,8,10"
OUT=$(mktemp -d); trap 'rm -rf "$OUT"' EXIT
BIN=$ROOT/jitllm
fails=0
note() { printf '%s\n' "$*"; }
ok()   { printf '  PASS  %s\n' "$*"; }
bad()  { printf '  FAIL  %s\n' "$*"; fails=$((fails+1)); }

$ROOT/scripts/cap 8G -- go build -o "$BIN" ./cmd/jitllm || exit 1

# ★ THE DEVICE IS DISCOVERED, NOT HARDCODED: `cuda:0` is a card on this laptop
# and nothing on the next one. Paging is a property of the tier, not of a backend,
# so any real device will do -- the software rasteriser will not.
if [ -z "$DEV" ]; then
  $CAP "$BIN" hardware >"$OUT/hw" 2>&1 || { cat "$OUT/hw"; exit 1; }
  ORD=$(awk '/ ptx +#[0-9]+/{ match($0, /#[0-9]+/); print substr($0, RSTART+1, RLENGTH-1); exit }' "$OUT/hw")
  if [ -n "${ORD:-}" ]; then DEV="cuda:$ORD"; else
    DEV=$(awk '/^ +vulkan:[0-9]+/ && !/software rasteriser/ { match($0, /vulkan:[0-9]+/); print substr($0, RSTART, RLENGTH); exit }' "$OUT/hw")
  fi
fi
[ -n "$DEV" ] || { note "paging: no real device on this host; jitllm hardware says:"; sed -n '/^gpu/,$p' "$OUT/hw"; exit 2; }

# ★ AND A SECOND DEVICE, THE UNIFIED ONE, BECAUSE IT IS A DIFFERENT MECHANISM
# AND NOT A FASTER ONE. On a device whose heap is the host's memory a page-in is
# a descriptor update rather than a transfer, so the assertions below are about
# bytes NOT moving -- which is the opposite of every assertion above and cannot
# be checked on the discrete card that runs them.
[ -f "$OUT/hw" ] || $CAP "$BIN" hardware >"$OUT/hw" 2>&1
UDEV=$(awk '/^ +vulkan:[0-9]+/ && /memory SHARED with the host/ { match($0, /vulkan:[0-9]+/); print substr($0, RSTART, RLENGTH); exit }' "$OUT/hw")
if [ -z "${UDEV:-}" ] && grep -qE '^(gpu +)?msl ' "$OUT/hw"; then UDEV=gpu; fi

note "host      $(uname -sm)"
note "device    $DEV"
note "model     $MODEL"
note "prompt    \"$PROMPT\"   -n $NTOK   greedy (-temp 0)"
note "budgets   paged $SMALL, fits $BIG, runtime shrink $SHRINK"
note ""

# Streaming is a placement: every block streams through $DEV (-placement
# '*=DEV~'). There is no device-wide switch to set instead.
PAGE_ENV=()
PAGE_ARGS=(-placement "*=$DEV~")
SHRINK_ENV=(JITLLM_PAGE_SHRINK=$SHRINK)
ARENA_ENV=()
IMPORT_ENV=()
case "$VIOLATE" in
  no-paging) PAGE_ARGS=() ;;
  fits-tiny) BIG=$SMALL ;;
  # ★ THE VIOLATION IS AN ABSENT SHRINK, NOT A VARIABLE NOBODY READS. This set
  # JITLLM_NO_SHRINK=1, which NO Go code has ever read -- so the arm ran the
  # same configuration as the normal one AND silently dropped the real
  # JITLLM_PAGE_SHRINK with it. The banner below promises "one assertion MUST
  # fail"; this one could not. Dropping the setting is the honest violation: the
  # budget never falls, nothing is demoted, and the page-out assertion fires.
  no-shrink) SHRINK_ENV=() ;;
  no-arena)  ARENA_ENV=(JITLLM_ARENA=0) ;;
  no-import) IMPORT_ENV=(JITLLM_ARENA=0) ;;
  ids|"")    ;;
  *) echo "paging: unknown violation $VIOLATE (ids|no-paging|fits-tiny|no-shrink|no-arena|no-import)" >&2; exit 2 ;;
esac
[ -n "$VIOLATE" ] && note "★ VIOLATION MODE \"$VIOLATE\": one assertion below MUST fail, or the gate is not a gate."

run() { # run TAG VRAM env...
  local tag=$1 vram=$2; shift 2
  $CAP env "$@" "$BIN" run -devices "$DEV=$vram" "${PAGE_ARGS[@]}" -n "$NTOK" -temp 0 "$MODEL" "$PROMPT" \
      >"$OUT/$tag.stdout" 2>"$OUT/$tag"
}
ids()      { sed -n 's/^ids \(\[.*\]\)$/\1/p' "$OUT/$1" | tail -1; }
seam()     { sed -n 's/^device .*: \([0-9][0-9]*\)\/\([0-9][0-9]*\) blocks.*/\1 \2/p' "$OUT/$1" | tail -1; }
pline()    { grep -m1 '^paging ' "$OUT/$1"; }
pagefield() { sed -n "s/^paging [0-9]* block slot(s), \([0-9]*\) page-in(s), \([0-9]*\) page-out(s).*/\\$2/p" "$OUT/$1" | tail -1; }
pagein()   { pagefield "$1" 1; }
pageout()  { pagefield "$1" 2; }
repacks()  { sed -n 's/^paging .*and \([0-9][0-9]*\) repacked.*/\1/p' "$OUT/$1" | tail -1; }
rate()     { sed -n 's/.*decode [0-9]* tok in [^(]*(\([0-9.]*\) tok\/s).*/\1/p' "$OUT/$1" | tail -1; }

# ------------------------------------------------------------------- reference
note "== reference: the host alone =="
$CAP env NOTHING=1 "$BIN" run -devices cpu -n "$NTOK" -temp 0 "$MODEL" "$PROMPT" \
    >"$OUT/cpu.stdout" 2>"$OUT/cpu" || { tail -3 "$OUT/cpu"; exit 1; }
REF=$(ids cpu)
[ -n "$REF" ] || { echo "paging: the cpu arm printed no ids"; cat "$OUT/cpu"; exit 1; }
note "  -devices cpu   $REF   ($(rate cpu) tok/s)"
if [ "$VIOLATE" = "ids" ]; then
  REF=$(printf '%s' "$REF" | sed 's/\[1 /[999 /')
  note "  ★ doctored reference: $REF"
fi
note ""

# ------------------------------------------------- 1/2/3: a model that does NOT fit
note "== a device smaller than the model, paging on =="
run paged "$SMALL" "${PAGE_ENV[@]}" "${ARENA_ENV[@]}" || { tail -3 "$OUT/paged"; bad "the paged run did not finish"; }
run today "$SMALL" NOTHING=1 || { tail -3 "$OUT/today"; bad "the no-paging control did not finish"; }
sed -n 's/^device /  device /p' "$OUT/paged"
note "  $(pline paged)"
note "  ids $(ids paged)   ($(rate paged) tok/s; the same budget with paging off: $(rate today) tok/s)"

read -r pl tot <<<"$(seam paged)"
read -r bl _   <<<"$(seam today)"
if [ "${pl:-0}" = "${tot:-0}" ] && [ "${tot:-0}" -gt 0 ] 2>/dev/null; then
  ok "every-block: the device holds ALL $tot blocks (paging off, the same budget holds ${bl:-0})"
else
  bad "every-block: the device holds ${pl:-0} of ${tot:-?} blocks (paging off holds ${bl:-0})"
fi
[ "$(pagein paged)" -gt 0 ] 2>/dev/null \
  && ok "paged: $(pagein paged) page-in(s) and $(pageout paged) page-out(s) -- blocks really left the card and came back" \
  || bad "paged: $(pagein paged) page-ins. NOTHING PAGED, so the assertion above proves nothing"
[ "$(ids paged)" = "$REF" ] && ok "ids: the paged run is identical to the host-only run" \
                            || bad "ids: the paged run differs from the host-only run"
[ "${bl:-0}" -lt "${tot:-1}" ] 2>/dev/null \
  && ok "control: without paging the same budget holds only ${bl:-0}/${tot:-?} -- the budget is what changed, not the model" \
  || bad "control: without paging the device still holds ${bl:-0}/${tot:-?}; this budget does not exercise the pager"
note ""

# --------------------------------------------------------- 4: a model that FITS
note "== a budget that holds the model =="
run fits "$BIG" "${PAGE_ENV[@]}" "${ARENA_ENV[@]}" || { tail -3 "$OUT/fits"; bad "the fitting run did not finish"; }
sed -n 's/^device /  device /p' "$OUT/fits"
note "  $(pline fits)"
note "  ids $(ids fits)   ($(rate fits) tok/s)"
read -r fl ftot <<<"$(seam fits)"
{ [ "$(pageout fits)" = "0" ] && [ "$(pagein fits)" = "0" ]; } \
  && ok "fits: ZERO page-ins and ZERO page-outs with paging ON -- the machinery is present and idle" \
  || bad "fits: a model that fits paged $(pagein fits) in and $(pageout fits) out"
[ "${fl:-0}" = "${ftot:-0}" ] && ok "fits: all ${ftot:-?} blocks resident" || bad "fits: ${fl:-0} of ${ftot:-?} blocks"
[ "$(ids fits)" = "$REF" ] && ok "fits: ids identical to the host-only run" || bad "fits: ids differ"
note ""

# ------------------------------------------- 5: the pool shrunk at runtime
note "== the budget SHRUNK mid-generation on a model that fits =="
run shrink "$BIG" "${PAGE_ENV[@]}" "${SHRINK_ENV[@]}" "${ARENA_ENV[@]}" || { tail -3 "$OUT/shrink"; bad "the shrink run did not finish"; }
sed -n 's/^device /  device /p' "$OUT/shrink"
grep -m1 '★ shrinking' "$OUT/shrink" | sed 's/^/  /'
note "  $(pline shrink)"
note "  ids $(ids shrink)   ($(rate shrink) tok/s)"
read -r sl stot <<<"$(seam shrink)"
[ "$(pageout shrink)" -gt 0 ] 2>/dev/null \
  && ok "shrink: $(pageout shrink) page-out(s) once the budget fell -- the card really gave memory back" \
  || bad "shrink: nothing demoted, so the pool did not shrink"
[ "${sl:-0}" = "${stot:-0}" ] && [ "${stot:-0}" = "${ftot:-0}" ] \
  && ok "shrink: the placement did NOT move -- still ${sl}/${stot} blocks on the device" \
  || bad "shrink: the placement moved, ${fl:-?}/${ftot:-?} -> ${sl:-?}/${stot:-?}"
[ "$(ids shrink)" = "$REF" ] && ok "shrink: not one token id moved" || bad "shrink: the ids moved across a demotion"
note ""

# ------------------------------------------------------------- 6: the arena
note "== the pack runs once per block, not once per page-in =="
note "  $(pline paged)"
[ "$(repacks paged)" = "0" ] \
  && ok "arena: 0 repacks over $(pagein paged) page-ins -- every one was a DMA from the packed arena" \
  || bad "arena: $(repacks paged) repacks over $(pagein paged) page-ins, at 1.01 GB/s against an upload's 3.19"
note ""

# ------------------------------------------------- 7: the unified device
note "== a unified device does not copy: the page-in is a descriptor update =="
if [ -z "${UDEV:-}" ]; then
  # RULE 11: an absent capability is a task, not a silent pass. This says which.
  note "  SKIPPED: no device on this host reports its memory as shared with the host."
  note "  (jitllm hardware names one as 'memory SHARED with the host'; an Apple"
  note "   machine is always one, an integrated GPU usually is, a discrete card is not.)"
else
  # The unified arm names its OWN device, so it cannot go through run()'s $DEV.
  UARGS=(); [ ${#PAGE_ARGS[@]} -gt 0 ] && UARGS=(-placement "*=$UDEV~")
  $CAP env "${PAGE_ENV[@]}" "${IMPORT_ENV[@]}" "$BIN" run -devices "$UDEV=$SMALL" "${UARGS[@]}" \
      -n "$NTOK" -temp 0 "$MODEL" "$PROMPT" >"$OUT/unified.stdout" 2>"$OUT/unified" \
      || { tail -3 "$OUT/unified"; bad "the unified run did not finish"; }
  sed -n 's/^device /  device /p' "$OUT/unified"
  note "  $(pline unified)"
  grep -m1 '^imports ' "$OUT/unified" | sed 's/^/  /'
  note "  ids $(ids unified)   ($(rate unified) tok/s)"
  IMP=$(sed -n 's/^imports \([0-9][0-9]*\) tensor.*/\1/p' "$OUT/unified" | tail -1)
  RES=$(sed -n 's/^device .*blocks[^,]*, \([0-9.]*\) GiB resident.*/\1/p' "$OUT/unified" | tail -1)
  read -r ul utot <<<"$(seam unified)"
  [ "${IMP:-0}" -gt 0 ] 2>/dev/null \
    && ok "import: $IMP tensor(s) wrapped in place -- the driver was handed the arena's own pointer" \
    || bad "import: NOTHING was wrapped, so every assertion here is about a run that copied"
  { [ "${ul:-0}" = "${utot:-0}" ] && [ "${utot:-0}" -gt 0 ]; } 2>/dev/null \
    && ok "import: all ${utot} blocks on a $((SMALL/1000000)) MB budget -- weights that cost the device nothing cannot exceed it" \
    || bad "import: ${ul:-0} of ${utot:-?} blocks"
  [ "${RES:-1}" = "0.00" ] \
    && ok "import: 0.00 GiB resident with the whole model placed -- nothing was copied onto the device" \
    || bad "import: ${RES:-?} GiB resident, so the weights were copied after all"
  [ "$(ids unified)" = "$REF" ] && ok "import: ids identical to the host-only run" \
                                || bad "import: the ids moved"
fi
note ""

if [ -n "$VIOLATE" ]; then
  if [ "$fails" -gt 0 ]; then
    note "VIOLATION \"$VIOLATE\": $fails assertion(s) failed, which is the expected result."
    exit 0
  fi
  note "VIOLATION \"$VIOLATE\": NOTHING FAILED. The assertion it targets is not a gate."
  exit 1
fi
if [ "$fails" -eq 0 ]; then
  note "PROVED on this host: $DEV holds ${tot:-?} of ${tot:-?} blocks under a budget that fits ${bl:-?},"
  note "  by paging $(pagein paged) block(s) in and $(pageout paged) out ($(sed -n 's/^paging .*, \([0-9.]*\) GiB moved.*/\1/p' "$OUT/paged" | tail -1) GiB moved), 0 repacked;"
  note "  the same model under a budget that fits pages ZERO times;"
  note "  a budget shrunk mid-generation demoted blocks ($(pageout shrink) page-outs) and moved no placement;"
  note "  token ids identical to -devices cpu in all three."
  if [ -n "${UDEV:-}" ]; then
    note "  and on $UDEV, whose memory IS the host's, the same budget holds all ${utot:-?} blocks"
    note "  with ${RES:-?} GiB resident: ${IMP:-?} tensors WRAPPED in place, nothing transferred."
  fi
  exit 0
fi
note "$fails assertion(s) failed"
exit 1
