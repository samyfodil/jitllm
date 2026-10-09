# shellcheck shell=bash
# Shared by cpu59.sh and v100.sh: start/stop arms under one memory-capped
# slice per arm, poll health, and drive dev/httpbench. Sourced, not run.
#
# Each arm runs in its own user slice (h2h-jitllm.slice, h2h-engine.slice)
# whose MemoryMax is $MEM, so an engine that serves one model per process
# holds all of its processes under the same limit as jitllmd's one process.

set -euo pipefail

H2H=${H2H:?set H2H to the kit directory on the server}
BIN=${BIN:-$H2H/bin}
MEM=${MEM:-48G}
CTX=${CTX:-4096}            # context per sequence, every engine
SLOTS=${SLOTS:-64}          # concurrency capacity, >= the highest level swept
JPORT=${JPORT:-8080}        # jitllmd
EPORT=${EPORT:-8081}        # the engine's first port; model i gets EPORT+i
LOADGEN=${LOADGEN:-taskset -c 14-27}
ROUNDS=${ROUNDS:-8}
RUN=${RUN:-$H2H/runs/$(date +%Y%m%d-%H%M%S)}
mkdir -p "$RUN"
STARTED=()

log() { echo "[h2h $(date +%T)] $*" | tee -a "$RUN/run.log" >&2; }

slice() { # NAME: (re)create a user slice capped at $MEM
	systemctl --user set-property --runtime "h2h-$1.slice" "MemoryMax=$MEM" "MemorySwapMax=0"
}

launch() { # SLICE LOGNAME CMD...: start CMD in the background inside the slice
	local s=$1 name=$2; shift 2
	systemd-run --user --scope -q --slice="h2h-$s.slice" -- "$@" >"$RUN/$name.log" 2>&1 &
	STARTED+=($!)
	log "started $name pid $! : $*"
}

up() { # PORT [TIMEOUT_S]: wait for /v1/models to answer 200
	local p=$1 t=${2:-900} i
	for ((i = 0; i < t; i += 2)); do
		curl -sf "http://127.0.0.1:$p/v1/models" >/dev/null && { log "port $p up after ${i}s"; return 0; }
		kill -0 "${STARTED[-1]}" 2>/dev/null || { log "port $p: the process died, see its log in $RUN"; return 1; }
		sleep 2
	done
	log "port $p NOT up after ${t}s"; return 1
}

stop_all() {
	local p
	for p in "${STARTED[@]}"; do pkill -TERM -P "$p" 2>/dev/null || true; kill -TERM "$p" 2>/dev/null || true; done
	sleep 3
	for p in "${STARTED[@]}"; do pkill -KILL -P "$p" 2>/dev/null || true; kill -KILL "$p" 2>/dev/null || true; done
	# the scopes' children: everything left in our two slices
	systemctl --user stop h2h-jitllm.slice h2h-engine.slice 2>/dev/null || true
	STARTED=()
	log "stopped every arm"
}
trap stop_all EXIT

# jitllmd ID=PATH...: one daemon serving every model, loaded at once.
# Extra serve flags go in $JFLAGS (e.g. "-devices cuda:0").
jitllmd_up() {
	local first=$1; shift
	slice jitllm
	# -no-mem-cache: jitllmd's memory cache is a prefix cache; off like every other engine's
	# shellcheck disable=SC2086
	launch jitllm jitllmd $SRV_PIN "$BIN/jitllmd" serve -addr "127.0.0.1:$JPORT" -models "$H2H/models" \
		-load "${first#*=}" -id "${first%%=*}" -max-batch "$SLOTS" -max-seq "$CTX" -no-mem-cache $JFLAGS
	up "$JPORT"
	local m
	for m in "$@"; do
		# shellcheck disable=SC2086
		"$BIN/jitllmd" models -addr "127.0.0.1:$JPORT" -load "${m#*=}" -id "${m%%=*}" $JLOADFLAGS | tee -a "$RUN/run.log"
	done
}

hb() { # httpbench ARGS...: on the load generator's cores, -o set by the caller
	log "httpbench $*"
	# shellcheck disable=SC2086
	$LOADGEN "$BIN/httpbench" "$@" 2>&1 | tee -a "$RUN/run.log"
}

# The shared-system-prompt workload: one ~1500-word prefix (a fixed word list,
# so the same file every run) and a short distinct question per line.
shared_jsonl() { # OUT [LINES]
	python3 - "$1" "${2:-256}" <<'PY'
import json, random, sys
out, n = sys.argv[1], int(sys.argv[2])
w = ("the of and to in is that for it as with was on be by this are or from at an which have not "
     "but all were when there can more one has been would their will about if so what out up into "
     "system policy customer account order service support request response product team data report "
     "time year people way day work part place case point group number world area state").split()
r = random.Random(7)
prefix = "You are the support assistant for a large retailer. Policy follows. " + \
         " ".join(r.choice(w) for _ in range(1500)) + "\n\n"
with open(out, "w") as f:
    for i in range(n):
        q = " ".join(r.choice(w) for _ in range(12))
        f.write(json.dumps({"prompt": prefix + f"Question {i}: {q}?\nAnswer:"}) + "\n")
PY
}

# The three workloads. ARMS is the -arm/-route/-model list, built by the caller.
#   aa_control ARM=URL MODELFLAGS...   A/A on one endpoint, before any A/B (RULE 2)
#   w_sweep    ARGS...                 -c 1,4,16,64, prompts 128/512 words, 128 tokens
#   w_shared   ARGS...                 the shared prefix file, -c 1,4,16
#   w_multi    ARGS...                 caller passes -model NAME:W and -route per model
aa_control() {
	local arm=$1; shift
	hb -aa -arm "$arm" "$@" -rounds "$ROUNDS" -c 1,16,64 -max-tokens 128 -prompt-lens 128,512 \
		-o "$RUN/aa-${arm%%=*}.json"
}
w_sweep() { hb "$@" -rounds "$ROUNDS" -c 1,4,16,64 -prompt-lens 128,512 -max-tokens 128 -o "$RUN/sweep.json"; }
w_shared() {
	[ -f "$RUN/shared.jsonl" ] || shared_jsonl "$RUN/shared.jsonl"
	hb "$@" -rounds "$ROUNDS" -c 1,4,16 -prompts "$RUN/shared.jsonl" -max-tokens 64 -o "$RUN/shared.json"
}
w_multi() { hb "$@" -rounds "$ROUNDS" -c 4,16 -prompt-lens 128,512 -max-tokens 128 -o "$RUN/multi.json"; }

# Smoke: one completion and one streamed chat request, 2 requests each, -c 1.
# Prints the at-max column; the caller reads it (100% means max_tokens and
# ignore_eos were honoured).
smoke() { # ARM=URL MODELFLAGS...
	local arm=$1; shift
	hb -arm "$arm" "$@" -c 1 -rounds 1 -requests 2 -warmup 1 -max-tokens 32 -prompt-lens 64 \
		-o "$RUN/smoke-${arm%%=*}-completions.json"
	hb -arm "$arm" "$@" -api chat -c 1 -rounds 1 -requests 2 -warmup 1 -max-tokens 32 -prompt-lens 64 \
		-o "$RUN/smoke-${arm%%=*}-chat.json"
	python3 - "$RUN/smoke-${arm%%=*}-completions.json" "$RUN/smoke-${arm%%=*}-chat.json" <<'PY' | tee -a "$RUN/run.log"
import json, sys
for p in sys.argv[1:]:
    d = json.load(open(p))
    mt = d["config"]["max_tokens"]
    s = [x for lv in d["levels"] for x in lv["samples"]]
    errs = [x.get("error") for x in s if x.get("error")]
    usage = all(x["usage_seen"] for x in s)
    atmax = sum(x["completion_tokens"] == mt for x in s)
    ok = s and not errs and usage and atmax == len(s)
    print(f'SMOKE {"OK" if ok else "FAIL"} {p.rsplit("/", 1)[-1]}: {len(s)} requests, '
          f'usage on all={usage}, at max {atmax}/{len(s)} (max_tokens {mt}), '
          f'prompt_tokens {[x["prompt_tokens"] for x in s]}, '
          f'finish {sorted({x["finish_reason"] for x in s})}, errors {errs[:2]}')
PY
}

# run WORKLOAD: what each server script's main calls once both arms are up.
run_workload() {
	case $1 in
	sweep) w_sweep "${ARMS[@]}" ;;
	shared) w_shared "${ARMS[@]}" ;;
	multi) w_multi "${ARMS[@]}" ;;
	*) log "unknown workload $1"; return 2 ;;
	esac
}
