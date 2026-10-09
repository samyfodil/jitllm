#!/usr/bin/env bash
# HTTP head-to-head on 66.172.10.59 (2x E5-2680 v4, CPU only).
#
#   cpu59.sh ENGINE WORKLOAD [MODEL]
#
# ENGINE:   llama | mistralrs            (ZML serves no HTTP: engines.md)
# WORKLOAD: smoke | aa | sweep | shared | multi
# MODEL:    l1b | l8b | q30              (sweep/shared/aa/smoke; multi loads all three)
#
# Servers on socket 0 (numactl --membind=0, cores 0-11, the board's set);
# httpbench on socket 1 (cores 14-27). Both arms up at once and idle while the
# other is measured (ABAB rounds): two CPU servers on the same 12 cores never
# run concurrently in this harness, since a round runs one arm then the other.
# Each arm under its own $MEM user slice.
#
# / is full on this box: everything lives in /dev/shm/h2h.
set -euo pipefail
ENGINE=${1:?engine: llama|mistralrs}
WORKLOAD=${2:?workload: smoke|aa|sweep|shared|multi}
MODEL=${3:-l1b}
export H2H=${H2H:-/dev/shm/h2h}
MEM=${MEM:-48G}
SRV_PIN="numactl --membind=0 taskset -c 0-11"
LOADGEN="taskset -c 14-27"
RUN=${RUN:-$H2H/runs/cpu59-$ENGINE-$WORKLOAD-$MODEL-$(date +%Y%m%d-%H%M%S)}
# shellcheck source=lib.sh
. "$(dirname "$0")/lib.sh"

M=$H2H/models
G=/dev/shm/cpu-gguf
declare -A FILE=([l1b]=Llama-3.2-1B-Instruct-Q4_K_M [l8b]=Meta-Llama-3.1-8B-Instruct-Q4_K_M [q30]=Qwen3-30B-A3B-Q4_K_M)
LLAMA=$HOME/cpu-bench/llama/llama-b11222/llama-server
MRS=$HOME/tools/mrs-bin/mistralrs
# llama-server's KV: one unified pool shared by its slots (64 full 4096
# contexts of the 8B would be 32 GiB per process; three in a multi row do not fit)
KVPOOL=${KVPOOL:-65536}
JFLAGS="-devices cpu"
JLOADFLAGS="-devices cpu"

# engine_up NAME PORT: one engine process serving model NAME (a FILE key).
engine_up() {
	local name=$1 port=$2 f=${FILE[$1]}
	case $ENGINE in
	llama)
		launch engine "llama-$name" $SRV_PIN "$LLAMA" -m "$G/$f.gguf" --host 127.0.0.1 --port "$port" \
			--alias "$name" -np "$SLOTS" -c "$KVPOOL" --kv-unified -t 12 -tb 12 \
			--no-cache-prompt --cache-reuse 0 --cache-ram 0 --no-webui ;;
	mistralrs)
		# mistral.rs wants a directory and a file name; the GGUF's own tokenizer is used
		mkdir -p "$M/mrs/$f" && ln -sf "$G/$f.gguf" "$M/mrs/$f/"
		launch engine "mistralrs-$name" $SRV_PIN "$MRS" serve --cpu --host 127.0.0.1 -p "$port" \
			--max-seqs "$SLOTS" --max-seq-len "$CTX" --prefix-cache-n 0 --no-ui \
			-m "$M/mrs/$f" -f "$f.gguf" ;;
	*) log "unknown engine $ENGINE"; exit 2 ;;
	esac
	up "$port" 1800
}

slice engine
case $WORKLOAD in
smoke)
	jitllmd_up "$MODEL=$M/${FILE[$MODEL]}.jlm"
	smoke jitllm="http://127.0.0.1:$JPORT" -model "$MODEL"
	stop_all; slice engine
	engine_up "$MODEL" "$EPORT"
	smoke "$ENGINE=http://127.0.0.1:$EPORT" -model "$MODEL" ;;
aa | sweep | shared)
	jitllmd_up "$MODEL=$M/${FILE[$MODEL]}.jlm"
	engine_up "$MODEL" "$EPORT"
	aa_control "$ENGINE=http://127.0.0.1:$EPORT" -model "$MODEL"
	aa_control "jitllm=http://127.0.0.1:$JPORT" -model "$MODEL"
	[ "$WORKLOAD" = aa ] && exit 0
	ARMS=(-arm "jitllm=http://127.0.0.1:$JPORT" -arm "$ENGINE=http://127.0.0.1:$EPORT" -model "$MODEL")
	run_workload "$WORKLOAD" ;;
multi)
	# all three at once: jitllmd one process; the engine three processes, one
	# per model, in one $MEM slice -- the row says "three processes".
	jitllmd_up l1b="$M/${FILE[l1b]}.jlm" l8b="$M/${FILE[l8b]}.jlm" q30="$M/${FILE[q30]}.jlm"
	engine_up l1b "$EPORT"
	engine_up l8b "$((EPORT + 1))"
	engine_up q30 "$((EPORT + 2))"
	ARMS=(-arm "jitllm=http://127.0.0.1:$JPORT" -arm "$ENGINE=http://127.0.0.1:$EPORT"
		-model l1b:3 -model l8b:1 -model q30:1
		-route "$ENGINE/l1b=http://127.0.0.1:$EPORT#l1b"
		-route "$ENGINE/l8b=http://127.0.0.1:$((EPORT + 1))#l8b"
		-route "$ENGINE/q30=http://127.0.0.1:$((EPORT + 2))#q30")
	aa_control "jitllm=http://127.0.0.1:$JPORT" -model l1b:3 -model l8b:1 -model q30:1
	run_workload multi ;;
*) log "unknown workload $WORKLOAD"; exit 2 ;;
esac
log "results in $RUN"
