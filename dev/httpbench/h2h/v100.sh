#!/usr/bin/env bash
# HTTP head-to-head on 66.172.10.109 (2x E5-2680 v4, 8x V100-SXM2-16GB, sm70).
#
#   v100.sh ENGINE WORKLOAD
#
# ENGINE:   llama | vllm   (SGLang 0.5.10 refuses sm70; mistral.rs does not build for sm70: engines.md)
#           sglang is kept so a newer card/release can be tried; on a V100 it exits at load.
# WORKLOAD: smoke | aa | sweep | shared | multi
#
# Arms: jitllmd on card $JCARD, the engine on card $ECARD (identical V100s), both
# up at once, interleaved ABAB rounds. SAMECARD=1 puts both on $JCARD, which
# does not fit beside vLLM/SGLang's preallocation; with it, run one arm at a
# time (the script then runs each arm alone with its A/A, no ratio).
#
# Server processes: socket 0 (numactl --membind=0, cores 0-13); httpbench on
# socket 1 (cores 14-27). Every arm under one $MEM user slice.
#
# Kit layout on the box: $H2H/{bin,models,runs}; see engines.md for the downloads.
set -euo pipefail
ENGINE=${1:?engine: llama|vllm|sglang}
WORKLOAD=${2:?workload: smoke|aa|sweep|shared|multi}
export H2H=${H2H:-$HOME/h2h}
MEM=${MEM:-64G}
JCARD=${JCARD:-0}
ECARD=${ECARD:-1}
[ "${SAMECARD:-0}" = 1 ] && ECARD=$JCARD
export CUDA_VISIBLE_DEVICES=$JCARD   # jitllmd's card; each engine sets its own
SRV_PIN="numactl --membind=0 taskset -c 0-13"
LOADGEN="taskset -c 14-27"
RUN=${RUN:-$H2H/runs/v100-$ENGINE-$WORKLOAD-$(date +%Y%m%d-%H%M%S)}
# shellcheck source=lib.sh
. "$(dirname "$0")/lib.sh"

M=$H2H/models
GGUF8=$M/gguf/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
JLM8=$M/Meta-Llama-3.1-8B-Instruct-Q4_K_M.jlm
GGUF1=$M/gguf/Llama-3.2-1B-Instruct-Q4_K_M.gguf
JLM1=$M/Llama-3.2-1B-Instruct-Q4_K_M.jlm
AWQ8=$M/llama8b-awq            # hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
AWQ1=$M/llama1b-tok            # unsloth/Llama-3.2-1B-Instruct tokenizer files: the 1B GGUF's tokenizer under vLLM
VLLM=$HOME/engines/vllm-env/bin
SGL=$HOME/engines/sglang-env/bin
LLAMA=$HOME/llama-sm70/build/bin/llama-server
# vLLM/SGLang: the GGUF by default (same file as the other two arms);
# QUANT=awq serves the AWQ-INT4 checkpoint instead -- a DIFFERENT quantization,
# label the row with both formats. vLLM 0.18.1 refuses AWQ on sm70 (its awq
# method's minimum capability is 75); the GGUF loads (gguf's minimum is 60).
# The 8B AWQ directory is kept as the 8B GGUF's tokenizer source.
QUANT=${QUANT:-gguf}
# llama-server's KV: one unified pool of $KVPOOL positions shared by its $SLOTS
# slots (64 x 4096 f16 positions of an 8B is 32 GiB, twice the card). Every
# request still gets up to $CTX; jitllmd pages its KV and vLLM/SGLang size a
# pool from their memory fraction, so no GPU arm holds 64 full contexts.
KVPOOL=${KVPOOL:-65536}

JFLAGS="-devices cuda:0"
JLOADFLAGS="-devices cuda:0"

# engine_up NAME PORT MODELFILE TOKDIR MEMFRAC: one engine process serving one model as NAME.
engine_up() {
	local name=$1 port=$2 file=$3 tok=$4 frac=$5
	case $ENGINE in
	llama)
		launch engine "llama-$name" env CUDA_VISIBLE_DEVICES=$ECARD $SRV_PIN "$LLAMA" -m "$file" \
			--host 127.0.0.1 --port "$port" --alias "$name" -ngl 999 -np "$SLOTS" -c "$KVPOOL" --kv-unified \
			-t 14 --no-cache-prompt --cache-reuse 0 --cache-ram 0 --no-webui ;;
	vllm)
		local src=$file extra=(--tokenizer "$tok")
		[ "$QUANT" = awq ] && { src=$tok; extra=(--quantization awq); }
		launch engine "vllm-$name" env CUDA_VISIBLE_DEVICES=$ECARD VLLM_NO_USAGE_STATS=1 $SRV_PIN \
			"$VLLM/vllm" serve "$src" "${extra[@]}" --host 127.0.0.1 --port "$port" \
			--served-model-name "$name" --dtype float16 --max-model-len "$CTX" --max-num-seqs "$SLOTS" \
			--gpu-memory-utilization "$frac" --no-enable-prefix-caching ;;
	sglang)
		local src=$file extra=(--tokenizer-path "$tok" --load-format gguf)
		[ "$QUANT" = awq ] && { src=$tok; extra=(--quantization awq); }
		launch engine "sglang-$name" env CUDA_VISIBLE_DEVICES=$ECARD $SRV_PIN "$SGL/python" -m sglang.launch_server \
			--model-path "$src" "${extra[@]}" --host 127.0.0.1 --port "$port" --served-model-name "$name" \
			--dtype float16 --context-length "$CTX" --max-running-requests "$SLOTS" \
			--mem-fraction-static "$frac" --disable-radix-cache --attention-backend triton \
			--sampling-backend pytorch ;;
	*) log "unknown engine $ENGINE"; exit 2 ;;
	esac
	up "$port" 1800
}

slice engine
case $WORKLOAD in
smoke)
	# one arm at a time, each alone on its card
	jitllmd_up llama8b="$JLM8"
	smoke jitllm="http://127.0.0.1:$JPORT" -model llama8b
	stop_all; slice engine
	engine_up llama8b "$EPORT" "$GGUF8" "$AWQ8" 0.85
	smoke "$ENGINE=http://127.0.0.1:$EPORT" -model llama8b ;;
aa | sweep | shared)
	jitllmd_up llama8b="$JLM8"
	engine_up llama8b "$EPORT" "$GGUF8" "$AWQ8" 0.85
	aa_control "$ENGINE=http://127.0.0.1:$EPORT" -model llama8b
	aa_control "jitllm=http://127.0.0.1:$JPORT" -model llama8b
	[ "$WORKLOAD" = aa ] && exit 0
	ARMS=(-arm "jitllm=http://127.0.0.1:$JPORT" -arm "$ENGINE=http://127.0.0.1:$EPORT" -model llama8b)
	run_workload "$WORKLOAD" ;;
multi)
	# 8B and 1B on one card: jitllmd one process, the engine one process per
	# model with the card's memory fraction divided between them.
	jitllmd_up llama8b="$JLM8" llama1b="$JLM1"
	engine_up llama8b "$EPORT" "$GGUF8" "$AWQ8" 0.55
	engine_up llama1b "$((EPORT + 1))" "$GGUF1" "$AWQ1" 0.30
	ARMS=(-arm "jitllm=http://127.0.0.1:$JPORT" -arm "$ENGINE=http://127.0.0.1:$EPORT"
		-model llama8b:1 -model llama1b:3
		-route "$ENGINE/llama8b=http://127.0.0.1:$EPORT#llama8b"
		-route "$ENGINE/llama1b=http://127.0.0.1:$((EPORT + 1))#llama1b")
	aa_control "jitllm=http://127.0.0.1:$JPORT" -model llama8b:1 -model llama1b:3
	run_workload multi ;;
*) log "unknown workload $WORKLOAD"; exit 2 ;;
esac
log "results in $RUN"
