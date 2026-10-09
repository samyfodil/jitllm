#!/bin/sh
# decisiongold.sh writes the decision models' llama.cpp goldens:
# engine/model/testdata/decision/<model>.<request>.llamacpp.json, the answer
# llama-server's POST /v1/systemone gives for each request on the same GGUF
# the gate converts (engine/model/decide_test.go).
#
#   JITLLM_LCPP=/path/to/llama-b11514  JITLLM_MODELS=/path/to/models  scripts/decisiongold.sh
#
# The sources: LiquidAI/d1-3B-GGUF d1-3B-Q8_0.gguf into $JITLLM_MODELS/d1/,
# ggml-org/lev-GGUF lev-Q8_0.gguf into $JITLLM_MODELS/lev/, ggml-org/Laya-GGUF
# Laya-BF16.gguf into $JITLLM_MODELS/laya/gguf/. llama.cpp b11514
# (ggml-org/llama.cpp releases, llama-b11514-bin-ubuntu-x64) or newer.
set -eu
here=$(cd "$(dirname "$0")/.." && pwd)
out=$here/engine/model/testdata/decision
: "${JITLLM_LCPP:?JITLLM_LCPP names the llama.cpp build directory holding llama-server}"
: "${JITLLM_MODELS:?JITLLM_MODELS names the model directory}"
port=${PORT:-18089}

gold() { # gold <gguf> <name>
	"$here/scripts/cap" 12G -- taskset -c 0,2,4,6,8,10 "$JITLLM_LCPP/llama-server" \
		-m "$JITLLM_MODELS/$1" --port "$port" -t 6 -c 8192 >"$out/.srv.log" 2>&1 &
	pid=$!
	i=0
	until curl -s "localhost:$port/health" | grep -q ok; do
		i=$((i + 1))
		if [ $i -gt 300 ]; then
			echo "llama-server did not come up: $out/.srv.log" >&2
			kill $pid
			exit 1
		fi
		sleep 1
	done
	for req in support ambiguous; do
		curl -s "localhost:$port/v1/systemone" -H 'Content-Type: application/json' \
			-d @"$out/$req.json" >"$out/$2.$req.llamacpp.json"
		echo "$2 $req: $(head -c 200 "$out/$2.$req.llamacpp.json")"
	done
	# The cap wrapper may not pass a signal on; the server is named by its model.
	pkill -f "llama-server -m $JITLLM_MODELS/$1" || true
	wait $pid || true
	rm -f "$out/.srv.log"
}

gold d1/d1-3B-Q8_0.gguf d1-3B-Q8_0
gold lev/lev-Q8_0.gguf lev-Q8_0
gold laya/gguf/Laya-BF16.gguf Laya-BF16
