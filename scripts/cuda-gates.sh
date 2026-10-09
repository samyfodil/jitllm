#!/usr/bin/env bash
# The CUDA backend's gates on a host with an NVIDIA card, run from a bundle of
# shipped test binaries rather than the source tree (scripts/kaggle/bundle.sh
# builds one; AGENTS.md RULE 11d has the shipped-binary recipe):
#
#   JITLLM_MODELS=/path/to/models ./cuda-gates.sh
#
# The bundle holds cap, jitllm, and pkg/<package>/<name>.test beside the
# package's testdata. Each binary runs from its package directory, so its
# testdata is where `go test` would have put it. Every step runs under cap
# (RULE 3). The run fails if any gate skipped for want of a device or a model,
# or if no gate named the ptx backend: on this host a skip is a gate that did
# not run.
set -uo pipefail
cd "$(dirname "$0")"
here=$PWD
: "${JITLLM_MODELS:?set JITLLM_MODELS to the model directory (docs/testing.md)}"
export JITLLM_SHIPPED_BINARY=1 CGO_ENABLED=0
log=$(mktemp)
trap 'rm -f "$log"' EXIT
fail=0

nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv 2>&1 | tee -a "$log"

# gate <package dir> <mem> [test flags...]
gate() {
	pkg=$1 mem=$2
	shift 2
	bin=$(ls "$here/pkg/$pkg"/*.test)
	echo "== $pkg $*"
	(cd "$here/pkg/$pkg" && "$here/cap" "$mem" -- "$bin" -test.count=1 -test.v -test.timeout=3h "$@") 2>&1 | tee -a "$log"
	rc=${PIPESTATUS[0]}
	[ "$rc" = 0 ] || { echo "== $pkg exited $rc"; fail=1; }
}

gate jit/gpu/cuda 8G
# Every backend kernel gate runs on every device backend.Open returns, the
# CUDA device among them.
gate jit/gpu/backend 16G
gate jit/gpu/tier 8G
# The model gates. gpu:0 is the first device backend.Open returns, which on
# an NVIDIA host is the CUDA device.
gate engine/model 24G -test.run \
	'TestEveryModelRunsGenerated|TestGreedyMatchesLlamaCpp|TestBatchMatchesForward|TestDecodeDoesNotAllocate|TestLayerRelocatesToAnotherDevice|TestRelocatedBlocksComeBack|TestKVPagesRelocateWithTheLayer|TestPagingDoesNotChangeTheAnswer|TestOpenHonoursAPageBudget|TestForwardBatchFaultsItsBlocksIn|TestSlidingWindowRunsOnTheDevice|TestDeviceMatchesHostOnABiasedModel'

echo "== skips"
grep -B1 -E -- '--- SKIP' "$log" | grep -vE -- '^--$' || true
if grep -E 'no GPU|no (cuda|CUDA) (device|driver)|no device|NO DEVICE|MODEL MISSING|CARD TOO SMALL|took no block' "$log" | grep -v 'no two devices'; then
	echo "FAIL: a gate skipped for want of a device or a model on the host meant to run it" >&2
	fail=1
fi
if ! grep -qE '/ptx\b|"ptx"| ptx ' "$log"; then
	echo "FAIL: no gate named the ptx backend; the CUDA device did not run" >&2
	fail=1
fi

# A short decode row, for the record only: one card, three runs, the GPU named.
llama=$JITLLM_MODELS/Llama-3.2-1B-Instruct-Q4_K_M
if [ -f "$llama.gguf" ]; then
	[ -f "$llama.jlm" ] || "$here/cap" 16G -- "$here/jitllm" convert "$llama.gguf"
	echo "== decode row: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), Llama-3.2-1B-Instruct Q4_K_M, cuda:0, n=128"
	for i in 1 2 3; do
		"$here/cap" 16G -- "$here/jitllm" run -devices cuda:0 -n 128 "$llama.jlm" "Once upon a time" 2>&1 | grep -E 'tok/s|B/token|device|error' | tee -a "$log"
	done
else
	echo "FAIL: no $llama.gguf for the decode row" >&2
	fail=1
fi

[ "$fail" = 0 ] && echo "CUDA gates passed" || echo "CUDA gates FAILED"
exit "$fail"
