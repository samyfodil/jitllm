# The HTTP head-to-head kit: engines on the two servers

How each engine is made an arm of `dev/httpbench` on the two remote boxes.
The method (what a fair arm is, the A/A control, the ratio gates) is
`docs/perf/http-bench.md`; this file records what is installed where, the
exact command each script runs, and what each engine cannot do. It carries no
speed numbers: the smoke tests below checked that an arm answers and honours
`max_tokens`/`ignore_eos`, nothing else.

Common to every arm, set by `lib.sh`:

| knob | value |
|---|---|
| context per sequence | `CTX=4096` (jitllmd `-max-seq`, llama-server per request, vLLM `--max-model-len`, SGLang `--context-length`, mistral.rs `--max-seq-len`) |
| concurrency capacity | `SLOTS=64` (jitllmd `-max-batch`, llama-server `-np`, vLLM `--max-num-seqs`, SGLang `--max-running-requests`, mistral.rs `--max-seqs`) |
| prefix caching | OFF on every arm: jitllmd `-no-mem-cache`; llama-server `--no-cache-prompt --cache-reuse 0 --cache-ram 0`; vLLM `--no-enable-prefix-caching`; SGLang `--disable-radix-cache`; mistral.rs `--prefix-cache-n 0` |
| temperature | 0 (httpbench's default; every request names it) |
| max_tokens / ignore_eos | httpbench sends both on every request; the "at max" column is the check |
| memory | each arm in its own user slice `h2h-jitllm.slice` / `h2h-engine.slice`, `MemoryMax=$MEM`, `MemorySwapMax=0` (systemd-run --user --scope; `scripts/cap` is not on the servers). An engine that serves one model per process runs all of its processes in the one engine slice |
| cores | servers on socket 0 under `numactl --membind=0`, httpbench on socket 1 (`taskset -c 14-27`) |
| API | `/v1/completions` (no template); the smoke also sends one `-api chat` request |

llama-server's KV is one unified pool (`--kv-unified -c $KVPOOL`, default
65536 positions) shared by its 64 slots, not 64 x 4096: 64 full contexts of
the 8B are 32 GiB of f16 KV, twice a V100, and three CPU processes of them do
not fit the multi row's slice. Each request still gets up to 4096. jitllmd
pages its KV and vLLM/SGLang size their pool from a memory fraction, so no
arm holds 64 full contexts; a row at -c 64 with 512-word prompts uses about
41k positions, inside the pool.

## 66.172.10.109: 2x E5-2680 v4, 8x V100-SXM2-16GB (sm70), CUDA driver 580 / 13.0

Kit at `~/h2h` (`bin/`, `models/`, `runs/`, `setup/` logs). Arms: jitllmd on
card `JCARD=0`, the engine on card `ECARD=1` (two identical cards, so both
servers stay up for interleaved ABAB rounds without sharing a card's memory);
`SAMECARD=1` puts both on card 0, which only fits with llama-server.

Models (downloaded on the box from Hugging Face):

| file | source |
|---|---|
| `models/gguf/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | bartowski/Meta-Llama-3.1-8B-Instruct-GGUF |
| `models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.jlm` | `jitllm convert` of the GGUF above (container from this branch's binary) |
| `models/gguf/Llama-3.2-1B-Instruct-Q4_K_M.gguf`, `models/Llama-3.2-1B-Instruct-Q4_K_M.jlm` | bartowski/Llama-3.2-1B-Instruct-GGUF; the multi row's second model |
| `models/llama8b-awq` | hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4: the 8B GGUF's tokenizer under vLLM, and the `QUANT=awq` arm (refused on sm70, below) |
| `models/llama1b-tok` | unsloth/Llama-3.2-1B-Instruct tokenizer files only: the 1B GGUF's tokenizer under vLLM |

| engine | version | path | serve command (as `v100.sh` runs it) | model name | smoke |
|---|---|---|---|---|---|
| jitllmd | origin/main bd662f68 | `~/h2h/bin/jitllmd` | `CUDA_VISIBLE_DEVICES=0 numactl --membind=0 taskset -c 0-13 jitllmd serve -addr 127.0.0.1:8080 -models ~/h2h/models -load <8B .jlm> -id llama8b -max-batch 64 -max-seq 4096 -no-mem-cache -devices cuda:0` | `llama8b` | OK: completions and chat, usage on every request, at max 2/2 |
| llama.cpp (CUDA) | 0.1.1-dev, commit 60eeeb6 | `~/llama-sm70/build/bin/llama-server` | `CUDA_VISIBLE_DEVICES=1 numactl --membind=0 taskset -c 0-13 llama-server -m <8B gguf> --host 127.0.0.1 --port 8081 --alias llama8b -ngl 999 -np 64 -c 65536 --kv-unified -t 14 --no-cache-prompt --cache-reuse 0 --cache-ram 0 --no-webui` | `llama8b` | OK: both routes, usage, at max 2/2. `-c 262144` (64 x 4096) fails: `cudaMalloc failed: out of memory` allocating 32 GiB of KV |
| vLLM | 0.18.1 (torch 2.10.0, transformers 4.57.6), V1 engine, TRITON_ATTN backend | `~/engines/vllm-env` (uv venv, Python 3.12) | `CUDA_VISIBLE_DEVICES=1 VLLM_NO_USAGE_STATS=1 numactl --membind=0 taskset -c 0-13 vllm serve <8B gguf> --tokenizer ~/h2h/models/llama8b-awq --host 127.0.0.1 --port 8081 --served-model-name llama8b --dtype float16 --max-model-len 4096 --max-num-seqs 64 --gpu-memory-utilization 0.85 --no-enable-prefix-caching` | `llama8b` | OK: both routes, usage, at max 2/2, on the SAME Q4_K_M GGUF |
| SGLang | 0.5.10 (torch 2.9.1+cu128) | `~/engines/sglang-env` | `... python -m sglang.launch_server --model-path <gguf> --tokenizer-path <tok> --load-format gguf --served-model-name llama8b --dtype float16 --context-length 4096 --max-running-requests 64 --mem-fraction-static 0.85 --disable-radix-cache --attention-backend triton --sampling-backend pytorch` | `llama8b` | REFUSED at load: `RuntimeError: SGLang only supports sm75 and above.` (`sglang/srt/model_executor/model_runner.py:1089`, `load_model`). Not a flag: the check is on the device's capability |
| mistral.rs (CUDA) | v0.9.4 | none built | -- | -- | NOT BUILDABLE for sm70. No sm70 release asset (wheels ship sm80/86/89/90/100/120 only). Source build `CUDA_COMPUTE_CAP=70 cargo build --release --features cuda -p mistralrs-cli` with the box's nvcc 12.0 fails in `mistralrs-paged-attn`: `flash_attn_sinks.cu(191): error: no instance of overloaded function "__ldg" matches the argument list, argument types are: (const __nv_bfloat16 *)` (a bf16 `__ldg` overload exists only for sm80+). The partial build is in `~/h2h/mistralrs-src`, rustup/cargo in `~/h2h/{rustup,cargo}` |
| ZML | -- | not on this box | -- | -- | not installed here; see .59 |

vLLM's history on the box: `~/finalbatch.log` and `~/.config/vllm/usage_stats.json`
show vLLM 0.18.1 ran there before (tensor-parallel 2, fp16, the same GGUF), from
a venv since deleted; 0.18.1 was reinstalled because that run proves it
supports sm70 (V1 with the Triton attention backend; FA2 is refused below
sm80 and vLLM falls back by itself).

What vLLM cannot do here:
- **AWQ on sm70**: vLLM's `awq` method has minimum capability 75 (`awq.py`),
  so `QUANT=awq` is refused on a V100. `gguf` and `gptq` have minimum 60. The
  GGUF arm is the same file the other arms run, so no quantization label is
  needed for the default; a GPTQ arm would be a different quantization and is
  labelled with both formats.
- GGUF in vLLM is an experimental path: it reads the GGUF's weights, but takes
  the tokenizer from `--tokenizer` (the model's HF tokenizer), and it dequantizes
  through its own GGUF kernels (not llama.cpp's). The prompt-token column must
  agree with the other arms before a row means anything; in the smoke it did
  (65 and 99 tokens on every arm).
- First start compiles (torch.compile + CUDA graphs for batch 1..128): about
  2-3 minutes; `-warmup` covers first-request costs after that.

## 66.172.10.59: 2x E5-2680 v4, CPU only

Kit at `/dev/shm/h2h` (`/` is full). Servers on socket 0's twelve cores
(`numactl --membind=0 taskset -c 0-11`, the board's set), httpbench on
`taskset -c 14-27`.

Models: the GGUFs in `/dev/shm/cpu-gguf` (Llama-3.2-1B-Instruct, Meta-Llama-3.1-8B-Instruct,
Qwen3-30B-A3B, all Q4_K_M); `.jlm` containers reconverted into
`/dev/shm/h2h/models` with this branch's `jitllm convert` (the ones in
`~/cpu-models` are container v26 and stale). Model names: `l1b`, `l8b`, `q30`.

| engine | version | path | serve command (as `cpu59.sh` runs it) | model name | smoke |
|---|---|---|---|---|---|
| jitllmd | origin/main bd662f68 | `/dev/shm/h2h/bin/jitllmd` | `numactl --membind=0 taskset -c 0-11 jitllmd serve -addr 127.0.0.1:8080 -models /dev/shm/h2h/models -load <.jlm> -id l1b -max-batch 64 -max-seq 4096 -no-mem-cache -devices cpu` (others added with `jitllmd models -load ... -id ...`) | `l1b`/`l8b`/`q30` | OK on l1b and q30: both routes, usage, at max 2/2 |
| llama.cpp (CPU) | b11222, commit a97cce86a | `~/cpu-bench/llama/llama-b11222/llama-server` | `numactl --membind=0 taskset -c 0-11 llama-server -m /dev/shm/cpu-gguf/<f>.gguf --host 127.0.0.1 --port 8081 --alias l1b -np 64 -c 65536 --kv-unified -t 12 -tb 12 --no-cache-prompt --cache-reuse 0 --cache-ram 0 --no-webui` | as aliased | OK on l1b and q30: both routes, usage, at max 2/2, prompt tokens equal to jitllm's |
| mistral.rs (CPU) | 0.9.4 | `~/tools/mrs-bin/mistralrs` | from `/dev/shm/h2h/models/mrs`: `numactl --membind=0 taskset -c 0-11 mistralrs serve --cpu --host 127.0.0.1 -p 8081 --max-seqs 64 --max-seq-len 4096 --prefix-cache-n 0 --no-ui -m l1b -f <f>.gguf` (`l1b/` holds a symlink to the GGUF; the GGUF's own tokenizer and template) | the `-m` directory name (`l1b`) | l1b: chat OK (usage, at max 2/2); completions **FAIL the usage check**: the streamed `/v1/completions` carries no usage object, so httpbench falls back to counting chunks (prompt_tokens reads 0; at max 2/2 by chunk count). Its chat prompt is 101 tokens where jitllm's and llama-server's are 99 (a different render of the GGUF template). q30 (Qwen3-30B-A3B): the server came up but the warm-up request (64-word prompt, 32 tokens) did not finish inside httpbench's 10-minute timeout: `warm-up request failed: context deadline exceeded` |
| ZML | commit 74c590b (2026-09-28) | `~/tools/zml` | -- | -- | NOT AN ARM: it has no OpenAI-compatible HTTP server (`examples/llm` is a CLI, `bazel-bin/examples/llm/llm --model=... --prompt=...`; no `/v1/completions` anywhere in the tree), and it runs bf16 safetensors, not the GGUF. It stays on `~/xboard.sh`'s CLI board only |

What this means for a row:
- mistral.rs on `/v1/completions` is counted by chunks, not usage; read its
  token column with that in mind or take its rows with `-api chat` and accept
  the 2-token template difference (name it in the row).
- mistral.rs on the MoE is not runnable inside the default timeout; the multi
  row with mistral.rs either drops q30 or raises `-timeout`, and says which.

## Scripts

    # on the box, after rsyncing dev/httpbench/h2h/ and the binaries into $H2H/bin
    v100.sh  ENGINE WORKLOAD          # ENGINE llama|vllm (sglang exits at load on sm70)
    cpu59.sh ENGINE WORKLOAD [MODEL]  # ENGINE llama|mistralrs; MODEL l1b|l8b|q30

WORKLOAD:

| name | what |
|---|---|
| `smoke` | one arm at a time: one completion and one streamed chat request, `-c 1 -rounds 1 -requests 2`, then a `SMOKE OK/FAIL` line from the JSON (usage seen, completion tokens == max_tokens) |
| `aa` | both arms up; A/A self-control on the engine, then on jitllmd (`-c 1,16,64`, `ROUNDS`) |
| `sweep` | A/A on each, then the interleaved A/B: `-c 1,4,16,64 -prompt-lens 128,512 -max-tokens 128` |
| `shared` | A/A on each, then the A/B on a generated `.jsonl`: a ~1500-word shared system prefix and a 12-word question per line (256 lines, fixed seed), `-c 1,4,16 -max-tokens 64` |
| `multi` | several models at once: jitllmd one process with every model loaded; the engine one process per model in the one engine slice, `-model NAME:WEIGHT` + `-route ARM/MODEL=URL#NAME`. GPU: llama8b:1 + llama1b:3 on one card (vLLM's memory fraction divided 0.55/0.30). CPU: l1b:3 + l8b:1 + q30:1. The A/A runs on the jitllm arm only |

Environment: `ROUNDS` (8), `MEM` (64G GPU, 48G CPU), `CTX` (4096), `SLOTS`
(64), `KVPOOL` (65536), `JPORT`/`EPORT` (8080/8081, a multi row's engine
models on 8081, 8082, ...), `RUN` (the run directory; default
`$H2H/runs/<server>-<engine>-<workload>-<time>`), GPU only `JCARD`/`ECARD`/
`SAMECARD`, `QUANT`.

Each run directory holds `run.log` (everything printed), one log per server
process, and one JSON per httpbench invocation (`aa-<arm>.json`, `sweep.json`,
`shared.json`, `multi.json`, `smoke-*.json`). The script stops everything it
started on exit (an EXIT trap: the launched scopes, then both slices).

## Before a row is fair

- Interleaving keeps both servers resident. On the CPU box both arms sit on
  the same twelve cores and only one is driven at a time; an engine that
  spins while idle would steal from the other arm. Check `top` on cores 0-11
  while the other arm is measured before trusting a row; if one spins, run
  process-per-arm (one arm up at a time, each with its A/A) and say so.
- GPU arms on two different (identical) cards: the row names both cards. A
  same-card row needs `SAMECARD=1` and an engine that leaves room (llama-server
  does; vLLM at 0.85 does not).
- The multi row's engine arm is N processes under one memory limit; the row
  says so (http-bench.md, "Several models at once").
- jitllmd's GPU arm also holds host frames and uses socket 0's cores; the
  engine's CPU side sits on the same socket. Same core set for both, by
  construction.
