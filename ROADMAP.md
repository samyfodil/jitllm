# jitllm roadmap

What is planned, in priority order. Each item carries its status. The
engineering history behind a status lives in `docs/engineering-history/`; this
file says only where each item stands and what is left.

## 1. Optimized kernels for MoE

**Largely done.**

- **Host dispatch is batched.** The k experts' gate/up share one activation and
  run in one pool region (`nn.MatVecPackedMulti`), and the down projections in
  one more (`MatVecPackedGather`): olmoe went from 906 to 234 regions a token,
  a dense model's density, and 10% less kernel CPU.
- **Routing is generated code.** The softmax/sigmoid, the top-k, the ascending
  sort and the renormalising divide run as kernels on the host and every device
  backend.
- **Every expert is its own page** (container v26): an over-committed mixture
  reads only the experts a token routes to, 3.10x on Moonlight-16B-A3B under a
  6 GB budget.
- **On the device** the indexed matvec splits over k (1.31x on
  Qwen3-MoE-4x0.6B), and a streamed placement (`tier.Config.StreamExperts`)
  uploads only the routed sheets, so Qwen3-Next-80B places 48 of 48 blocks on a
  2 GiB card.
- **gpt-oss is supported**, on the host and every device backend, with MXFP4 stored natively (container v24).

Where it stands: olmoe beats llama.cpp on the Xeon (1.06); the M4's CPU mixture
rows are the weakest on the board.

Left:
- a batched mixture path that pays on arm64 (it exists and is off there: the
  fused kernel's walk loses 24% to the per-expert loop at a mixture's shapes;
  it would have to drive the tiled kernel);
- fusing gate and up into one launch on the device, unmeasured.

## 2. Small models: BERT and ModernBERT

**BERT is done; ModernBERT is not started.**

`bert` and `nomic-bert` run as encoder graphs (`engine/model/encoder.go`), and
qwen3-embedding and EmbeddingGemma run as pooled readouts of their decoder
graphs. One API serves both (`Model.NewEmbedder`, `jitllm embed`), WordPiece is
in `tok/wpm.go`, and the outputs are gated against llama.cpp and float32
sentence-transformers.

Left: ModernBERT (llama.cpp's `modern-bert` pre-tokenizer is already in the
table; the architecture is not in `convert.archOf`).
ModernBERT is also the encoder most current PII taggers and routing
classifiers are built on, so it comes before items 13 and 14.

## 3. Model and request parameters, complete

**Mostly done.** `model.Sampler` has temperature, top-k, top-p, min-p, a
repetition penalty with its window, and a seed, and the sampler itself is
generated code. The server's `GenerateRequest` adds stop strings and a token
budget. Rope and YaRN parameters come from the container.

Left: frequency and presence penalties.

## 4. Concurrency

**Serving works on the host; the next piece is designed below.**

Done:
- **Ragged positions.** Each batch row has its own position, rotary table, KV
  offset and attention length, and `Retire(i)` frees a row.
- **A scheduler.** `model.Scheduler` admits queued sequences into free rows,
  steps them, and retires them on a stop or a budget.
- **Per-row prefill.** `PrefillSeq` runs an admitted row's whole prompt in one
  pass instead of one token per step.
- **Several sequences on one device tier.** `tier.GPU.Attach` gives each
  `model.State` its own session and KV history; sessions are serialised per
  device, not yet parallel.

Still refused: a *batched* State with blocks on a device (`errBatchDevice`).

**Next: fold prefill into the decode step.** The idea as put was "prefill the
next prompt while the current one finishes inference". It is not about
concurrency -- prefill and decode would contend for the same cores -- it is
about the weight pass. A decode step is memory-bound with its arithmetic idle,
so prompt tokens folded into it ride on a weight read already being paid for.
`Scheduler.admit` runs `PrefillSeq` as its own pass, so every live row
stalls for it: on tinyllama, admitting a 500-token prompt costs about 260
decode tokens of everyone else's time.

It needs a token *count* per row, not just a position: row A contributes one
decode token, row B sixteen prompt tokens, in one step over sum(n_i) tokens
with a per-token (slot, position) map. `Prefill` and `ForwardBatch` then become
one function -- they are two transcriptions of one graph. This is what
vLLM calls chunked prefill. Do not assume llama.cpp's fixed-width slots are the
right shape; jitllm's batch is a property of the State.

## 5. KV cache as an interface

**Done, apart from a remote backend and 5a.**

The cache is paged (`engine/model/kvpage.go`): pages sized per layer from the
kernels, living on whichever device holds their layer, growing a page at a time
instead of reserving the whole context. A page leaves a tier through the
user's interface -- `Get(cacheId, layer, index, io.Writer)` plus `Set`/`Drop`
-- with three backends: `NoStore` (the default), `MemStore` and `FileStore`.

Prefix reuse is built on it and keyed by content: a page is stored under a
hash of the tokens that produced it, so sessions that share a system prompt
reuse it without knowing about each other (`jitllm run -kv-cache DIR`: 5.8x on
a repeated 1181-token prompt). `SetCacheKey` names a namespace, which is also
the tenancy dial.

Left:
- a remote backend, which belongs outside core;
- restoring a cached prefix onto device blocks (`PrefillCached` refuses when
  blocks are placed, rather than attending over history the card never saw).

The backends, the adversarial audit of the cache and the prefix
design are recorded in `docs/engineering-history/placement.md`.

### 5a. The RAM prefill buffer (not started)

The one piece of the paged cache deliberately left unbuilt: when the first
layers run on a GPU and VRAM is short, a RAM prefill buffer lets the transfer be
batched rather than paid page by page.

It is not a `KVStore`: a store is where a page lives when no tier holds it. The
prefill buffer is a tier on the path between the host and the card: prefill
seals many pages per layer at once, and the point is to batch them into one
bulk transfer so the card need not hold the whole cache while the prompt runs.

Already true and not to be re-derived: sealing is the trigger
(`kvCache.seal(pos)` offers each page once as the token moves past it);
`State.migrateKV` already gathers pages into one run for
`nn.LayerDevice.MigrateKV` and scatters them back; and the device is the
constrained side, since the host commits pages lazily.

**Its design is open.** Whether the buffer is a tier with its own residency or
a batching window over the existing migrate path is undecided, and that choice
comes before any code.

## 6. Swarm orchestration over the network

**Not started.**

Distributed inference across machines. The interface belongs in core with no
dependency on libp2p or any transport; an implementation lives outside, the
dependency pointing inward -- the shape `nn.Device` already has (declared in
`engine/nn`, implemented in `jit/gpu`).

What that protects: `go.mod` has one require (goffi) and libp2p is about 150
modules; and there is no cgo, since goffi reaches the drivers through dlopen.

## 7. Pagination as an interface

**Half done.** The engine no longer maps weights: `format/jlm` reads pages with
`pread` (O_DIRECT where the filesystem allows) into frames it owns and evicts
itself; GGUF's mapping is the converter's alone.

Left: `jlm.Open` takes a path and reads an `*os.File`. A remote source (an
object store, a peer) needs the container to read through an interface shaped
like `io.ReaderAt` instead.

---

The common thread in 5, 6 and 7 is one architectural move: an interface in
core, implementations outside, the dependency pointing inward so `cmd/jitllm`
links none of them. `nn.Device` is the precedent.

## 8. MLX 4-bit checkpoints

Convert and run MLX's quantized safetensors (mlx-community and `mlx_lm.convert`
output). The format is affine group quantization: 4-bit codes packed eight to a
`uint32`, one scale and one bias per group of 64 weights (32 and 128 also
exist), `w = scale*q + bias`, described by config.json's `quantization` block.

- **Converter:** the HF safetensors path (`convert/safetensors.go`,
  `hfArchTable`) already reads MLX's tensor names; it needs the `quantization`
  config, each `weight`/`scales`/`biases` triple joined into one tensor, MLX's
  nibble order, and quantized embeddings and heads.
- **Format:** a native group-64 type (one scale/bias pair shared by two
  32-element activation blocks, +0% bytes) rather than a Q4_1-style repack,
  which stores every pair twice and reads ~11% more bytes on a
  bandwidth-bound decode. A new quant type: reference dequantizer first
  (RULE 8), then kernels on every tier. Scales are often bf16; storing them
  as f16 must be exact, or refused at conversion.
- **Oracle:** `mlx_lm` on the M4, teacher-forced against MLX's own logits for
  the same checkpoint (RULE 7m -- llama.cpp does not read MLX).
- **Scope:** 4-bit at group 64 first, 32/128 if cheap, then 8-bit. MLX's
  mxfp4 mode maps onto the existing MXFP4.

## 9. Ideas from Strata (single-model MoE engine for gaming GPUs)

Strata (github.com/Niko1221/Strata, C++/CUDA/HIP) runs one ~125B hybrid MoE,
Qwen3.8-Flash-Next, on a 12-16 GB card plus RAM, and claims 53-93 tok/s
decode on an RTX 5070 (not measured here). Our streamed Qwen3-Next-80B on a
4 GB card reads 0.78 tok/s; the gap is design, not kernels. In value order:

1. **The expert as the unit of placement, computed where it lives.** The GPU
   computes the experts cached in VRAM while the CPU computes the uncached
   ones in place from pinned RAM, concurrently, and a tunable share of misses
   (`pcie_frac`, 0.2-0.55) is fetched over PCIe instead
   (`src/kernels/cpu/expert.cpp`, `src/program/generate.cpp`). We place whole
   blocks on one side, or stream every routed expert to the card.
2. **An adaptive expert cache on the device.** Every 4 steps it swaps up to 96
   of the most-routed missing experts in for the least-routed resident ones,
   by decayed counts, while the CPU computes the outgoing ones so no token
   stalls (`adapt_every`/`adapt_swaps`, `resident_stage_swaps`); an offline
   routing profile seeds it. The device-side twin of our host expert LRU.
3. **Speculative decoding with the model's own MTP head:** up to 3 drafted
   tokens checked in one pass, 2.4-3.2 tokens a pass, 1.6-1.8x decode. Our
   converter used to set MTP blocks aside (`convert/qwen35.go`). Built:
   `jitllm run|speed -spec mtp`, with an adaptive draft count that stops
   speculating where no count wins (`engine/model/spec.go`).
4. **Recurrent-state checkpoints for conversations:** at turn boundaries,
   every 16k prompt tokens and at the end of the system prompt, six kept
   (~118 MB each) -- where a hybrid now restores at one point per prompt.
5. Smaller: prompt-lookup drafting enabled only where measured acceptance
   pays; the next layer's experts streamed over PCIe during this layer's
   attention in prefill; long-context KV keeping only its most-read part on
   the device; a multi-GPU layer split from a fitted cost model.

Not applicable: per-model kernel specialisation, the ggml kernels, a
single-user server, and its "speed projection" (a refusal-removing control
vector, not an optimisation).

## 10. Ideas from "AI as a compiler" (arXiv 2609.36800)

An LLM agent lowers Triton kernels straight to PTX, accepted by randomized
differential testing, and reaches 0.83-3.34x autotuned Triton on Ada, Hopper
and Blackwell at $1-15 of search per kernel. That search is offline and
per-target, so it cannot replace a JIT that emits at load in milliseconds
(RULE 8); what transfers is two transformations it found. Their matvec gained
0.98-1.15x and RMSNorm/RoPE at most 4-5%, which agrees with decode being
bound by bytes here. Their numbers are mostly single runs with best-of-run
headlines, and the kernels a formal checker could prove equivalent lost most
of the gain (GEMM 1.03x -> 0.24x). Neither idea below is measured here.

1. **A thread owns a whole softmax row in device attention.** Their
   FlashAttention kernel (1.37x) gives each thread one complete score row in
   the tensor-core layout, so the row's max and sum are register-local and
   no cross-lane shuffle is issued. Ours reduces across lanes, which is part
   of why some kernels need a guaranteed 32-lane subgroup. A row-per-thread
   form needs no subgroup at all, so it may be both the portable form
   (llvmpipe, Intel) and a faster one. Measure against the current kernel on
   CUDA and Metal before choosing.
2. **Quantized weights decoded straight into tensor-core operands.** Their
   1-bit BitDelta GEMM (3.34x) builds the fp16 operands in registers -- a bit
   permutation plus one `lop3` that sets sign bits of a packed constant --
   instead of unpack, convert, store to shared memory and reload. Check
   whether the device prefill GEMM (`jit/gpu/kernels/mma.go`) stages
   unpacked weights through shared memory; if it does, unpacking Q4_K/Q6_K
   directly into the mma fragments is a prefill lever, not a decode one.

## 11. Decision models

**Not started; scope to be defined.** Open-weight "decision" models, as opposed
to text generators. The kind is
not settled yet: candidates are JEPA-style world models (encoder plus predictor in
embedding space), rerankers and classifiers that output a score or a label,
reward and judge models, and action models (Decision Transformer, VLAs). Each
would ride the same principles as every other model here.

## 12. Ideas from ds4 (DwarfStar)

**Reviewed; not implemented or measured here.** Source:
[antirez/ds4 at 0aaea5a](https://github.com/antirez/ds4/tree/0aaea5a238fb41a35106a551e73c8409dfb751ac).

1. **Expert eviction by decaying routing frequency.** Its Metal cache counts
   selections including misses, ages the counts, and evicts by lowest frequency
   with recency as the tie-breaker. Counting misses avoids penalizing experts
   evicted before they can register a second hit. Compare this with our host
   expert LRU by replaying real routing traces at identical byte budgets,
   including prompt, decode and tool-result transitions. Measure missed bytes
   before runtime integration. Related to item 9's device-cache proposal;
   this experiment concerns the existing host pager.
2. **Checkpoint-matched continuation fixtures.** Its
   [quality tooling](https://github.com/antirez/ds4/tree/0aaea5a238fb41a35106a551e73c8409dfb751ac/gguf-tools/quality-testing)
   scores fixed hosted-model continuations by token negative log likelihood,
   including long-context cases. Extend our existing teacher-forced checks
   with independent reference coverage when adding relevant model families.
   Hosted continuations complement numerical oracles; they do not establish
   exact correctness.
3. **Reference for future architectures.** Revisit its model implementations
   and fixtures when a corresponding architecture is selected for support.
   Kernel algorithms may inform generated implementations; copied kernels,
   mapped-file overflow and its default distribution-changing speculative
   sampling do not fit our constraints. Selected-expert reads, I/O overlap,
   prefix reuse and adaptive speculation already exist here.

## 13. PII detection

**Not started.** Find and redact personal data in text as an engine feature: a
token-classification (NER-style) head on an encoder -- BERT/ModernBERT PII
taggers -- returning labelled spans, plus a redaction call that masks or replaces
them. It runs the encoder path the embedder already uses (`engine/model/encoder.go`)
with a per-token classifier readout in place of pooling, and the server can apply
it to a request before generation and to the output after. Needs ModernBERT
(item 2) for the current models; gated against transformers' token-classification
pipeline on the published checkpoints.

## 14. Semantic routing

**Not started.** Choose where a request goes by what it means: embed the request
(the embedder that exists, or a classifier head), compare it against each route's
examples or centroid, and send it to the model, instance or tool that route names
-- the server is mostly a multi-model server, so the natural target is one of its
loaded models. Routes and thresholds are the caller's (options, not built-in
policy), a request that matches nothing falls to a default, and the route taken
is reported with its score. Builds on `Model.NewEmbedder` and the server's
per-model instances; a classifier-based router shares item 13's readout.

## 15. Full Kimi-K3 from disk

**Not started.** Run the whole 2.78T Kimi-K3 on one machine,
paged off disk with a memory budget far below the model, the way
kimi-k3-in-c does (1.56 TB checkpoint, about 8 GB resident, about 26 s a token
on a laptop), and compare rate and peak memory on the same box. jitllm's K3 is
verified on trained weights at small scale (Kimi-K3-0.40B and its Kazakh
continuation, f32 rounding against Moonshot's own code, `TestKimiK3RealMatchesReference`);
the full model has never been opened.

What it needs:
- **Disk.** About 1.5 TB for the checkpoint plus the container; converting
  straight from a GGUF and dropping it keeps the peak near 1.5 TB rather than 3.
- **A format the converter reads.** The official release stores its experts as
  compressed-tensors MXFP4 safetensors, which the safetensors path does not
  decode yet; the GGUF route works for Q4_K-class files (the 2-bit quants are not
  dequantable).
- **q, k and v as three projections in a KDA block.** A mixed-quant GGUF stores
  q at Q4_K beside k and v at Q8_0; the converter joins them as one Q8_0
  projection, which grows q from about 4.5 to 8.5 bits a weight in every
  linear block. Running the three as separate projections keeps the file's
  bytes and changes every tier's linear block.
- **Scale checks** nobody has run at this size: page-table size, open time,
  conversion time, and the expert pager under a budget a small fraction of the
  model.
