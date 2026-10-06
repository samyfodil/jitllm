# Open questions and shelved levers — the work list

This file is a WORK LIST, not a rule book. `AGENTS.md` governs; RULE 6 decides
what happens to every entry here: a number decides ORDER, never admission, and
only a clear regression refuses a change. An entry leaves this file when it is
closed, and its record goes to the matching evidence file under
`docs/engineering-history/`.

Before working an entry, read its evidence: several were open when written down
and the code has moved since (RULE 12). Paths below are under
`docs/engineering-history/` unless they say otherwise.

## The list

| open item | what is recorded | evidence |
|---|---|---|
| native Q3_K packing | the one format still widened (+34.5% over its GGUF); native needs a 2-bit primary plane, a third emitter shape whose per-byte cost has never been measured -- one `nn.TestPerByteCost` row would settle whether it pays | cpu-kernels.md "Native narrow packing" |
| AVX-512 on the Xeon D | worth 3.1% of a decode token there; admitted and scheduled behind larger levers, needs an EVEX encoder and the 32-register allocation | cpu-kernels.md "The Xeon D-2123IT" |
| Metal decode's unattributed term | 93.7% of an M4 decode token attributed to measured causes and ~0.56 ms not; no single term is over 5% of the token | gpu-kernels.md "Metal decode against the oracles" |
| fusion on Metal | dispatch overhead is 6.2% of a dense token, and merging adjacent elementwise runs can recover ~1.4% of it; a mixture layer has far more to gain | same |
| the in-group reduction as default | forcing it is ~0.5% and makes the submission deterministic; whether it should be the default is unsettled | same |
| pinning the split set on Metal | each process measures its own splits, which costs ~0.85 points of dispersion; whether pinning the best set beats tuning is unmeasured | same |
| tinyllama's Metal ceiling | ~120 tok/s regardless of traffic, unexplained | same |
| the hybrid CPU+GPU row | the split configuration trails llama.cpp by ~10% at matched card memory while each half runs at full speed: a per-token overhead in the seam, unresolved | perf/current.md "Boards that stood in AGENTS.md RULE 1" |
| the coherence gate on a mixture | it may discard an engine difference rather than a clock accident on MoE models (`ensureExperts`' synchronous reads drive the clock differently); unmeasured, the gate stays on | scheduling-and-measurement.md "The 0.90 and 0.70 readings" |
| a fair over-committed row | both arms under one shrunken cgroup; not yet taken | placement.md "Qwen3-Next-80B on the card" |
| `-tune-seam` on a slower second device | whether it recovers the placement an iGPU makes worse is untested | placement.md "Scope as it stood in AGENTS.md" |
| the iGPU's proportional share | a bare `-devices gpu` spreads a model the card alone cannot hold over the card and the iGPU in proportion to their room, so the iGPU takes blocks while the card has room (gemma-3-4b: 15 on the RTX and 19 on the Iris Xe, where the RTX alone takes 20); whether the card should fill first is unmeasured | placement.md 16a |
| what an iGPU keeps with no block | the tier keeps a device's block scratch and the head after its blocks come home and after the session detaches (~31 MiB of host memory on the Iris Xe for stories15M) | same |
| sessions per device | concurrent sessions on one device, and how many a card holds at a given context, are bounded by pages and unmeasured | same |
| a mid-block fault injector | `injectedFail()` acts on whole calls; a `phase@block` injector is owed for the mid-block failure paths | placement.md "Qwen3-Next-80B on the card" |
| hybrid restore points during prefill | a hybrid restores at one point per prompt; capturing more during prefill is priced and unbuilt | placement.md "The prefix cache" |
| a dense lead block's page size | it sets the base page for every block (15.4% padding on Moonlight); a fourth array or charging filled bytes would remove it | placement.md "No mmap of files" |
| reading the next layer ahead | the over-committed case wants the next layer's reads issued before this layer's compute; benchmark against neurostream on bytes read per token | below, "A page is a whole block when the model does not fit" |
| unprofiled subsystems | the vision projector, prefill's pack phase and the device submission path | cpu-kernels.md "Profiling the subsystems nobody had profiled" |
| a batched arm64 mixture that pays | on the M4 the batched dispatch over the fused kernel costs 24%; it would have to drive the tiled kernel over the experts' rows | same |
| Kimi-K3 against a trained checkpoint | gated only on a randomised fixture; the trained 0.40B test model (inference-optimization/Kimi-K3-0.40B and its F16 GGUF) is the cheap first real check, then the M4 and Metal gates | model-correctness.md "kimi-k3" |
| Kimi-K3 from safetensors, its vision tower and its prediction blocks | published K3 experts are compressed-tensors MXFP4, which the HF path does not read; the tower and the MTP blocks are not built | same |
| the shelved levers | small levers re-opened under RULE 6 | below |
| the smaller items | SPIR-V Tok-invariance, the M4 MoE rows, `centerMinTok`, the amd64 f16 KV rate, the arm64 integer fold for Q4_K/Q5_K, `emitA64KScales`' mask immediates | below |

## Levers shelved for being small, not for a regression

RULE 6 changed after these were shelved, so every earlier shelving needed
re-reading. An audit of them found these were set aside for being SMALL and are
candidates again, in rough order of what the evidence says they are worth:

| candidate | shelved on | status |
|---|---|---|
| whole-layer fusion | "Nothing clears 5%" | **reopened** — 14.18%/14.81% on MoE; dense tinyllama crosses parity at 1.0076 |
| batched CPU MoE dispatch | never proposed | **BUILT** (`nn.MatVecPackedMulti` + `MatVecPackedGather`): olmoe 906 -> **234** regions/token, kernel CPU -10%, rate UNRESOLVED and the 1.22x prediction did not hold. Dispatch is now at dense density, so what is left is the SERIAL stretch. See "Profile the subsystems nobody has profiled" |
| q/k/v fused matvec | 1.0072x "REJECTED" | **reopened** — separate the residency complexity from the size verdict |
| Metal indirect command buffer | "a 3% ceiling ... not worth it now" | **reopened** — ordering semantics need investigating, not assuming |
| f32 softmax at depth | "~2% of the token" | **reopened** |
| accumulator rotation (arm64) | "+1%, i.e. nothing" | **reopened** |
| interleaved-emitter prefetch | "already read 94-95% of the wall" | **reopened** — untested |
| sum-free quantizer | "the SIZE is what kills it" | **reopened**, low priority — ~0.4%, an estimate of an estimate |
| baking scalar constants | "(0.04%)" | **reopened**, lowest priority |

**Tried or priced, and each came out a REGRESSION** (measured, or predicted where
said): pack8-plus-unroll (31.7 KB against a 32 KB shared L1i), the GPU integer
fold (127->173 registers, 25% loss), head-major KV for decode (measured
whole-model loss; prefill untested), online softmax (111x on ascending input),
fused GPU elementwise kernels (64 threads against 2048), and:

the spread-row decode tile on the device (`docs/engineering-history/gpu-kernels.md`,
"The decode row tile on the device, spread rows"), and the adjacent-row tile
after it ("THE DECODE ROW TILE, RE-BUILT AS ADJACENT ROWS").

## A page is a whole block when the model does not fit

- **★ A jlm PAGE IS A WHOLE BLOCK, WHICH COSTS NOTHING UNTIL THE MODEL DOES NOT
  FIT, AND THEN COSTS 15x.**

  ★ **THE SCOPE IS THE WHOLE POINT AND THE FIRST VERSION OF THIS ENTRY GOT IT
  WRONG.** Stated by the user: *a model that fits would not need
  pages to be kicked out, so a token would not need any read.* That is the
  design working -- "a budget that holds every block simply never evicts" -- and
  it means page-ins per token is ZERO for every model that fits and the number
  below does not exist. What follows is a property of the OVER-COMMITTED case,
  not of the pager, and it must not be quoted as though it described loading a
  model that fits.

  Measured on Qwen3-Next-80B (46.5 GiB of weights, 48 blocks of 1049.7 MiB, 10
  of 512 experts routed) against a 12.6 GiB budget:

      the token NEEDS       2.56 GiB
      the pager READS      37.8 GiB   (36 faults x 1.05 GiB at 12 of 48 resident)
      amplification          ~15x

  ★ **AND llama.cpp IS NOT A FAIR ARM FOR THIS ROW, WHICH RULE 1 ALREADY SAYS
  AND THIS SESSION NEARLY BROKE.** It runs the same model on this box at 3.51
  tok/s on the CPU -- but it has no pager: it mmaps and delegates residency to
  the kernel, which faults at 4 KiB, so only the routed experts are ever
  touched. "llama.cpp 3.51 against jitllm slower" is therefore not an engine
  comparison; it is the KERNEL's 4 KiB demand paging against jitllm's 1 GiB
  block paging, on a model neither holds. The falsifiable form is BYTES, above,
  and it stands without llama.cpp in the row.

  jlm reads rather than maps for a measured reason -- ReadAt x16 is 4.73 GiB/s
  against an mmap fault's 2.22, because the device needs several requests
  outstanding and one fault cannot produce them (format/jlm/read.go). None of this
  changes that.

  **Do not read the 80B's rate as a statement about the delta-net kernels.**
  They are separately verified against llama.cpp's own intermediates, which is
  what an oracle is for and which memory does not touch (see RULE 7's qwen3next
  entry).

  ★ **AND THE ROUTING PAGES MUST BE PINNED, which is the enabling condition and
  not an optimisation.** Stated by the user: the pages routing
  depends on are never evicted, though they stay RELOCATABLE (host <-> device,
  or between devices). You cannot know which experts to fetch until the router
  has run, so an evictable router puts a synchronous fault on the critical path
  of every layer and no prefetch schedule can exist. The cost is a
  **loadability bar** -- the pinned set must fit or the model cannot load at
  all. Small here (512x2048 F32 is 4 MiB a layer, 192 MiB for 48) and real.

  ★ **AND THE SECOND HALF IS SCHEDULING, WHICH `neurostream` ALREADY PRICED.**
  It streams GPT-OSS-120B off NVMe onto an 8 GB card: 4 of 128 experts routed,
  4.2 GB/token read, 6.6% of the file, 61 GB never in memory. Its first working
  version stalled **76 seconds per token** at a 66% prefetch hit rate with the
  pipeline correctly built -- it drained 12 times per layer, 432 times per
  token, and effective queue depth never rose above 3, so a 2 GB/s drive
  delivered its queue-depth-1 rate. One change, planning the prefetch once per
  LAYER instead of once per projection: **stall 76.3s -> 0.8s, hit rate 66.3% ->
  99.5%.** Their sentence is the one to keep: *"too big to fit turned out to be
  a scheduling problem, not a memory one."*

  `jlm.ReadThreads` already knows depth is the rate on an NVMe; what is missing
  is issuing the NEXT layer's reads before the current layer's compute. **And
  jitllm should be benchmarked against it** (the user asked) -- not
  on tok/s, where a Python streamer at 0.46-0.73 tok/s is not the interesting
  arm, but on BYTES READ PER TOKEN, which is the quantity jitllm is 15x wrong
  on.

## Smaller open items

- Q4_K/Q5_K are not Tok-invariant on SPIR-V — ~40% of elements differ at 6.1e-05,
  spirv only, ptx and msl exact. Bounded at NMSE 1e-8, **undiagnosed**; it is the
  min term's fold.
- The M4's OLMoE row is **0.8456 as of the older M4 board's re-take, not 0.7280** --
  that older number is the pre-elementwise board and should not be quoted. It is
  still the Mac's weakest row alongside Qwen3-MoE at 0.8392, and Linux wins the
  same Qwen3-MoE at 1.0289: a 19-point cross-host swing on one model that nothing
  explains yet. The MoE gap is now a MAC problem specifically, not a jitllm-wide
  one.

- `centerMinTok = 8` is measured on CUDA (0.981 at Tok=1, a real regression) and
  **indistinguishable from zero on Metal** at this harness's 4.8% resolution.
- The f16 KV cache now DEFAULTS ON for amd64 and OFF for arm64, chosen by
  `nn.KVWidthPaysOff` from the emitted kernel sizes -- and **the amd64 rate is
  unmeasured**, because the box was power-clamped all night. The rule is a
  tie-break (same instructions + half the bytes cannot be slower), not a rate
  prediction, and the arm64 half IS measured at 0.9832 (depth 1536, M4,
  self-control 0.9985). Take the amd64 row on a rested box before trusting the
  default, and note it changes logits by 0.2-0.94 -- inside the engine's own
  CPU-vs-device band, but a change.
- **Q4_K and Q5_K still cannot use the arm64 INTEGER fold** (36.9% of a
  tinyllama prefill's MACs). They now get the other half of it -- the activation
  scale applied once per super-block instead of per sub-block, worth +8.5% and
  +7.7% -- but keeping the DOT integer needs int32 min scales, i.e. a SECOND flag
  in `emitA64KScales` (gemmk_a64.go's intFold comment warns that widening `intSc` applies dmin
  twice or not at all) AND a third accumulator array that does not fit at 2x2.
  Worth the remaining 5 -> 3 instructions per (sub-block, row, vector). They DO
  now have a constant pool -- `dareg` stopped reserving registers it only reads
  in the epilogue, which is worth another +5.8%/+3.1% and got Q6_K +2.8%.
- **`emitA64KScales` rebuilds three mask immediates per ROW per super-block**, so
  the shipping 2x2 spends six MOVIs a super-block on loop invariants. Hoisting
  them needs three more pool slots than any format has spare, so it wants either
  a wider budget (a smaller tile) or the masks folded into the unpack's existing
  constants. `TestGEMMUnpackKonstsAreHoisted` prices them: they are the entire
  gap between its bound and the pool size.
