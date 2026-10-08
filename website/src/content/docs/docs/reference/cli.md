---
title: CLI reference
description: Every jitllm and jitllmd subcommand and flag.
---

Flags always go **before** the model path; anything after it is the prompt or text. Every subcommand prints its own flags with `-h`.

Every size, on `jitllm` and `jitllmd` alike, is a byte count or a number with a unit: `4294967296`, `4G`, `512M`.

## jitllm

| Command | What it does |
|---|---|
| `jitllm hardware` | the CPU tier, every GPU each backend can open, and the spendable memory |
| `jitllm library [-refs]` | the catalog `convert` fetches by name; `-refs` prints sizes and `hf://` references for scripts |
| `jitllm convert [-o DIR] [-q8] SOURCE [MMPROJ] OUT.jlm` | write a container from a GGUF, a Hugging Face directory or URL, or a catalog name. See [Convert models](/docs/guides/convert/). |
| `jitllm run [flags] MODEL.jlm PROMPT...` | generate |
| `jitllm embed [-lines] [-ids] MODEL.jlm TEXT...` | print an embedding model's normalised vector as JSON |
| `jitllm speed [flags] MODEL.jlm` | prompt and generation rates, warmed, as `llama-bench` measures them; or time to first token, or sessions decoded together |
| `jitllm batch [flags] MODEL.jlm PROMPT...` | decode many sequences together on one device and report the aggregate rate |
| `jitllm verify [flags] MODEL.jlm` | run on the CPU and a device and report where their logits differ |
| `jitllm tokenize [-tokenizer FILE] MODEL TEXT...` | print token ids |
| `jitllm info FILE.gguf` | a GGUF's keys and tensors, before converting it |
| `jitllm asm Q4_0\|Q8_0\|Q4_K[x4]` | disassemble the x86-64 kernel generated for a format on this machine |
| `jitllm version` | the build's version |

### run

| Flag | Default | Meaning |
|---|---|---|
| `-n N` | until the end | tokens to generate; by default the model runs until it ends its reply or fills its context |
| `-chat` | off | apply the model's chat template |
| `-system TEXT` | | a system message, with `-chat` |
| `-image FILE` | | an image for a vision model |
| `-devices SPEC` | `auto` | where to run; see [the device grammar](/docs/guides/devices/#the-device-grammar) |
| `-placement MAP` | | where named blocks and the head run, with `!` to pin and `~` to stream; see [Placement maps](/docs/guides/devices/#placement-maps) |
| `-placement-strict` | off | fail when a `-placement` entry cannot be met, instead of placing that block elsewhere |
| `-gpu-layers N` | -1 | cap the blocks offered to devices; -1 fits as many as the card holds |
| `-vram SIZE` | 0 | device weight budget; 0 asks each device what is free and keeps an eighth of it, at least 256 MiB |
| `-maxmem SIZE` | automatic | host weight budget; see [Memory and paging](/docs/guides/memory/#the-page-budget) |
| `-gpu-grow` | off | start on the CPU and move blocks to the device while generating |
| `-relocate` | on | hand the device's last block to the CPU when the context cannot grow on the device. On wherever `-devices` admits the host; `-relocate=false` turns it off |
| `-tune-seam` | off | measure whether fewer device blocks are faster, and move the seam |
| `-no-gpu-fallback` | off | fail instead of finishing on the CPU when the device breaks |
| `-kv-cache DIR` | | reuse KV pages of earlier prompts with a shared prefix, and spill history there under `-kv-budget` |
| `-kv-cache-max SIZE` | 8 GiB | cache size before least-recently-used eviction; 0 is unbounded |
| `-kv-budget SIZE` | 0 | KV history held in memory; older pages spill to `-kv-cache` and come back when read. 0 is unbounded |
| `-depth N` | 0 | pad the prompt to N tokens, to measure decode at that context length |
| `-tokenizer FILE` | | a `tokenizer.json` whose pre-tokenizer replaces the model's |
| `-spec mtp` | | speculative decoding: draft with the model's own multi-token-prediction block and verify the drafts in one pass; see [Speculative decoding](/docs/guides/run/#speculative-decoding) |
| `-spec-k N` | 0 | tokens drafted a round; 0 chooses each round from the measured acceptance and timing |
| `-spec-min-p F` | 0 | stop a round's drafting at the first draft below this probability |
| `-spec-rollback MODE` | `auto` | how a hybrid's recurrent state is taken back after a rejected draft: `auto`, `rows` or `replay` |
| `-temp`, `-top-k`, `-top-p`, `-min-p`, `-repeat-penalty`, `-repeat-last-n`, `-seed` | greedy | sampling, named as in llama.cpp; see [Sampling](/docs/guides/run/#sampling) |
| `-cpuprofile FILE` | | write a Go CPU profile |

### speed

| Flag | Default | Meaning |
|---|---|---|
| `-devices SPEC` | `auto` | as for `run` |
| `-p N` | 512 | prompt tokens |
| `-n N` | 128 | generated tokens |
| `-r N` | 3 | timed repetitions, after one untimed warm-up |
| `-vram SIZE` | 0 | device weight budget |
| `-ttft` | off | time to first token instead: cold (opening the model to the first token) and warm (a fresh session tokenizing, prefilling and sampling a `-p`-token prompt, `-r` times) |
| `-sessions N` | | decode N separate sessions `-n` tokens each, one after another and then together, interleaved over `-r` rounds |
| `-gcstats` | off | what the Go collector did during each timed phase: allocations a token, cycles, pauses, GC CPU |
| `-cpuprofile FILE`, `-tgprofile FILE` | | a CPU profile of the timed prompts, or of the timed decodes |
| `-memprofile PREFIX` | | allocation profiles around the first timed prompt and decode |
| `-spec`, `-spec-k`, `-spec-min-p`, `-spec-rollback` | | speculative decoding, as for `run` |

### batch

| Flag | Default | Meaning |
|---|---|---|
| `-devices SPEC` | `gpu` | the one device every block and the head go on |
| `-batch N` | 16 | sequences decoded together |
| `-n N` | 128 | tokens per sequence |
| `-logits` | off | read every row's logits back and pick on the host, the path a sampled request takes, instead of the device's argmax |
| `-cpuprofile FILE` | | write a Go CPU profile |

### verify

| Flag | Default | Meaning |
|---|---|---|
| `-devices SPEC` | `auto` | the device to compare with the CPU |
| `-n N` | 32 | tokens |
| `-dlogit F` | 0 | the largest per-logit difference to accept; an argmax flip inside it counts as a tie. 0 does not check |
| `-migrate` | off | move the CPU/GPU seam during the comparison |
| `-prefill N` | 0 | also compare the batched prefill of N tokens |
| `-verify` | off | recompute every served matrix product on the host and report disagreements |
| `-vram SIZE`, `-maxmem SIZE` | automatic | device and host budgets |
| `-ab ARM` | | run a paired A/B of one engine choice instead (`split`, `submit`, `head`, `graph`, `softmax`, `ropetab`, `attnchunk`, `attnpair`), with `-rounds N` and `-depth N` |

### convert, embed, library, tokenize

| Command | Flag | Meaning |
|---|---|---|
| `convert` | `-o DIR` | where a downloaded GGUF lands (default `JITLLM_MODELS`) |
| | `-q8` | safetensors only: store every weight matrix as Q8_0 |
| `embed` | `-lines` | embed each line of stdin, loading the model once |
| | `-ids` | print the token ids to stderr |
| `library` | `-refs` | name, size in bytes and `hf://` references, one per line |
| `tokenize` | `-tokenizer FILE` | as for `run` |

## jitllmd

| Command | What it does |
|---|---|
| `jitllmd serve [flags]` | run the server |
| `jitllmd run [flags] PROMPT...` | stream a generation from a running server |
| `jitllmd models` | list, load and unload models |
| `jitllmd devices` | the server's hardware and spendable memory |
| `jitllmd sessions` | create, list, reset and close sessions |
| `jitllmd place` | read and move a session's blocks, tune its seam, and re-budget a model's pager |
| `jitllmd stats` | engine counters, once or streamed |

Every client command takes `-addr` (`host:port`, `:port` or a full URL; default `localhost:8080`).

### serve

| Flag | Default | Meaning |
|---|---|---|
| `-addr` | `:8080` | listen address |
| `-models DIR` | a developer path | directory scanned by the model list and used to resolve a bare model name. Always pass it. |
| `-load FILE` | | a `.jlm` to load at startup, absolute or relative to `-models` |
| `-id NAME` | derived | the model name clients send |
| `-devices SPEC` | CPU only | devices for `-load`, in the [device grammar](/docs/guides/devices/#the-device-grammar) |
| `-maxmem SIZE` | engine's own | host page budget for `-load`, e.g. `8G` |
| `-gpu-layers N` | -1 | at most N blocks on a device; -1 is as many as fit |
| `-sessions N` | 1 | concurrent sessions each linear device block reserves a recurrent state for; attention history is paged and reserves nothing |
| `-max-seq N` | model's context | default KV capacity per session, in positions |
| `-max-batch N` | 0 | requests of one device model that decode as rows of one step; 0 is the widest step the device runs, 1 turns batching off |
| `-prompt-chunk N` | 0 | prompt tokens a joining request feeds into one shared step; 0 is one device prefill chunk |
| `-joint-steps MODE` | `auto` | `auto` times a joint step against the rows one after another and runs the faster; `always` or `never` |
| `-kv-f16` | engine's own | the KV cache width of every load that names none: `-kv-f16` binary16, `-kv-f16=false` f32 |
| `-kv-cache DIR` | | where sessions created with `prompt_cache` keep their prompt prefixes; without it such a session is refused |
| `-kv-cache-max SIZE` | `8G` | what `-kv-cache` may occupy before its least recently used pages go; 0 is unbounded |
| `-version S` | `dev` | the version `GetServerInfo` reports |

### run

| Flag | Meaning |
|---|---|
| `-model ID` | generate in an ephemeral session on this model |
| `-session ID` | generate in an existing session, keeping its history |
| `-continue` | continue from the session's position instead of prefilling from scratch |
| `-chat`, `-system TEXT` | apply the model's chat template, with a system message |
| `-n N` | tokens to generate (default: until the model ends its reply or fills the session's context) |
| `-stop A,B` | stop strings |
| `-temp`, `-top-k`, `-top-p`, `-min-p`, `-repeat-penalty`, `-repeat-last-n`, `-seed` | sampling, as for `jitllm run` |
| `-queue-timeout MS` | how long to wait for a device another session holds; 0 waits indefinitely |
| `-echo` | have the server send the prompt's tokens back first |
| `-ids` | print the generated token ids to stderr |
| `-quiet` | no started/finished lines on stderr |

### models

| Flag | Meaning |
|---|---|
| (none) | list the model directory and what is loaded; `-loaded` lists only loaded models, `-dir` scans another directory |
| `-load FILE` | load a container, with `-id`, `-devices`, `-gpu-layers`, `-maxmem` as for `serve`, and `-kv-f16` to force an f16 KV cache |
| `-unload ID` | unload a model; `-force` closes sessions holding it instead of refusing |

### sessions

| Flag | Meaning |
|---|---|
| (none) | list sessions; `-model` filters |
| `-new` | create a session on `-model`, with `-id`, `-devices`, `-gpu-layers`, `-max-seq`, `-gpu-grow`, `-keep-off-host`, `-prompt-cache` and `-cache-key` |
| `-reset ID` | back to position 0 |
| `-close ID` | close it |
| `-queue DEVICE` | the queue waiting on a device |

### place

| Flag | Meaning |
|---|---|
| `-session ID` | report where each of the session's blocks runs |
| `-devices SPEC` | re-place the session's blocks on these devices (they must be ones the model was loaded with); empty brings every block home |
| `-gpu-layers N` | at most N blocks on a device |
| `-head-on-device` | place the output projection on a device too |
| `-move FIRST:LAST -to WHERE` | relocate a block range to `host` or a device |
| `-relocate` | adopt one block per token as device memory frees |
| `-tune-seam` | measure whether fewer device blocks are faster and move there, with `-tokens`, `-rounds` and `-warmup` |
| `-model ID` | report the model's pager residency; with `-maxmem SIZE`, re-budget it |

### stats

| Flag | Meaning |
|---|---|
| `-watch` | stream a snapshot every `-interval` milliseconds (default 1000) |
| `-model ID`, `-session ID` | only this model or session |

## Environment

Read by `jitllm` only; `jitllmd` and the Go packages read no environment variables.

| Variable | Effect |
|---|---|
| `JITLLM_MODELS` | default directory for downloaded models |
| `HF_TOKEN` | token for private or gated Hugging Face repositories (`HUGGING_FACE_HUB_TOKEN` and `~/.cache/huggingface/token` also work) |
| `JITLLM_GPU_STREAM=1` | place mixture blocks without their expert banks and upload only the chosen experts each token |
| `JITLLM_ARENA=BYTES` | bound the host copy that streamed blocks are uploaded from; unset, a streaming placement sizes it |
| `JITLLM_KV_F16=0` or `1` | force the KV cache's width instead of the per-CPU-tier default (f16 on AVX2, f32 elsewhere) |
| `JITLLM_CORESET`, `JITLLM_CORES=N` | which cores decode uses (`p` by default) and how many of them |
| `JITLLM_NUMA=off` | leave weight memory to the kernel's first-touch policy instead of interleaving across NUMA nodes |
| `JITLLM_HUGEPAGES=0` | keep weight memory on the kernel's default page size |
| `JITLLM_OFFHEAP=0` | put weight memory back on the Go heap (a control for measurements) |
| `GOMAXPROCS` | when set, overrides the thread count jitllm aligns to its decode cores |

The rest of the `JITLLM_*` names `cmd/jitllm` reads are measurement switches for engine development. The Go packages take options instead; see [Embed it in Go](/docs/guides/go/).
