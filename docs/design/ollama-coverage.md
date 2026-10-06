# Ollama library coverage: what jitllm runs, and the work to run the rest

Research spike. Engine code is unchanged. This is EVIDENCE (see
AGENTS.md's header): it records what was measured and what each gap costs, and
it decides nothing. `docs/design/model-coverage.md` is the earlier survey
against llama.cpp and vLLM. This file asks a narrower question with a concrete
denominator: **every family on https://ollama.com/search, read from the bytes
Ollama actually serves.**

## 0. The headline

★ **RE-CLASSIFIED BY THE HUGGING FACE ARTEFACTS, NOT BY OLLAMA'S
BLOBS.** The first version of this table graded each family by the file the
Ollama registry serves, and Ollama's converter writes its own dialect of GGUF
(`gptoss`/`attn_out`, `glm4moelite` with the MLA width keys swapped, a bare
`ssm_dt`, the legacy fused `ssm_in`, the vision tower inside the text file,
`tokenizer.ggml.pre = "default"`). Every one of those models is also published
on Hugging Face as a llama.cpp GGUF or as safetensors, and the user's decision
is that Ollama-FILE compatibility is not worth effort: those quirks block only
Ollama's copies. So the status below is what jitllm's converter says about the
standard artefact -- read with range requests (header and tensor table only, no
weights) and run through `configOf`, `identify` and `typeOf` themselves, not
through a reading of them. Section 9 lists the artefact behind each row that
moved.

| status | families | pulls (ollama.com counter) |
|---|---|---|
| SUPPORTED: the standard artefact of the default size converts and runs today, every advertised modality | **140** | 745.8M |
| PARTIAL: the text graph exists, but an architecture NAME, a tokenizer entry or a vision projector blocks or limits it (or only some sizes run) | **27** | 111.5M |
| UNSUPPORTED: some text graph feature is missing on every size | **59** | 201.8M |
| CLOUD-ONLY: Ollama serves no local weights (`:cloud` tags only) | **12** | 5.0M |

156 families (837.4M pulls, ~79% of all pulls) can already produce text, where
the Ollama-blob grading said 125 (724.3M): 138 (794.9M) from the re-grading
alone, six more from C9 (the qwen35 block) and twelve from C5 (the Granite
scalars), both landed the same day and verified against llama.cpp (section 4).
rnj-1 moves from PARTIAL to SUPPORTED on a converter fix rather than a new
capability (below). The findings that matter most, in order:

1. **Six families the Ollama grading refused are SUPPORTED from their HF
   artefact**: gpt-oss (13.3M), glm-4.7-flash, qwen3-coder-next, qwen3-next,
   devstral-2 and gpt-oss-safeguard. Seven more produce TEXT and miss only a
   vision projector: gemma3 (40.7M), translategemma, medgemma, medgemma1.5,
   qwen2.5vl, mistral-small3.2 and mistral-small3.1 -- llama.cpp ships their
   towers as a separate mmproj, so the in-file-tower problem does not exist.
2. **What was "C1" is two different things, and only one of them survives.**
   The Ollama-only spellings are dropped. The survivor is llama.cpp's OWN
   architecture names for graphs jitllm already runs: `internlm2` (llama's
   block), `exaone` (llama + NEOX), `qwen3vl`/`qwen3vlmoe` (qwen3/qwen3moe for a
   text prompt) and `mistral3` (llama, plus the temperature C8 when the file
   asks for it). Those names are what the HF GGUFs carry, so they still block
   internlm2, bespoke-minicheck, exaone3.5, exaone-deep, qwen3-vl and
   mistral-medium-3.5.
3. **Embeddings are 12 families and 125M pulls** (`nomic-embed-text` alone is
   the third most-pulled model on the site), and jitllm has no embedding path:
   no bidirectional attention flag, no pooling, no BERT block, no WordPiece
   (jlm.VocabWPM is declared and nothing in tok implements it), no Unigram.

★ **AND ONE ROW PASSED THE CONVERTER AND RAN WRONG; FIXED THE SAME DAY.**
rnj-1's HF GGUF is gemma3 with no `attention.sliding_window` key -- every layer
global -- and `configOf` defaulted gemma3 to a 1024 window on five layers of
six. llama.cpp's loader reads an absent key as no sliding layers, and so does
`configOf` now, along with gemma3's final_logit_softcapping (rnj-1 carries 30),
rope.freq_base_swa and the pattern key. On a ~1700-token prompt the fixed engine
answers word for word what llama.cpp answers, and the 1024-window container
answers something else from the first generated token; teacher-forced 20 of 20
on the short prompts. "config: ok" was not "correct", which is the lesson.

**The work plan, re-scoped:** C9 (the qwen35 block) and C5 (the granite
scalars) are DONE; the C1 survivor (four llama.cpp names, S),
embeddings (C12a/b) and the classic-transformer kit (C6) follow. Vision is a
separate track (Gemma 3 SigLIP, V2; Qwen3-VL, V0+V4). The ranking in section 5
was computed on the Ollama-blob grading and has NOT been recomputed; its C1/C2/C3
rows are stale by the re-scoping in section 4.

## 1. Method, and what the numbers mean

**Enumeration.** `https://ollama.com/library` and `https://ollama.com/search`
(paged through its htmx endpoint, `?page=1..12`, 20 per page) list the same
**238 families**. Name, description, capability badges (tools, thinking,
vision, audio, embedding, cloud), size chips, pull counter and update date were
parsed from that HTML.

**Architecture: read from the served bytes, not from a model card.** For each
family, `https://ollama.com/library/<name>/tags` lists its tags; every tag
without a quantisation suffix (`latest`, `8b`, `30b-a3b`, `it-qat`, ...;
at most 12 a family) was resolved through the registry
(`registry.ollama.ai/v2/library/<name>/manifests/<tag>`) to its config blob and
its `application/vnd.ollama.image.model` (and `.projector`) layers, and the GGUF
**header** of each layer was read with HTTP range requests -- KV section and,
for the first tag of each family, the full tensor-info table. Reading stops at
the end of the header; no weight byte was fetched (1-16 MiB per file, the size
of the vocabulary). Tags whose manifest points at a digest already read are
recorded as aliases. The probe (`probe.py`, `summ.py`, `digest.py`,
`classify.py` in the session scratchpad) is not checked in; this file is the
artefact.

**jitllm's side is read from the code at 1c923ed**, not from docs:

| list | where | contents |
|---|---|---|
| GGUF architectures | `convert.archOf` (convert/config.go) | llama, qwen2, qwen3, qwen3moe, olmoe, gemma, gemma2, gemma3, phi3, qwen3next, qwen2vl, gpt-oss, deepseek2, clip |
| safetensors classes | `hfArchTable` (convert/safetensors.go) | Llama, Qwen2, Qwen3, Qwen3Moe, Olmoe, Mixtral, Gemma, Phi3, DeepseekV2, DeepseekV3, Glm4MoeLite, KimiLinear |
| projectors | `convert.visionOf` | idefics3, mlp, qwen2vl_merger -- and only from a separate `clip` GGUF |
| tokenizer models | `convert.VocabOf` | `llama` (SPM), `gpt2` (BPE); `bert` maps to jlm.VocabWPM but tok has no WordPiece; `t5`, `gemma4`, `rwkv`, `plamo2` are refused |
| pre-tokenizers | `pretok.Table` (67 names) + tok/bpe.go families | missing: `chatglm-bpe`, `llama4`, `gemma4`, `jina-v2-en`, and every file that says `default` gets llama3's split |
| weight types | `quant.Dequantable` / `jlm.TypeOf` | F32 F16 BF16 Q4_0 Q5_0 Q8_0 Q3_K Q4_K Q5_K Q6_K MXFP4 |
| tensor names | `convert/roles.go` | llama.cpp spellings only; an unknown name is a refusal of the file |

**Status rules.** SUPPORTED means the family's default tag has an architecture
in the list, every tensor name has a role, the tokenizer resolves, and every
advertised modality has its tower and projector. **Quantisation does not demote
a family**: every default tag on the site uses a type jitllm has (censused:
default file types are Q4_0 256 tags, Q4_K_M 164, F16 22, Q8_0 14, BF16 3,
MXFP4 4 -- and the tensor-type census adds only Q5_0, Q5_K, Q6_K, F32).
Non-default tags are priced separately (section 5, "Tag completeness").

**Pull counts** are the site's own counter at the time of the survey, per family, not per
tag. They measure historical popularity, not current demand: `llama3.1` at 120M
is two years old, `qwen3.5` at 21M is four weeks old. Weighting by them favours
old dense llama-shaped models, which is exactly what jitllm already runs.

**What "supported" does not establish.** Nothing here was converted or run. A
SUPPORTED row says the converter's lists accept the file; the arch-level
correctness gates (RULE 7: llama.cpp / transformers goldens) exist for the
architectures, not for each of these 122 checkpoints. `library.
TestEveryArchitectureIsObtainable` is the nearest existing gate and it checks a
download catalogue, not this list.

## 2. Family map

Every family, most-pulled first. `general.architecture` is what the served GGUF
says (several families ship different architectures per size). *shape* is MoE
when the file carries `expert_count > 0`, hybrid when it carries SSM /
ShortConv / gated-delta keys. *missing* lists capability IDs from section 4;
V-prefixed ones are vision/audio only, so a family whose only gaps are V items
already produces text.

| # | family | pulls | sizes | general.architecture | modality | shape | status | missing |
|---|---|---|---|---|---|---|---|---|
| 1 | llama3.1 | 119.9M | 8b,70b,405b | llama | text | dense | SUPPORTED | - |
| 2 | deepseek-r1 | 93.2M | 1.5b,7b,8b,14b,32b,70b,671b | qwen2, deepseek2, llama, qwen3 | text | MoE | SUPPORTED |  -- distills are qwen2/qwen3/llama; 671b is deepseek2 -- all run |
| 3 | nomic-embed-text | 87.2M | - | nomic-bert | embedding | dense | SUPPORTED | C12a C12b done: cos 0.9999999 vs float32 |
| 4 | llama3.2 | 84.4M | 1b,3b | llama | text | dense | SUPPORTED | - |
| 5 | qwen2.5 | 41M | 0.5b,1.5b,3b,7b,14b,32b,72b | qwen2 | text | dense | SUPPORTED | - |
| 6 | gemma3 | 40.7M | 270m,1b,4b,12b,27b | gemma3 | text+vision | dense | PARTIAL | V2 -- text: the HF GGUFs (ggml-org/gemma-3-4b-it-GGUF) convert; the tower ships as a separate mmproj, projector `gemma3` |
| 7 | qwen3 | 38.1M | 0.6b,1.7b,4b,8b,14b,30b,32b,235b | qwen3, qwen3moe | text | MoE | SUPPORTED | - |
| 8 | mistral | 33.7M | 7b | llama | text | dense | SUPPORTED | - |
| 9 | gemma2 | 33.4M | 2b,9b,27b | gemma2 | text | dense | SUPPORTED | - |
| 10 | gemma4 | 25.8M | e2b,e4b,12b,26b,31b | gemma4 | text+vision+audio | MoE | UNSUPPORTED | C3 C4b C14 V10 |
| 11 | llama3 | 25.3M | 8b,70b | llama | text | dense | SUPPORTED | - |
| 12 | qwen2.5-coder | 21.8M | 0.5b,1.5b,3b,7b,14b,32b | qwen2 | text | dense | SUPPORTED | - |
| 13 | qwen3.5 | 21M | 0.8b,2b,4b,9b,27b,35b,122b | qwen35, qwen35moe | text+vision | MoE/hybrid | PARTIAL | V0 V4 -- text: C9 done; Qwen3.5-0.8B/2B/4B/9B Q4_K_M teacher-forced 20/19/20/20 of 20 against llama.cpp, and the 0.8B MTP file 20 of 20 with its nextn block set aside |
| 14 | phi3 | 18.2M | 3.8b,14b | phi3 | text | dense | SUPPORTED | - |
| 15 | mxbai-embed-large | 15.1M | 335m | bert | embedding | dense | SUPPORTED | C12a C12b done: cos 0.9999896 vs float32 |
| 16 | llava | 15M | 7b,13b,34b | llama | text+vision | dense | SUPPORTED |  -- llava 1.5-style mmproj (no anyres grid): runs today |
| 17 | gpt-oss | 13.3M | 20b,120b | gptoss | text | MoE | SUPPORTED | - -- via ggml-org/gpt-oss-20b-GGUF (arch `gpt-oss`, pre `gpt-4o`); Ollama's own blob is the `gptoss` dialect, not pursued |
| 18 | qwen3-coder | 9.6M | 30b,480b | qwen3moe | text | MoE | SUPPORTED | - |
| 19 | gemma | 8.3M | 2b,7b | gemma | text | dense | SUPPORTED | - |
| 20 | qwen | 7.9M | 0.5b,1.8b,4b,7b,14b,32b,72b,110b | qwen2 | text | dense | SUPPORTED |  -- no tokenizer.ggml.pre: llama3 split (\p{N}{1,3}) where Qwen1.5 splits digits singly |
| 21 | phi4 | 7.7M | 14b | phi3 | text | dense | SUPPORTED | - |
| 22 | glm-ocr | 7.6M | - | glmocr | text+vision | dense | UNSUPPORTED | C3 C16 V0 V11 -- ggml-org/GLM-OCR-GGUF is arch `glm4`, pre `chatglm-bpe`, mmproj `glm4v` |
| 23 | llama2 | 7.5M | 7b,13b,70b | llama | text | dense | SUPPORTED | - |
| 24 | bge-m3 | 7M | 567m | bert | embedding | dense | UNSUPPORTED | C12a C12b C12c |
| 25 | qwen3.6 | 6.8M | 27b,35b | qwen35, qwen35moe | text+vision | MoE/hybrid | PARTIAL | V0 V4 -- text: C9 done; Qwen3.6-35B-A3B-UD-Q4_K_M (qwen35moe) 19 of 20 teacher-forced against llama.cpp, the 20th a 0.043 tie |
| 26 | codellama | 6.5M | 7b,13b,34b,70b | llama | text | dense | SUPPORTED | - |
| 27 | qwen3-vl | 6.3M | 2b,4b,8b,30b,32b,235b | qwen3vlmoe, qwen3vl | text+vision | MoE | PARTIAL | C1 V0 V4 -- the HF GGUFs (Qwen/Qwen3-VL-*-GGUF) say arch `qwen3vl`/`qwen3vlmoe`: llama.cpp's own names, not in archOf |
| 28 | qwen2 | 6.2M | 0.5b,1.5b,7b,72b | qwen2 | text | dense | SUPPORTED | - |
| 29 | tinyllama | 5.8M | 1.1b | llama | text | dense | SUPPORTED | - |
| 30 | mistral-nemo | 5.7M | 12b | llama | text | dense | SUPPORTED | - |
| 31 | minicpm-v | 5.5M | 8b | qwen2 | text+vision | dense | PARTIAL | V8 -- text runs (qwen2); resampler projector missing |
| 32 | llama3.2-vision | 5.4M | 11b,90b | mllama | text+vision | dense | UNSUPPORTED | C24 V7 -- llama.cpp dropped mllama, so there is no HF GGUF of the text model |
| 33 | qwen2.5vl | 5.1M | 3b,7b,32b,72b | qwen25vl | text+vision | dense | PARTIAL | V0 V3 -- text: ggml-org/Qwen2.5-VL-3B-Instruct-GGUF is arch `qwen2vl` and converts; mmproj projector `qwen2.5vl_merger` |
| 34 | deepseek-coder | 4.6M | 1.3b,6.7b,33b | llama | text | dense | SUPPORTED |  -- no tokenizer.ggml.pre: falls back to llama3's split, as llama.cpp does |
| 35 | llama3.3 | 4.2M | 70b | llama | text | dense | SUPPORTED | - |
| 36 | dolphin3 | 4.1M | 8b | llama | text | dense | SUPPORTED | - |
| 37 | qwen3-embedding | 4.1M | 0.6b,4b,8b | qwen3 | embedding | dense | SUPPORTED | C12a done: f16 cos 0.9999998 vs float32; Ollama's Q8_0 0.99898, llama.cpp on it 0.99899 |
| 38 | smollm2 | 4M | 135m,360m,1.7b | llama | text | dense | SUPPORTED | - |
| 39 | deepseek-v3 | 3.8M | 671b | deepseek2 | text | MoE | SUPPORTED | - |
| 40 | olmo2 | 3.8M | 7b,13b | olmo2 | text | dense | UNSUPPORTED | C7 |
| 41 | all-minilm | 3.6M | 22m,33m | bert | embedding | dense | SUPPORTED | C12a C12b done: 22m cos 0.9999983 vs float32 |
| 42 | codegemma | 3.2M | 2b,7b | gemma | text | dense | SUPPORTED | - |
| 43 | deepseek-coder-v2 | 3.1M | 16b,236b | deepseek2 | text | MoE | SUPPORTED | - |
| 44 | mistral-small | 3.1M | 22b,24b | llama | text | dense | SUPPORTED |  -- 22b is llama/SPM; 24b tags are llama/BPE labelled `qwen2` (Tekken vocabulary): audit the label (C3) |
| 45 | snowflake-arctic-embed | 3.1M | 22m,33m,110m,137m,335m | bert, nomic-bert | embedding | dense | SUPPORTED | C12a C12b done: 22m (xs) cos 0.9999995 vs float32; other tags are the same two graphs, not individually gated |
| 46 | orca-mini | 3.1M | 3b,7b,13b,70b | llama | text | dense | SUPPORTED | - |
| 47 | granite3.1-moe | 3M | 1b,3b | granitemoe | text | MoE | SUPPORTED | - -- C5 done; granite-3.1-1b-a400m Q4_K_M 20 of 20 teacher-forced against llama.cpp |
| 48 | starcoder2 | 3M | 3b,7b,15b | starcoder2 | text | dense | SUPPORTED | - -- C6 starcoder2 done; starcoder2-3b Q4_K_M 20 of 20 teacher-forced against llama.cpp, all 30 blocks and the head on CUDA; the sliding window is not applied, as llama.cpp does not |
| 49 | nemotron-3-super | 3M | 120b | nemotron_h_moe | text | MoE/hybrid | UNSUPPORTED | C11 |
| 50 | mixtral | 2.9M | 8x7b,8x22b | llama | text | MoE | SUPPORTED | - |
| 51 | llama2-uncensored | 2.7M | 7b,70b | llama | text | dense | SUPPORTED | - |
| 52 | qwen3.8 | 2.7M | 27b | qwen35 | text+vision | hybrid | PARTIAL | V0 V4 -- text: C9 done (qwen35); unsloth's UD-Q4_K_M carries IQ4_NL/IQ4_XS, which have no container type -- its Q8_0 converts |
| 53 | falcon3 | 2.6M | 1b,3b,7b,10b | llama | text | dense | SUPPORTED | - |
| 54 | mistral-small3.2 | 2.5M | 24b | mistral3 | text+vision | dense | PARTIAL | V5 -- text: bartowski's GGUF is arch `llama`, pre `tekken`, no temperature key, and converts; mmproj projector `pixtral` |
| 55 | translategemma | 2.5M | 4b,12b,27b | gemma3 | text+vision | dense | PARTIAL | V2 -- text: mradermacher/translategemma-4b-it-GGUF (gemma3) converts |
| 56 | minimax-m2.7 | 2.4M | cloud | minimax_m2 MoE, 229B | text | MoE | CLOUD-ONLY | C22 |
| 57 | llava-llama3 | 2.3M | 8b | llama | text+vision | dense | SUPPORTED | - |
| 58 | qwq | 2.3M | 32b | qwen2 | text | dense | SUPPORTED | - |
| 59 | embeddinggemma | 2.2M | 300m | gemma3 | embedding | dense | SUPPORTED | C12a done: cos 0.9999817 vs float32, symmetric window gated past 512 tokens |
| 60 | gemma3n | 2.2M | e2b,e4b | gemma3n | text | dense | UNSUPPORTED | C15 |
| 61 | smollm | 2.2M | 135m,360m,1.7b | llama | text | dense | SUPPORTED | - |
| 62 | dolphin-llama3 | 2.1M | 8b,70b | llama | text | dense | SUPPORTED | - |
| 63 | cogito | 2.1M | 3b,8b,14b,32b,70b | qwen2, llama | text | dense | SUPPORTED |  -- qwen2 and llama sizes |
| 64 | qwen3-coder-next | 2M | - | qwen3next | text | MoE/hybrid | SUPPORTED | - -- unsloth/Qwen3-Coder-Next-GGUF converts (split attn_qkv/attn_gate, `ssm_dt.bias`) |
| 65 | glm-4.7-flash | 2M | - | glm4moelite | text | MoE | SUPPORTED | - -- unsloth/GLM-4.7-Flash-GGUF is arch `deepseek2` (llama.cpp's MLA key order) and zai-org's safetensors are Glm4MoeLiteForCausalLM; both convert |
| 66 | moondream | 1.9M | 1.8b | phi2 | text+vision | dense | PARTIAL | - -- the phi2 text runs (C6 done); its `mlp` mmproj is not verified |
| 67 | sqlcoder | 1.9M | 7b,15b | starcoder, llama | text | dense | SUPPORTED | - -- 7b is llama; 15b is starcoder (C6 done; the 15b file itself not run) |
| 68 | dolphin-mixtral | 1.9M | 8x7b,8x22b | llama | text | MoE | SUPPORTED | - |
| 69 | llama4 | 1.8M | 16x17b,128x17b | llama4 | text+vision | MoE | UNSUPPORTED | C3 C4a C8 C13 V6 -- HF GGUFs say pre `llama4`, which has no pretok entry |
| 70 | dolphin-mistral | 1.8M | 7b | llama | text | dense | SUPPORTED | - |
| 71 | phi4-reasoning | 1.7M | 14b | phi3 | text | dense | SUPPORTED | - |
| 72 | hermes3 | 1.7M | 3b,8b,70b,405b | llama | text | dense | SUPPORTED | - |
| 73 | dolphin-phi | 1.6M | 2.7b | phi2 | text | dense | SUPPORTED | - -- C6 phi2 done (the phi-2 graph; see `phi`) |
| 74 | granite-code | 1.6M | 3b,8b,20b,34b | starcoder, llama | text | dense | SUPPORTED | - -- 3b/8b are llama; 20b/34b are starcoder (C6 done; those files themselves not run) |
| 75 | phi | 1.5M | 2.7b | phi2 | text | dense | SUPPORTED | - -- C6 phi2 done; TheBloke phi-2 Q4_K_M (fused attn_qkv) token-identical to llama.cpp over 32 tokens, synth-phi2 4.7e-13 against transformers; every block and the head on CUDA and Vulkan |
| 76 | command-r | 1.5M | 35b | command-r | text | dense | SUPPORTED | - -- C6 command-r done (the 35b file itself not run; aya-expanse-8b is the same arch, 20 of 20) |
| 77 | granite4 | 1.5M | 350m,1b,3b | granite, granitehybrid | text | hybrid | PARTIAL | C11 -- the dense sizes are granite and run (C5 done; granite-4.0-micro 20 of 20, head_count_kv an array); the -h tags are granitehybrid (Mamba-2) |
| 78 | yi | 1.5M | 6b,9b,34b | llama | text | dense | SUPPORTED | - |
| 79 | magistral | 1.5M | 24b | llama | text | dense | SUPPORTED | - |
| 80 | phi4-mini | 1.5M | 3.8b | phi3 | text | dense | SUPPORTED | - |
| 81 | ministral-3 | 1.5M | 3b,8b,14b | mistral3 | text+vision | dense | UNSUPPORTED | C1 C8 V5 -- mistralai's own GGUF: arch `mistral3` with attention.temperature_scale 0.1 |
| 82 | starcoder | 1.4M | 1b,3b,7b,15b | starcoder | text | dense | SUPPORTED | - -- C6 starcoder done; starcoderbase-1b Q4_K_M 18 of 19 teacher-forced against llama.cpp and 32 greedy tokens identical, every block and the head on CUDA (0/32 flips) |
| 83 | openchat | 1.4M | 7b | llama | text | dense | SUPPORTED | - |
| 84 | codestral | 1.3M | 22b | llama | text | dense | SUPPORTED | - |
| 85 | devstral-small-2 | 1.3M | 24b | mistral3 | text+vision | dense | UNSUPPORTED | C1 C8 V5 -- bartowski's GGUF: arch `mistral3` with attention.temperature_scale 0.1 |
| 86 | mistral-large | 1.3M | 123b | llama | text | dense | SUPPORTED | - |
| 87 | wizard-vicuna-uncensored | 1.3M | 7b,13b,30b | llama | text | dense | SUPPORTED | - |
| 88 | zephyr | 1.3M | 7b,141b | llama | text | MoE | SUPPORTED | - |
| 89 | lfm2.5-thinking | 1.3M | 1.2b | lfm2 | text | hybrid | UNSUPPORTED | C10 |
| 90 | deepscaler | 1.3M | 1.5b | qwen2 | text | dense | SUPPORTED | - |
| 91 | vicuna | 1.2M | 7b,13b,33b | llama | text | dense | SUPPORTED | - |
| 92 | glm4 | 1.2M | 9b | chatglm | text | dense | UNSUPPORTED | C3 C16 |
| 93 | wizardcoder | 1.2M | 33b | llama | text | dense | SUPPORTED |  -- 33b is deepseek-coder based: no tokenizer.ggml.pre, llama3 fallback as in llama.cpp |
| 94 | deepseek-llm | 1.2M | 7b,67b | llama | text | dense | SUPPORTED |  -- no tokenizer.ggml.pre: falls back to llama3's split, as llama.cpp does |
| 95 | openhermes | 1.2M | - | llama | text | dense | SUPPORTED | - |
| 96 | nous-hermes | 1.2M | 7b,13b | llama | text | dense | SUPPORTED | - |
| 97 | deepseek-v2 | 1.2M | 16b,236b | deepseek2 | text | MoE | SUPPORTED | - |
| 98 | wizardlm2 | 1.2M | 7b,8x22b | llama | text | MoE | SUPPORTED | - |
| 99 | falcon | 1.2M | 7b,40b,180b | falcon | text | dense | SUPPORTED | C6 falcon done: falcon-7b-instruct Q8_0 10 of 10 teacher-forced and 32 greedy tokens identical against llama.cpp, 0/32 flips on CUDA. Q5_1 landed with it, so the K-quants convert: mradermacher's Q4_K_M (32 Q5_1 attn_qkv) 11 of 11 teacher-forced against llama.cpp, 0/32 flips on CUDA with 10 blocks placed, 1/32 on Vulkan with 13, max\|dlogit\| 0.40/0.44 |
| 100 | openthinker | 1.2M | 7b,32b | qwen2 | text | dense | SUPPORTED | - |
| 101 | neural-chat | 1.1M | 7b | llama | text | dense | SUPPORTED | - |
| 102 | lfm2 | 1.1M | 24b | lfm2moe | text | MoE/hybrid | UNSUPPORTED | C10 |
| 103 | qwen2-math | 1.1M | 1.5b,7b,72b | qwen2 | text | dense | SUPPORTED | - |
| 104 | codeqwen | 1.1M | 7b | qwen2 | text | dense | SUPPORTED | - |
| 105 | granite3.3 | 1.1M | 2b,8b | granite | text | dense | SUPPORTED | - -- C5 done; granite-3.3-2b-instruct Q4_K_M 20 of 20 |
| 106 | llama2-chinese | 1.1M | 7b,13b | llama | text | dense | SUPPORTED | - |
| 107 | nous-hermes2 | 1.1M | 10.7b,34b | llama | text | dense | SUPPORTED | - |
| 108 | aya | 1.1M | 8b,35b | command-r | text | dense | SUPPORTED | - -- C6 command-r done (these files themselves not run) |
| 109 | stablelm2 | 1.1M | 1.6b,12b | stablelm | text | dense | PARTIAL | 12b's per-head q/k LayerNorm, refused by name -- 1.6b done: stablelm-2-1_6b 17 of 20 and stablelm-2-zephyr-1_6b 19 of 20 teacher-forced against llama.cpp (every miss a tie), every block and the head on CUDA |
| 110 | yi-coder | 1.1M | 1.5b,9b | llama | text | dense | SUPPORTED | - |
| 111 | stable-code | 1.1M | 3b | stablelm | text | dense | SUPPORTED | - -- C6 stablelm done (this file itself not run) |
| 112 | wizard-math | 1.1M | 7b,13b,70b | llama | text | dense | SUPPORTED | - |
| 113 | llama-guard3 | 1.1M | 1b,8b | llama | text | dense | SUPPORTED | - |
| 114 | llama3-chatqa | 1M | 8b,70b | llama | text | dense | SUPPORTED | - |
| 115 | devstral | 1M | 24b | llama | text | dense | SUPPORTED | - |
| 116 | internlm2 | 1M | 1m,1.8b,7b,20b | internlm2 | text | dense | PARTIAL | C1 -- the HF GGUFs say arch `internlm2`, llama.cpp's own name for llama's block |
| 117 | granite3.1-dense | 1M | 2b,8b | granite | text | dense | SUPPORTED | - -- C5 done (header keys the same set the verified files carry) |
| 118 | phi3.5 | 1M | 3.8b | phi3 | text | dense | SUPPORTED | - |
| 119 | granite3-dense | 1M | 2b,8b | granite | text | dense | SUPPORTED | - -- C5 done (header keys the same set the verified files carry) |
| 120 | aya-expanse | 1M | 8b,32b | command-r | text | dense | SUPPORTED | - -- C6 command-r done; aya-expanse-8b Q4_K_M 20 of 20 teacher-forced against llama.cpp, 24 greedy tokens identical on two prompts |
| 121 | xwinlm | 1M | 7b,13b | llama | text | dense | SUPPORTED | - |
| 122 | dolphincoder | 1M | 7b,15b | starcoder2 | text | dense | SUPPORTED | - -- C6 starcoder2 done (these files themselves not run) |
| 123 | samantha-mistral | 1M | 7b | llama | text | dense | SUPPORTED | - |
| 124 | llama3-gradient | 1M | 8b,70b | llama | text | dense | SUPPORTED | - |
| 125 | llama3-groq-tool-use | 1M | 8b,70b | llama | text | dense | SUPPORTED | - |
| 126 | granite3.2-vision | 1M | 2b | granite | text+vision | dense | PARTIAL | V9 -- text: C5 done |
| 127 | yarn-llama2 | 1M | 7b,13b | llama | text | dense | SUPPORTED | - |
| 128 | starling-lm | 997.6K | 7b | llama | text | dense | SUPPORTED | - |
| 129 | phind-codellama | 992.1K | 34b | llama | text | dense | SUPPORTED | - |
| 130 | solar | 978.3K | 10.7b | llama | text | dense | SUPPORTED |  -- SOLAR-10.7B: plain llama arch |
| 131 | nomic-embed-text-v2-moe | 969.6K | - | nomic-bert-moe | embedding | MoE | UNSUPPORTED | C12a C12b C12c |
| 132 | granite3-moe | 967K | 1b,3b | granitemoe | text | MoE | SUPPORTED | - -- C5 done (header keys the same set the verified files carry) |
| 133 | stable-beluga | 960.3K | 7b,13b,70b | llama | text | dense | SUPPORTED | - |
| 134 | deepcoder | 956.8K | 1.5b,14b | qwen2 | text | dense | SUPPORTED | - |
| 135 | shieldgemma | 950.5K | 2b,9b,27b | gemma2 | text | dense | SUPPORTED | - |
| 136 | orca2 | 949.7K | 7b,13b | llama | text | dense | SUPPORTED | - |
| 137 | paraphrase-multilingual | 943.5K | 278m | bert | embedding | dense | UNSUPPORTED | C12a C12b C12c |
| 138 | reader-lm | 941.4K | 0.5b,1.5b | qwen2 | text | dense | SUPPORTED | - |
| 139 | wizardlm | 933.5K | - | llama | text | dense | SUPPORTED |  -- no `latest` tag; 13b-* tags are llama |
| 140 | llama-pro | 923.7K | - | llama | text | dense | SUPPORTED | - |
| 141 | yarn-mistral | 914K | 7b | llama | text | dense | SUPPORTED | - |
| 142 | nexusraven | 908.2K | 13b | llama | text | dense | SUPPORTED | - |
| 143 | meditron | 900.7K | 7b,70b | llama | text | dense | SUPPORTED | - |
| 144 | bakllava | 894.3K | 7b | llama | text+vision | dense | PARTIAL | V1 -- mmproj carries no clip.projector_type |
| 145 | nemotron-3-nano | 862.7K | 4b,30b | nemotron_h_moe, nemotron_h | text | MoE/hybrid | UNSUPPORTED | C11 -- the HF GGUFs say pre `pixtral`, which resolves |
| 146 | command-r-plus | 808.2K | 104b | command-r | text | dense | UNSUPPORTED | its per-head q/k LayerNorm ({head_dim, n_head} weights) -- refused by name at conversion |
| 147 | mistral-small3.1 | 792.6K | 24b | mistral3 | text+vision | dense | PARTIAL | V5 -- text: bartowski's GGUF is arch `llama`, pre `tekken`, and converts |
| 148 | mistral-openorca | 785.9K | 7b | llama | text | dense | SUPPORTED | - |
| 149 | exaone-deep | 772.2K | 2.4b,7.8b,32b | exaone | text | dense | PARTIAL | C1 -- the HF GGUFs say arch `exaone` (llama's block, NEOX rotary) |
| 150 | tinydolphin | 733.6K | 1.1b | llama | text | dense | SUPPORTED | - |
| 151 | medllama2 | 731.2K | 7b | llama | text | dense | SUPPORTED | - |
| 152 | deepseek-v3.1 | 730.2K | 671b | deepseek2 | text | MoE | SUPPORTED | - |
| 153 | nemotron-mini | 719.3K | 4b | nemotron | text | dense | SUPPORTED | - -- C6 nemotron done; Nemotron-Mini-4B-Instruct Q4_K_M 20 of 20 teacher-forced against llama.cpp, 32 greedy tokens identical, all 32 blocks on CUDA (0/32 flips) |
| 154 | codegeex4 | 697.8K | 9b | chatglm | text | dense | UNSUPPORTED | C3 C16 |
| 155 | nemotron3 | 666.7K | 33b | nemotron_h_omni | text+vision+audio | MoE/hybrid | UNSUPPORTED | C11 V12 |
| 156 | opencoder | 654.5K | 1.5b,8b | llama | text | dense | SUPPORTED | - |
| 157 | wizardlm-uncensored | 652.6K | 13b | llama | text | dense | SUPPORTED | - |
| 158 | nemotron | 631.2K | 70b | llama | text | dense | SUPPORTED |  -- Llama-3.1-Nemotron: plain llama arch |
| 159 | reflection | 623.3K | 70b | llama | text | dense | SUPPORTED | - |
| 160 | codeup | 602.2K | 13b | llama | text | dense | SUPPORTED | - |
| 161 | nous-hermes2-mixtral | 597.6K | 8x7b | llama | text | MoE | SUPPORTED | - |
| 162 | athene-v2 | 596.4K | 72b | qwen2 | text | dense | SUPPORTED | - |
| 163 | qwen3-next | 592.7K | 80b | qwen3next | text | MoE/hybrid | SUPPORTED | - -- the HF GGUFs (unsloth/bartowski) carry split attn_qkv/attn_gate and convert; Qwen3-Next-80B-A3B-Instruct is token-identical to llama.cpp (AGENTS.md RULE 7) |
| 164 | megadolphin | 575.8K | 120b | llama | text | dense | SUPPORTED | - |
| 165 | everythinglm | 573K | 13b | llama | text | dense | SUPPORTED | - |
| 166 | solar-pro | 565.7K | 22b | solar | text | dense | UNSUPPORTED | C19 |
| 167 | exaone3.5 | 565.7K | 2.4b,7.8b,32b | exaone | text | dense | PARTIAL | C1 -- the HF GGUFs say arch `exaone` (llama's block, NEOX rotary) |
| 168 | magicoder | 562.5K | 7b | llama | text | dense | SUPPORTED | - |
| 169 | nuextract | 556.6K | 3.8b | phi3 | text | dense | SUPPORTED | - |
| 170 | mathstral | 556.4K | 7b | llama | text | dense | SUPPORTED | - |
| 171 | falcon2 | 546.9K | 11b | falcon | text | dense | PARTIAL | as falcon; Q5_1 converts now (this file itself not run) |
| 172 | notus | 546.6K | 7b | llama | text | dense | SUPPORTED | - |
| 173 | notux | 545.4K | 8x7b | llama | text | MoE | SUPPORTED | - |
| 174 | deepseek-ocr | 540K | 3b | deepseekocr | text+vision | MoE | UNSUPPORTED | C25 V11 -- ggml-org/DeepSeek-OCR-GGUF is arch `deepseek2-ocr`, pre `deepseek-v3` |
| 175 | stablelm-zephyr | 539.4K | 3b | stablelm | text | dense | SUPPORTED | - -- C6 stablelm done (this file itself not run) |
| 176 | bespoke-minicheck | 536.3K | 7b | internlm2 | text | dense | PARTIAL | C1 -- no HF GGUF; its safetensors are InternLM2 (remote code), and a GGUF of it says `internlm2` |
| 177 | duckdb-nsql | 534.1K | 7b | llama | text | dense | SUPPORTED | - |
| 178 | ornith | 534K | 9b,35b | qwen35moe, qwen35 | text | MoE/hybrid | SUPPORTED | - -- qwen35/qwen35moe, C9 done |
| 179 | firefunction-v2 | 530.5K | 70b | llama | text | dense | SUPPORTED | - |
| 180 | mistrallite | 530K | 7b | llama | text | dense | SUPPORTED | - |
| 181 | wizard-vicuna | 524.3K | 13b | llama | text | dense | SUPPORTED | - |
| 182 | open-orca-platypus2 | 516.3K | 13b | llama | text | dense | SUPPORTED | - |
| 183 | minimax-m3 | 515.8K | cloud | minimax_m3 sparse attention + vision | text+vision | dense | CLOUD-ONLY | C20 |
| 184 | rnj-1 | 510.9K | 8b | gemma3 | text | dense | SUPPORTED | - -- no sliding_window key means no sliding layers (fixed; it had defaulted to a 1024 window, 5 local of 6); 20 of 20 teacher-forced, and a ~1700-token prompt word for word against llama.cpp |
| 185 | codebooga | 506.3K | 34b | llama | text | dense | SUPPORTED | - |
| 186 | goliath | 485.5K | - | llama | text | dense | SUPPORTED | - |
| 187 | kimi-k2.6 | 481.7K | cloud | kimi_k25 = deepseek2 text (MLA, V3 router) + MoonViT, 1.04T | text+vision | dense | CLOUD-ONLY | V13 |
| 188 | granite4.1 | 477.9K | 3b,8b,30b | granite | text | dense | SUPPORTED | - -- C5 done (header keys the same set the verified files carry) |
| 189 | olmo-3 | 462.4K | 7b,32b | olmo3 | text | dense | UNSUPPORTED | C7 C4a -- HF GGUFs say arch `olmo2` |
| 190 | granite3.2 | 460.8K | 2b,8b | granite | text | dense | SUPPORTED | - -- C5 done (header keys the same set the verified files carry) |
| 191 | snowflake-arctic-embed2 | 458.2K | 568m | bert | embedding | dense | UNSUPPORTED | C12a C12b C12c |
| 192 | llava-phi3 | 442.5K | 3.8b | llama | text+vision | dense | SUPPORTED | - |
| 193 | medgemma | 426.3K | 4b,27b | gemma3 | text+vision | dense | PARTIAL | V2 -- text: unsloth/medgemma-4b-it-GGUF (gemma3) converts |
| 194 | deepseek-v4-pro | 426K | cloud | deepseek_v4: hyper-connections, compressed attention, indexer, SWA 128, sinks, 1.65T | text | dense | CLOUD-ONLY | C20 |
| 195 | r1-1776 | 425.4K | 70b,671b | deepseek2, llama | text | MoE | SUPPORTED |  -- 70b llama, 671b deepseek2 |
| 196 | sailor2 | 419.6K | 1b,8b,20b | qwen2 | text | dense | SUPPORTED |  -- labelled `llama-bpe` on a Qwen2.5 vocabulary: audit the label (C3) |
| 197 | mistral-medium-3.5 | 416.3K | 128b | mistral3 | text+vision | dense | PARTIAL | C1 V5 -- bartowski's GGUF: arch `mistral3` with attention.temperature_scale 0.0 (so llama's graph), pre `pixtral` |
| 198 | tulu3 | 402.7K | 8b,70b | llama | text | dense | SUPPORTED | - |
| 199 | granite-embedding | 375.1K | 30m,278m | bert | embedding | dense | UNSUPPORTED | C12a C12b C12c |
| 200 | dbrx | 369.6K | 132b | dbrx | text | MoE | PARTIAL | C6 dbrx done on a DbrxForCausalLM fixture only (host, CUDA, Vulkan; clamp_kqv on both tiers); no real file fits this box, so no real-file gate |
| 201 | devstral-2 | 358K | 123b | mistral3 | text | dense | SUPPORTED | - -- ggml-org/Devstral-2-123B-Instruct-2512-GGUF is arch `llama` (YaRN x64), pre `tekken`, and converts |
| 202 | glm-5.2 | 355.7K | cloud | glm_moe_dsa: MLA + DSA indexer, 756B | text | MoE | CLOUD-ONLY | C20 |
| 203 | ornith-1.5 | 349.3K | 9b,35b,397b | qwen35moe, qwen35 | text+vision | MoE/hybrid | PARTIAL | V0 V4 -- text: qwen35/qwen35moe, C9 done |
| 204 | granite3-guardian | 341K | 2b,8b | granite | text | dense | SUPPORTED | - -- C5 done (header keys the same set the verified files carry) |
| 205 | command-r7b | 327.7K | 7b | cohere2 | text | dense | SUPPORTED | - -- C6 cohere2 done; command-r7b Q4_K_M 20 of 20 teacher-forced against llama.cpp, 24 greedy tokens identical on two prompts |
| 206 | phi4-mini-reasoning | 314.1K | 3.8b | phi3 | text | dense | SUPPORTED | - |
| 207 | olmo-3.1 | 293K | 32b | olmo3 | text | dense | UNSUPPORTED | C7 C4a |
| 208 | deepseek-v2.5 | 292K | 236b | deepseek2 | text | MoE | SUPPORTED | - |
| 209 | bge-large | 287.3K | 335m | bert | embedding | dense | UNSUPPORTED | C12a C12b |
| 210 | alfred | 271.3K | 40b | falcon | text | dense | PARTIAL | as falcon; Q5_1 converts now (this file itself not run) |
| 211 | smallthinker | 262K | 3b | qwen2 | text | dense | SUPPORTED | - |
| 212 | kimi-k2.7-code | 243.3K | cloud | kimi_k25 = deepseek2 text + MoonViT, 1.04T | text+vision | dense | CLOUD-ONLY | V13 |
| 213 | muse-glimmer | 235K | 30b | muse-glimmer | text+vision | dense | UNSUPPORTED | C3 C4a C17a V12 |
| 214 | command-a | 230.7K | 111b | cohere2 | text | dense | SUPPORTED | - -- C6 cohere2 done (the 111b file itself not run) |
| 215 | cogito-2.1 | 224.2K | 671b | deepseek2 | text | MoE | SUPPORTED | - |
| 216 | marco-o1 | 215K | 7b | qwen2 | text | dense | SUPPORTED | - |
| 217 | command-r7b-arabic | 206.2K | 7b | cohere2 | text | dense | SUPPORTED | - -- C6 cohere2 done (this file itself not run) |
| 218 | medgemma1.5 | 201.4K | 4b | gemma3 | text+vision | dense | PARTIAL | V2 -- text: unsloth/medgemma-1.5-4b-it-GGUF (gemma3) converts |
| 219 | functiongemma | 192.6K | 270m | gemma3 | text | dense | SUPPORTED | - |
| 220 | nemotron-3.5-lightning | 186.9K | 30b | nemotron_h_moe | text | MoE/hybrid | UNSUPPORTED | C11 |
| 221 | laguna-s-2.1 | 165.2K | - | laguna | text | MoE | UNSUPPORTED | C4a C4b C17b |
| 222 | lfm2.5 | 162.7K | 8b | lfm2moe | text | MoE/hybrid | UNSUPPORTED | C10 |
| 223 | gpt-oss-safeguard | 158.5K | 20b,120b | gptoss | text | MoE | SUPPORTED | - -- unsloth/gpt-oss-safeguard-20b-GGUF is arch `gpt-oss`, pre `gpt-4o` |
| 224 | glm-5.3-flash | 154.3K | cloud | glm5_next: KDA + MLA + DSA + hyper-connections, 321B | text+vision | hybrid | CLOUD-ONLY | C20 C21 |
| 225 | qwen3.8-flash-next | 151.7K | - | qwen4exp | text+vision | dense | UNSUPPORTED | C20 V0 V4 -- arch qwen4exp; its qwen35 half is C9 (done) |
| 226 | nemotron-cascade-2 | 149.6K | 30b | nemotron_h_moe | text | MoE/hybrid | UNSUPPORTED | C11 |
| 227 | mistral-large-3 | 120.8K | cloud | mistral4 = deepseek2 graph, 675B | text+vision | dense | CLOUD-ONLY | C23 |
| 228 | laguna-xs-2.1 | 119.1K | - | laguna | text | MoE | UNSUPPORTED | C4a C4b C17b |
| 229 | nemotron-3-ultra | 107.6K | cloud | nemotron_h (Mamba-2 + MoE), 550B | text | MoE/hybrid | CLOUD-ONLY | C11 |
| 230 | kimi-k3 | 90.4K | cloud | kimi_linear (KDA + MLA) + additions, 2.81T | text+vision | hybrid | CLOUD-ONLY | C21 |
| 231 | granite4.2 | 90.1K | 3b,8b,30b | granite | text | dense | SUPPORTED | - -- C5 done (header keys the same set the verified files carry) |
| 232 | glm-5.3 | 61.8K | cloud | glm_moe_dsa: MLA + DSA indexer, 753B | text | MoE | CLOUD-ONLY | C20 |
| 233 | north-mini-code-1.0 | 59.6K | - | cohere2moe | text | MoE | UNSUPPORTED | C6 C4a C17c |
| 234 | minicpm-v4.6 | 55.8K | 1b | qwen35 | text+vision | hybrid | PARTIAL | V8 -- text: qwen35, C9 done; resampler projector missing |
| 235 | deepseek-v4.1-flash | 52.8K | cloud | deepseek_v41: v4 + engram/compressed vocab, 763B | text+vision | dense | CLOUD-ONLY | C20 |
| 236 | minicpm-v4.5 | 44.5K | 8b | qwen3 | text+vision | dense | PARTIAL | V8 -- text runs (qwen3); resampler projector missing |
| 237 | laguna-xs.2 | 31.7K | - | laguna | text | MoE | UNSUPPORTED | C4a C4b C17b |
| 238 | granite4.1-guardian | 27.9K | 8b | granite | text | dense | SUPPORTED | - -- C5 done (header keys the same set the verified files carry) |


## 3. Architecture map

One row per `general.architecture` string served by Ollama. Several are
Ollama-only names with no llama.cpp arch of that name (`gptoss`, `glm4moelite`,
`qwen25vl`, `olmo3`, `mllama`, `glmocr`, `deepseekocr`, `nemotron_h_omni`,
`solar`): Ollama's Go runtime (`model/models/*` in ollama/ollama) is the
reference for those, and for the three that alias an existing graph the alias
is the whole job. *pulls\** double-counts a family that ships two
architectures.

★ **THIS MAP IS BY THE OLLAMA BLOB'S NAME AND WAS NOT RE-GRADED.** Section 2 and
the headline are graded by the Hugging Face artefact (see section 0); the
Ollama-only names here -- `gptoss`, `glm4moelite`, `qwen25vl`, `olmo3`,
`mllama`, `glmocr`, `deepseekocr`, `nemotron_h_omni` -- are what the registry
serves, and the HF GGUFs of the same models say `gpt-oss`, `deepseek2`,
`qwen2vl`, `olmo2`, (none), `glm4`, `deepseek2-ocr` and `nemotron_h_moe`.

| general.architecture | llama.cpp / reference | status | missing (text) | families | pulls* |
|---|---|---|---|---|---|
| `llama` | llama.cpp | SUPPORTED | - | llama3.2, llama3.1, deepseek-r1, mistral, llama3, llava, llama2, codellama, tinyllama, mistral-nemo, deepseek-coder, llama3.3 (+78) | 500.9M |
| `qwen2` | qwen2.cpp | SUPPORTED | - | deepseek-r1, qwen2.5, qwen2.5-coder, qwen, qwen2, minicpm-v, qwq, cogito, deepscaler, openthinker, qwen2-math, codeqwen (+6) | 188.1M |
| `qwen3` | qwen3.cpp | SUPPORTED | - | deepseek-r1, qwen3, qwen3-embedding, minicpm-v4.5 | 135.4M |
| `deepseek2` | deepseek2.cpp | SUPPORTED | - | deepseek-r1, deepseek-v3, deepseek-coder-v2, deepseek-v2, deepseek-v3.1, r1-1776, deepseek-v2.5, cogito-2.1 | 103.0M |
| `nomic-bert` | nomic-bert.cpp | SUPPORTED | - | nomic-embed-text, snowflake-arctic-embed | 90.3M |
| `qwen3moe` | qwen3moe.cpp | SUPPORTED | - | qwen3, qwen3-coder | 47.7M |
| `gemma3` | gemma3.cpp | SUPPORTED | - | gemma3, translategemma, embeddinggemma, rnj-1, medgemma, medgemma1.5, functiongemma | 46.7M |
| `gemma2` | gemma2.cpp | SUPPORTED | - | gemma2, shieldgemma | 34.4M |
| `qwen35` | qwen35.cpp | UNSUPPORTED | C1 C9 | qwen3.5, qwen3.6, qwen3.8, ornith, ornith-1.5, minicpm-v4.6 | 31.4M |
| `phi3` | phi3.cpp | SUPPORTED | - | phi3, phi4, phi4-reasoning, phi4-mini, phi3.5, nuextract, phi4-mini-reasoning | 31.0M |
| `bert` | bert.cpp | SUPPORTED (WordPiece; the Unigram tags need C12c) | C12c | mxbai-embed-large, bge-m3, all-minilm, snowflake-arctic-embed, paraphrase-multilingual, snowflake-arctic-embed2, granite-embedding, bge-large | 30.9M |
| `qwen35moe` | qwen35moe.cpp | UNSUPPORTED | C9 | qwen3.5, qwen3.6, ornith, ornith-1.5 | 28.7M |
| `gemma4` | gemma4.cpp | UNSUPPORTED | C3 C4b C14 | gemma4 | 25.8M |
| `gptoss` | openai-moe.cpp (llama.cpp spells it `gpt-oss`) | PARTIAL | C1 C3 | gpt-oss, gpt-oss-safeguard | 13.5M |
| `gemma` | gemma.cpp | SUPPORTED | - | gemma, codegemma | 11.5M |
| `glmocr` | glm4.cpp text + glm4v tower (ollama name) | UNSUPPORTED | C2 C16 | glm-ocr | 7.6M |
| `granite` | granite.cpp | SUPPORTED (C5 done) | - | granite4, granite3.3, granite3.1-dense, granite3-dense, granite3.2-vision, granite4.1, granite3.2, granite3-guardian, granite4.2, granite4.1-guardian | 7.0M |
| `mistral3` | mistral3.cpp | UNSUPPORTED | C1 C2 C3 C8 | mistral-small3.2, ministral-3, devstral-small-2, mistral-small3.1, mistral-medium-3.5, devstral-2 | 6.9M |
| `qwen3vlmoe` | qwen3vlmoe.cpp | PARTIAL | C1 C2 | qwen3-vl | 6.3M |
| `qwen3vl` | qwen3vl.cpp | PARTIAL | C1 C2 | qwen3-vl | 6.3M |
| `mllama` | ollama model/models/mllama (llama.cpp dropped it) | UNSUPPORTED | C2 C24 | llama3.2-vision | 5.4M |
| `qwen25vl` | qwen2vl.cpp (ollama-only name) | PARTIAL | C1 C2 | qwen2.5vl | 5.1M |
| `phi2` | phi2.cpp | SUPPORTED (C6 phi2 done) | - | moondream, dolphin-phi, phi | 5.0M |
| `starcoder` | starcoder.cpp | SUPPORTED (C6 starcoder done) | - | sqlcoder, granite-code, starcoder | 4.9M |
| `command-r` | command-r.cpp | SUPPORTED (C6 command-r done; command-r-plus's per-head q/k norm refused) | - | command-r, aya, aya-expanse, command-r-plus | 4.4M |
| `nemotron_h_moe` | nemotron-h-moe.cpp | UNSUPPORTED | C3 C11 | nemotron-3-super, nemotron-3-nano, nemotron-3.5-lightning, nemotron-cascade-2 | 4.2M |
| `starcoder2` | starcoder2.cpp | SUPPORTED (C6 starcoder2 done) | - | starcoder2, dolphincoder | 4.0M |
| `granitemoe` | granite-moe.cpp | SUPPORTED (C5 done) | - | granite3.1-moe, granite3-moe | 4.0M |
| `olmo2` | olmo2.cpp | UNSUPPORTED | C7 | olmo2 | 3.8M |
| `stablelm` | stablelm.cpp | SUPPORTED (C6 stablelm done; 12B's per-head q/k norm refused) | - | stablelm2, stable-code, stablelm-zephyr | 2.7M |
| `qwen3next` | qwen3next.cpp | PARTIAL | C1 | qwen3-coder-next, qwen3-next | 2.6M |
| `gemma3n` | gemma3n.cpp | UNSUPPORTED | C15 | gemma3n | 2.2M |
| `falcon` | falcon.cpp | SUPPORTED (C6 falcon done; Q8_0 and Q4_K_M, the latter's Q5_1 attn_qkv included, verified against llama.cpp) | -- | falcon, falcon2, alfred | 2.0M |
| `glm4moelite` | deepseek2 graph (ollama-only name) | PARTIAL | C1 | glm-4.7-flash | 2.0M |
| `chatglm` | chatglm.cpp | UNSUPPORTED | C3 C16 | glm4, codegeex4 | 1.9M |
| `llama4` | llama4.cpp | UNSUPPORTED | C2 C3 C4a C8 C13 | llama4 | 1.8M |
| `internlm2` | internlm2.cpp | PARTIAL | C1 | internlm2, bespoke-minicheck | 1.5M |
| `granitehybrid` | granite-hybrid.cpp | UNSUPPORTED | C11 | granite4 | 1.5M |
| `exaone` | exaone.cpp | PARTIAL | C1 | exaone-deep, exaone3.5 | 1.3M |
| `lfm2` | lfm2.cpp | UNSUPPORTED | C10 | lfm2.5-thinking | 1.3M |
| `lfm2moe` | lfm2moe.cpp | UNSUPPORTED | C10 | lfm2, lfm2.5 | 1.3M |
| `nomic-bert-moe` | nomic-bert-moe.cpp | UNSUPPORTED | C12a C12b C12c | nomic-embed-text-v2-moe | 970K |
| `nemotron_h` | nemotron-h.cpp | UNSUPPORTED | C3 C11 | nemotron-3-nano | 863K |
| `cohere2` | cohere2.cpp | SUPPORTED (C6 cohere2 done) | - | command-r7b, command-a, command-r7b-arabic | 765K |
| `olmo3` | olmo2.cpp + SWA (ollama name) | UNSUPPORTED | C1 C7 C4a | olmo-3, olmo-3.1 | 755K |
| `nemotron` | nemotron.cpp | UNSUPPORTED | C6 | nemotron-mini | 719K |
| `nemotron_h_omni` | nemotron-h-moe + towers (ollama name) | UNSUPPORTED | C2 C3 C11 | nemotron3 | 667K |
| `solar` | ollama model/models (bskcn) | UNSUPPORTED | C19 | solar-pro | 566K |
| `deepseekocr` | deepseek2ocr.cpp (ollama name) | UNSUPPORTED | C2 C3 C25 | deepseek-ocr | 540K |
| `dbrx` | dbrx.cpp | PARTIAL (C6 dbrx done, synthetic fixture only) | -- | dbrx | 370K |
| `laguna` | laguna.cpp | UNSUPPORTED | C4a C4b C17b | laguna-s-2.1, laguna-xs-2.1, laguna-xs.2 | 316K |
| `muse-glimmer` | muse-glimmer.cpp | UNSUPPORTED | C3 C4a C17a | muse-glimmer | 235K |
| `qwen4exp` | qwen4exp.cpp | UNSUPPORTED | C9 C20 | qwen3.8-flash-next | 152K |
| `cohere2moe` | cohere2moe.cpp | UNSUPPORTED | C6 C4a C17c | north-mini-code-1.0 | 60K |

### 3b. What each missing architecture needs, read from its graph

Read from llama.cpp master (`src/models/*.cpp`, fetched for this survey) unless the
row says Ollama. Only the difference from something jitllm already runs is
listed; "exists" names the jitllm piece being reused.

- **qwen35 / qwen35moe** (qwen3.5, qwen3.6, qwen3.8, ornith, ornith-1.5,
  minicpm-v4.6 text). qwen3next's block, re-cut: `attn_qkv` + `attn_gate`
  instead of the legacy fused `ssm_in`, `ssm_alpha` and `ssm_beta` as two
  tensors instead of `ssm_ba`, `ssm.v_head_reordered` (a value-head ORDER
  flag the converter must honour -- a wrong order is a tensor of the right
  shape whose heads belong to other streams, RULE 7's fuseOrder lesson), a
  DENSE-FFN hybrid (qwen35)
  beside the MoE one, `rope.dimension_sections` with `mrope_interleaved`
  (identical to NEOX rope for text-only positions; qwen2vl already proved the
  equivalence), and `mtp.*` / `nextn` layers to set aside. The delta rule,
  conv, gated norm, attention out-gate and shared expert all exist.
- **granite / granitemoe** (10 + 2 families). llama plus four scalars:
  `embedding_scale` multiplies the embedding, `residual_scale` each residual
  branch, `attention.scale` replaces 1/sqrt(d), `logit_scale` divides logits.
  granitemoe adds a softmax top-k mixture (exists). granite4's dense tags carry
  `head_count_kv` as an ARRAY and `ssm.*` keys with no SSM tensors.
- **granitehybrid** (granite4 `-h` tags): granite + Mamba-2 layers (C11).
- **The classic-transformer archs (C6)**, all built from pieces the vision
  tower already has on host and device (LayerNorm{Part,Var,Apply}Rows,
  ungated GELU MLP with biases, biased q/k/v/o):
  - `phi2`: LayerNorm+bias, ONE norm feeding attention and FFN in parallel,
    fused biased qkv, partial rotary (32 of 80), GELU FFN, biased head.
  - `starcoder`: GPT-2 block -- learned `position_embd`, LayerNorm, MQA,
    fused biased qkv, GELU.
  - `starcoder2`: LayerNorm+bias, rope, GQA, biases everywhere, GELU FFN.
  - `falcon`: LayerNorm, parallel attention/FFN, fused qkv, NEOX rope, MQA
    (7b) or two input norms (`attn_norm_2`, 40b/180b), GELU.
  - `stablelm`: LayerNorm+bias, 25% partial rotary, qkv bias, parallel
    residual (`use_parallel_residual`), per-head QK LayerNorm on 12b.
  - `command-r`: bias-free LayerNorm, parallel attention/FFN from one norm,
    SwiGLU, `logit_scale`, per-head QK LayerNorm on command-r-plus.
  - `cohere2`: command-r + 3:1 SWA with rope ONLY on the sliding layers (C4a).
  - `nemotron` (nemotron-mini): LayerNorm1p (weight+1, bias), ReLU^2
    non-gated FFN, 50% partial rotary.
  - `dbrx`: bias-free LayerNorm pre and post attention, fused qkv,
    `clamp_kqv` 8.0, softmax top-4 mixture of SwiGLU experts (exists).
- **olmo2 / olmo3** (C7). No pre-norm at all: RMSNorm on the attention output
  and on the FFN output before each residual add, QK RMSNorm over the whole
  projection (olmoe's FlagQKNormWide exists), post-norm tensors exist
  (gemma2). olmo3 adds a 3:1 sliding pattern from the file and YaRN.
- **mistral3** (6 families). llama's graph plus (a) YaRN written as
  `rope.scaling.beta_fast/beta_slow/mscale/mscale_all_dim` (Ollama spelling;
  jitllm reads `yarn_beta_*`), and (b) attention temperature
  (`rope.scaling_beta` = 0.1 on the Ministral/Devstral-Small files):
  q *= 1 + beta * log(1 + floor(pos / original_ctx)). Pixtral tower is V5.
- **chatglm** (glm4, codegeex4) and **glmocr** text (C16). GLM-4-9B: fused
  biased qkv, rotary on HALF of each head, `ffn_up` carrying gate|up (split
  at conversion, as phi3's is), RMSNorm; pre-tokenizer `chatglm-bpe` has no
  table entry (its generating repo 404s -- a `handWritten` entry is the
  answer, as kimi-k2's was). GLM-OCR text is the GLM-4-0414 shape: sandwich
  post-norms and M-RoPE (`[16, 24, 24]`).
- **lfm2 / lfm2moe** (C10). Per-layer kinds: `head_count_kv[i] == 0` marks a
  ShortConv layer: in_proj -> (B, C, x), x*B, depthwise causal conv of width
  `shortconv.l_cache` = 3, *C, out_proj. The causal conv with a shifting
  history is `Conv1dRows`/`Conv1dShift` (exist, from qwen3next). Attention
  layers are GQA with QK RMSNorm; `token_embd_norm` is the final norm.
  lfm2moe adds sigmoid+bias routing (exists: DeepSeek-V3 router) with 2 dense
  lead blocks (exists: NDenseLead).
- **nemotron_h / nemotron_h_moe / nemotron_h_omni** (C11), also
  nemotron-3-ultra (cloud) and granite4 `-h`. Each layer is ONE of Mamba-2,
  attention (no rope), MLP or MoE. Mamba-2: in_proj -> (z, x, B, C, dt),
  causal conv1d WITH bias, selective scan over a [heads x head_dim x state]
  state with scalar-per-head A and dt softplus, D skip, group RMSNorm gated by
  z, out_proj. MoE: sigmoid + `exp_probs_b` + group top-k (exists), shared
  expert, NON-gated ReLU^2 experts (up/down only), and on nemotron-3-super a
  latent bottleneck (`ffn_latent_in`/`ffn_latent_out`) around the experts.
  The recurrent-state machinery (recPair, MigrateRec, hybrid prefix-cache
  rules) exists from qwen3next; the scan kernel does not, on any tier.
- **gemma4** (C14 + C4b + C3; vision/audio V10). Listed in C14. Two further
  facts from the file: Ollama serves two tokenizer encodings for it (tag 12b:
  `tokenizer.ggml.model = gemma4`; tag 31b: `llama` with pre `gemma4`), and
  the E2B/E4B tags put the vision AND audio towers in the same file.
- **gemma3n** (C15): AltUp (4 parallel streams, predict/correct with a
  router), LAuReL low-rank residual, per-layer token embeddings, the last 10
  layers sharing earlier layers' KV, gaussian top-k activation sparsity on the
  first layers. One family, 2.2M pulls, and the costliest dense graph here.
- **muse-glimmer** (C17a): llama-shaped with QK norm, sigmoid attention
  out-gate (exists: FlagAttnOutGate), sandwich post-norms (exist), 3:1 SWA
  with rope only on sliding layers, final softcap (exists) and `logit_scale`;
  tokenizer pre `llama4` (no table entry).
- **laguna** (C17b): sigmoid+bias router, shared expert, 1 dense lead,
  attention out-gate, QK norm; the sliding layers use a different rope width
  (128) and base (10000) from the global ones (64, 500000), and
  `head_count` is an array.
- **cohere2moe** (C17c): cohere2's block with a sigmoid-routed mixture and one
  dense lead block.
- **deepseekocr** text (C25): DeepSeek-V2's MoE (softmax, shared expert, dense
  lead) under ordinary MHA -- no MLA tensors.
- **mllama** text (C24): llama with gated cross-attention layers at the
  indices in `attention.cross_attention_layers`; with no image they are
  skipped, which is what transformers does.
- **solar** (solar-pro, Ollama-only arch, C19): llama plus four block skip
  connections (`bskcn_tv`, `attention.block_skip_connection.N`).
- **bert / nomic-bert / nomic-bert-moe** (C12a-c): post-LN encoder,
  bidirectional, token-type and absolute position embeddings (bert) or rope +
  SwiGLU + fused qkv (nomic), MoE every 2nd layer (nomic v2); pooling type
  from `pooling_type` (1 mean, 2 CLS, 3 last); WordPiece (`bert`) or Unigram
  (`t5`) tokenizers. `qwen3-embedding` (qwen3, pooling 3) and
  `embeddinggemma` (gemma3, `attention.causal = false`, mean pooling, two
  dense heads) need only C12a.
- **qwen4exp** (qwen3.8-flash-next, C20): qwen35's hybrid plus
  hyper-connections (4 streams, `hc_*`), an n-gram per-layer embedding table
  of 320M rows, a DSA-style indexer and 4x-compressed attention every 4th
  layer. Not worth pricing below XL until the pieces land elsewhere.

## 4. Capability catalogue

Cost scale: S = days, one package; M = a week or two, several packages or one
new kernel on all tiers; L = a new kernel family on host + three device
backends with gates; XL = several of those. "families" lists every family that
needs the capability, whether or not it is the last gap.

| cap | cost | what | families needing it | their pulls |
|---|---|---|---|---|
| C1 | S | llama.cpp's OWN architecture names for a graph jitllm runs: `internlm2` (llama), `exaone` (llama + NEOX rotary), `qwen3vl`/`qwen3vlmoe` (qwen3/qwen3moe for a text prompt), `mistral3` (llama; the temperature is C8). RE-SCOPED: the Ollama-only spellings that stood here (gptoss, glm4moelite, qwen25vl, attn_out, bare attn_sinks/ssm_dt, legacy ssm_in, head_count_kv arrays) block only Ollama's blobs and are dropped by decision | qwen3-vl, ministral-3, devstral-small-2, internlm2, exaone-deep, exaone3.5, bespoke-minicheck, mistral-medium-3.5 | 12.4M |
| C2 | -- | Dropped from the plan. It was Ollama's single-file multimodal packaging (the tower inside the text GGUF). Checking the artefacts showed every family it listed publishes an HF GGUF with the tower in a separate mmproj, so it blocks nothing but Ollama's copies | -- | -- |
| C3 | S | Tokenizer entries the standard artefacts need and jitllm lacks: pre `llama4` and `chatglm-bpe` (no pretok.Table entry, and the chatglm-bpe generating repo 404s -- a handWritten entry, as kimi-k2's was) and the `gemma4` tokenizer model. RE-SCOPED: resolving Ollama's `pre = "default"` is dropped with the rest of the Ollama dialect | gemma4, glm-ocr, llama4, glm4, codegeex4, muse-glimmer | 37.3M |
| C5 | S | **DONE.** Granite scalars: embedding_scale, residual_scale, attention.scale, logit_scale -- jlm.ArchGranite, host only (the device declines a residual scale by name). Teacher-forced against llama.cpp: granite-3.3-2b 20/20, granite-3.1-1b-a400m (granitemoe) 20/20, granite-4.0-micro 20/20; the logit scale moves no argmax, so a value gate holds the logits to llama-eval-callback. On the way: a per-layer head_count_kv array read as the MHA default, and the router ignoring a routed scale it was not built with | granite3.1-moe, granite4, granite3.3, granite3.1-dense, granite3-dense, granite3.2-vision, granite3-moe, granite4.1, granite3.2, granite3-guardian, granite4.2, granite4.1-guardian | 11.0M |
| C6 | M | Classic-transformer kit: LayerNorm (+bias, and LayerNorm1p), non-gated FFN (GELU / GELU-tanh / ReLU^2), biases on every projection, parallel attention+FFN residual, learned absolute position embeddings, partial rotary, clamp_kqv; then one arch entry each for phi2, starcoder, starcoder2, falcon, stablelm, command-r, cohere2, nemotron, dbrx | starcoder2, moondream, sqlcoder, dolphin-phi, granite-code, phi, command-r, starcoder, falcon, aya, stablelm2, stable-code, aya-expanse, dolphincoder, command-r-plus, nemotron-mini, falcon2, stablelm-zephyr, dbrx, command-r7b, alfred, command-a, command-r7b-arabic, north-mini-code-1.0 | 25.0M |
| C7 | S | OLMo2 block: post-norm only (no pre-norm), whole-projection QK RMSNorm | olmo2, olmo-3, olmo-3.1 | 4.6M |
| C8 | S | Attention temperature: q *= 1 + beta*log(1+floor(pos/L)) (llama4 NoPE layers, mistral3 yarn models) | llama4, mistral-small3.2, ministral-3, devstral-small-2, mistral-small3.1, mistral-medium-3.5, devstral-2 | 8.7M |
| C9 | S/M | **DONE.** qwen35/qwen35moe: qwen3next's gated delta net with split alpha/beta (fused back into ssm_ba at conversion) and qkv/gate tensors, the value heads TILED rather than grouped (jlm.FlagDeltaKeyTiled, host and all three device backends), dense-FFN hybrid variant, interleaved M-RoPE on attention layers (== NEOX for text), MTP blocks set aside. Teacher-forced against llama.cpp: Qwen3.5-0.8B 20/20, 2B 19/19, 4B 20/20 (4/20 with the grouped pairing), 9B 20/20, 0.8B-MTP 20/20, Qwen3.6-35B-A3B 19/20 (a 0.043 tie) | qwen3.5, qwen3.6, qwen3.8, ornith, ornith-1.5, qwen3.8-flash-next, minicpm-v4.6 | 31.6M |
| C10 | M | LFM2 ShortConv block (gated depthwise conv, L=3; reuses Conv1dRows/Conv1dShift) + hybrid layer kinds from head_count_kv==0 | lfm2.5-thinking, lfm2, lfm2.5 | 2.6M |
| C11 | L | Mamba-2 selective SSM (scan kernel on 4 tiers, conv1d+bias, D skip, grouped RMSNorm) + ReLU^2 experts + latent FFN (ffn_latent_in/out) | nemotron-3-super, granite4, nemotron-3-nano, nemotron3, nemotron-3.5-lightning, nemotron-cascade-2 | 6.4M |
| C13 | M | Llama 4 text graph (see the Llama 4 section) | llama4 | 1.8M |
| C14 | L | Gemma 4 text graph: per-layer head geometry, V RMSNorm, scale 1.0, proportional rope on global layers, KV-sharing layers, per-layer embeddings, layer_output_scale, parallel dense+MoE FFN with 3 extra norms and scaled router input, `gemma4` BPE tokenizer model | gemma4 | 25.8M |
| C15 | L | Gemma 3n graph: AltUp, LAuReL, per-layer embeddings, KV-sharing, activation sparsity | gemma3n | 2.2M |
| C16 | M | GLM family (chatglm GLM-4-9B, GLM-4-0414-style sandwich norms, glmocr text): fused biased qkv, fused gate|up split at conversion, GLM partial rotary, chatglm-bpe pre-tokenizer | glm-ocr, glm4, codegeex4 | 9.5M |
| C19 | S | solar-pro block skip connections (bskcn_tv) | solar-pro | 566K |
| C20 | XL | Sparse-attention/indexer + hyper-connection generation: DSA indexer (glm-dsa, minimax-m3), DeepSeek-V4 compressed attention + hyper-connections + sinks, qwen4exp (HC + n-gram per-layer embeddings + indexer + compressed layers) | qwen3.8-flash-next | 152K |
| C24 | S | mllama text path: skip cross-attention layers when no image (vision itself is V7) | llama3.2-vision | 5.4M |
| C25 | S/M | DeepSeek-V2 MHA (non-MLA) variant used by deepseekocr | deepseek-ocr | 540K |
| C4a | S | Per-layer attention schedule from the file: sliding_window_pattern arrays (global-first included), NoPE on global layers (rope only on SWA layers) | llama4, olmo-3, command-r7b, olmo-3.1, muse-glimmer, command-a, command-r7b-arabic, laguna-s-2.1, laguna-xs-2.1, north-mini-code-1.0, laguna-xs.2 | 4.4M |
| C4b | M | Per-layer geometry: head_count / head_count_kv / head_dim / rope dims / rope base that differ by layer kind (gemma4 256 vs 512 head_dim, laguna SWA rope 128 vs 64) | gemma4, laguna-s-2.1, laguna-xs-2.1, laguna-xs.2 | 26.1M |
| C12a | S | Embedding API for decoder models: bidirectional flag, pooling (mean / CLS / last), normalize; embeddinggemma's dense heads | nomic-embed-text, mxbai-embed-large, bge-m3, qwen3-embedding, all-minilm, snowflake-arctic-embed, embeddinggemma, nomic-embed-text-v2-moe, paraphrase-multilingual, snowflake-arctic-embed2, granite-embedding, bge-large | 125.3M |
| C12b | M | BERT-family encoder: post-LN block, token-type + absolute position embeddings, nomic-bert (rope, SwiGLU, fused qkv), nomic-bert-moe; WordPiece tokenizer (jlm.VocabWPM exists, tok has no implementation) | nomic-embed-text, mxbai-embed-large, bge-m3, all-minilm, snowflake-arctic-embed, nomic-embed-text-v2-moe, paraphrase-multilingual, snowflake-arctic-embed2, granite-embedding, bge-large | 119.0M |
| C12c | M | Unigram (T5 / XLM-R sentencepiece `t5`) tokenizer | bge-m3, nomic-embed-text-v2-moe, paraphrase-multilingual, snowflake-arctic-embed2, granite-embedding | 9.7M |
| C17a | S/M | muse-glimmer: sandwich norms + sigmoid attention out-gate + QK norm + NoPE global layers + final softcap + logit scale | muse-glimmer | 235K |
| C17b | M | laguna: sigmoid+bias router with shared expert and dense lead, attention out-gate, per-layer head counts and SWA rope geometry | laguna-s-2.1, laguna-xs-2.1, laguna-xs.2 | 316K |
| C17c | M | cohere2moe: cohere2 (LayerNorm, parallel, SWA, NoPE global) + sigmoid-routed MoE, dense lead | north-mini-code-1.0 | 60K |
| V0 | S/M | M-RoPE with real image positions in the text model (Config.RopeSections is stored and never consumed) | qwen3.5, glm-ocr, qwen3.6, qwen3-vl, qwen2.5vl, qwen3.8, ornith-1.5, qwen3.8-flash-next | 50.0M |
| V1 | S | mmproj with no clip.projector_type -> `mlp` (llama.cpp's default) | bakllava | 894K |
| V2 | M | Gemma 3 vision: SigLIP tower + avg-pool + soft-emb RMSNorm + linear projector; bidirectional attention over image tokens | gemma3, translategemma, medgemma, medgemma1.5 | 43.8M |
| V3 | M | Qwen2.5-VL tower: window attention, RMSNorm/SwiGLU ViT, merger | qwen2.5vl | 5.1M |
| V4 | M | Qwen3-VL tower: qwen3vl_merger + deepstack injection (qwen3-vl, qwen3.5/3.6/3.8, ornith-1.5) | qwen3.5, qwen3.6, qwen3-vl, qwen3.8, ornith-1.5, qwen3.8-flash-next | 37.3M |
| V5 | M | Pixtral tower (2-D rope ViT, patch merger) for mistral3 | mistral-small3.2, ministral-3, devstral-small-2, mistral-small3.1, mistral-medium-3.5 | 6.5M |
| V6 | M | Llama 4 vision tower + pixel-shuffle MLP projector | llama4 | 1.8M |
| V7 | L | mllama: tiled vision encoder + gated cross-attention layers | llama3.2-vision | 5.4M |
| V8 | M | MiniCPM-V resampler projector (+ minicpmv4_6) | minicpm-v, minicpm-v4.6, minicpm-v4.5 | 5.6M |
| V9 | M | LLaVA-NeXT anyres + multi-layer features (granite3.2-vision) | granite3.2-vision | 1.0M |
| V10 | L | Gemma 4 / Gemma 3n vision and audio encoders | gemma4 | 25.8M |
| V11 | L | OCR towers: GLM-V (glm-ocr), SAM+CLIP (deepseek-ocr) | glm-ocr, deepseek-ocr | 8.1M |
| V12 | L | Omni towers: nemotron3 (RADIO vision + Parakeet-style audio), muse-glimmer projector | nemotron3, muse-glimmer | 902K |

## 5. The order

Ranked greedily by **value / cost**: at each step, the capability whose
*completed* families (every remaining gap closed by it) carry the most pulls,
plus half-credit for families it moves closer, divided by its cost (S=1,
S/M=2, M=3, L=8, XL=20). Two tracks, because a family that produces text is
usable and a family that does not is not: track A closes text gaps only, track
B then adds vision/audio. RULE 6 applies to the ordering and not to admission:
nothing below is refused for being small, and a pull count decides ORDER.

★ **STALE SINCE THE RE-CLASSIFICATION (section 0): computed on the Ollama-blob
grading.** C2 is dropped, C1 and C3 are re-scoped (section 4), and six
families these rows credit to C1/C3 are SUPPORTED from their HF artefact. The
order actually worked, by the user's direction, was C9 then
C5, both done.

### Track A -- text

| # | cap | cost | families completed at this step | their pulls | cumulative (on top of today's 724.3M) | score |
|---|---|---|---|---|---|---|
| 1 | C2 | S | gemma3, translategemma, medgemma, medgemma1.5 | 43.8M | 43.8M | 54.67 |
| 2 | C12a | S | qwen3-embedding, embeddinggemma | 6.3M | 50.1M | 35.22 |
| 3 | C12b | M | nomic-embed-text, mxbai-embed-large, all-minilm, snowflake-arctic-embed | 109.0M | 159.1M | 37.17 |
| 4 | C1 | S | qwen3-vl, qwen2.5vl, qwen3-coder-next, glm-4.7-flash, internlm2, exaone-deep, qwen3-next, exaone3.5, bespoke-minicheck | 18.9M | 178.0M | 31.49 |
| 5 | C3 | S | gpt-oss, bge-large, gpt-oss-safeguard | 13.7M | 191.7M | 21.85 |
| 6 | C9 | S/M | qwen3.5, qwen3.6, qwen3.8, ornith, ornith-1.5, minicpm-v4.6 | 31.4M | 223.2M | 15.74 |
| 7 | C5 | S | granite3.1-moe, granite3.3, granite3.1-dense, granite3-dense, granite3.2-vision, granite3-moe, granite4.1, granite3.2, granite3-guardian, granite4.2, granite4.1-guardian | 9.5M | 232.6M | 9.84 |
| 8 | C6 | M | starcoder2, moondream, sqlcoder, dolphin-phi, granite-code, phi, command-r, starcoder, falcon, aya, stablelm2, stable-code, aya-expanse, dolphincoder, command-r-plus, nemotron-mini, falcon2, stablelm-zephyr, dbrx, alfred | 24.2M | 256.8M | 8.12 |
| 9 | C8 | S | mistral-small3.2, ministral-3, devstral-small-2, mistral-small3.1, mistral-medium-3.5, devstral-2 | 6.9M | 263.7M | 7.17 |
| 10 | C24 | S | llama3.2-vision | 5.4M | 269.1M | 5.40 |
| 11 | C7 | S | olmo2 | 3.8M | 272.9M | 3.99 |
| 12 | C12c | M | bge-m3, nomic-embed-text-v2-moe, paraphrase-multilingual, snowflake-arctic-embed2, granite-embedding | 9.7M | 282.6M | 3.25 |
| 13 | C16 | M | glm-ocr, glm4, codegeex4 | 9.5M | 292.1M | 3.17 |
| 14 | C4a | S | rnj-1, olmo-3, command-r7b, olmo-3.1, command-a, command-r7b-arabic | 2.0M | 294.1M | 2.61 |
| 15 | C4b | M | (enabler: completes nothing alone) | 0K | 294.1M | 2.18 |
| 16 | C14 | L | gemma4 | 25.8M | 319.9M | 3.23 |
| 17 | C10 | M | lfm2.5-thinking, lfm2, lfm2.5 | 2.6M | 322.5M | 0.85 |
| 18 | C11 | L | nemotron-3-super, granite4, nemotron-3-nano, nemotron3, nemotron-3.5-lightning, nemotron-cascade-2 | 6.4M | 328.9M | 0.80 |
| 19 | C13 | M | llama4 | 1.8M | 330.7M | 0.60 |
| 20 | C19 | S | solar-pro | 566K | 331.2M | 0.57 |
| 21 | C15 | L | gemma3n | 2.2M | 333.4M | 0.28 |
| 22 | C25 | S/M | deepseek-ocr | 540K | 334.0M | 0.27 |
| 23 | C17a | S/M | muse-glimmer | 235K | 334.2M | 0.12 |
| 24 | C17b | M | laguna-s-2.1, laguna-xs-2.1, laguna-xs.2 | 316K | 334.5M | 0.11 |
| 25 | C17c | M | north-mini-code-1.0 | 60K | 334.6M | 0.02 |
| 26 | C20 | XL | qwen3.8-flash-next | 152K | 334.7M | 0.01 |

**Read the first five rows as one bundle.** C2, C1 and C3 are converter and
tokenizer work only -- no kernel, no graph -- and together with the embedding
pair they add 191.7M of the 334.7M pulls the whole of track A adds (57%) for
one M and four S items. The ordering among them is
an artefact of the scoring; the dependency is not: C2 is what makes the other
Ollama-native files readable at all.

**Two cautions on the embedding rows.** `nomic-embed-text`'s 87M pulls are
real, but an embedding model is a different API (a vector, not a stream of
tokens), so C12a is a product decision -- where the vector comes out, in the
CLI and the server -- before it is a kernel one. And C12b's cost assumes the
LayerNorm/GELU/bias pieces are taken from C6 or from the vision tower; if C6
lands first, C12b shrinks to S/M.

**The Mamba-2 row (C11) is ranked low on pulls and is the most strategic item
on the list.** NVIDIA's whole Nemotron-3 line (super, nano, ultra, cascade,
lightning, the omni model) and IBM's granite4 hybrids are built on it, and they
are the newest families on the site; pull counts lag release dates.

### Track B -- vision and audio (after track A)

| # | cap | cost | families completed | their pulls | score |
|---|---|---|---|---|---|
| 1 | V2 | M | gemma3, translategemma, medgemma, medgemma1.5 | 43.8M | 14.61 |
| 2 | V0 | S/M | (enabler) | 0K | 6.25 |
| 3 | V4 | M | qwen3.5, qwen3.6, qwen3-vl, qwen3.8, ornith-1.5, qwen3.8-flash-next | 37.3M | 12.43 |
| 4 | V10 | L | gemma4 | 25.8M | 3.23 |
| 5 | V5 | M | mistral-small3.2, ministral-3, devstral-small-2, mistral-small3.1, mistral-medium-3.5 | 6.5M | 2.17 |
| 6 | V8 | M | minicpm-v, minicpm-v4.6, minicpm-v4.5 | 5.6M | 1.87 |
| 7 | V3 | M | qwen2.5vl | 5.1M | 1.70 |
| 8 | V11 | L | glm-ocr, deepseek-ocr | 8.1M | 1.02 |
| 9 | V1 | S | bakllava | 894K | 0.89 |
| 10 | V7 | L | llama3.2-vision | 5.4M | 0.68 |
| 11 | V6 | M | llama4 | 1.8M | 0.60 |
| 12 | V9 | M | granite3.2-vision | 1.0M | 0.33 |
| 13 | V12 | L | nemotron3, muse-glimmer | 902K | 0.11 |

V0 (M-RoPE with real image positions) completes nothing alone and gates four
towers; AGENTS.md already records it as a known divergence on qwen2vl, where it
was left unbuilt because no answer had been shown to depend on it. Qwen2.5-VL,
Qwen3-VL and the qwen3.5 line are trained with it, so that condition is now met
by 50M pulls of families.

### Tag completeness -- not ranked, because no default tag needs it

| cap | cost | what it adds |
|---|---|---|
| C18a | S/M | `q4_1` and `q5_1` tags: 127 and 128 families publish them. Both are Q4_0/Q5_0 plus a per-block minimum -- the d-plane layout already has a slot for a second scalar (it is what the k-quants' dmin uses), so this is a packer and emitter variant, not a new shape. |
| C18b | M | `q2_K` tags on 125 families; the earlier survey (model-coverage.md section G) already prices Q2_K as the only K-quant worth adding. |
| -- | out of scope | `nvfp4`, `mxfp8`, `int4`, `mlx*` tags (11 families): these are NOT GGUF. Their manifests carry hundreds of `application/vnd.ollama.image.tensor` layers for Ollama's MLX engine -- a different source format and two new weight types. Every such family also publishes a GGUF tag. |

## 6. Llama 4, in full

**Status: the TEXT graph (items 1-9) is built, on the host and on
CUDA and Vulkan; see RULE 7 and `docs/engineering-history/model-correctness.md`.
Not built: vision (11), Ollama's single-file blob (10, which is C2 above and
untested against llama4's own `v.*` names), and the KV ring (8). Verified on
unsloth's llama.cpp-converted Scout GGUFs. The spike below is kept as
written.**

`llama4` (Scout `16x17b` = 108.6B, Maverick `128x17b` = 401.6B, 1.8M pulls,
vision + tools). Read from llama.cpp's `llama4.cpp`/`llama-graph.cpp` and from
the two Ollama headers. Each item names what jitllm has to build on.

**Text graph**
1. **Interleaved attention ("iRoPE")**: layers with `(il+1) % 4 != 0` use
   RoPE and CHUNKED attention; every 4th layer uses NO positional encoding and
   full causal attention. The chunk is `attention.chunk_size` = 8192 in the
   Ollama file (llama.cpp hardcodes it): a token attends only to keys in its
   own 8192-aligned chunk, which is a different mask from a sliding window --
   position 8192 sees one key, not 8192. Builds on: SWAWindow/SWAPeriod (the
   period exists, the chunked mask does not), FlagNoPosEnc (whole-model today,
   needs to become per-layer: C4a).
2. **Attention temperature on the NoPE layers**: q *= 1 +
   0.1 * log(1 + floor((pos + 1) / 8192)) (llama.cpp's
   `f_attn_temp_scale = 0.1`, `n_attn_temp_floor_scale = 8192`, offset 1).
   Neither key is in the Ollama file; the constants are the model's. Shared
   with mistral3 (C8).
3. **QK L2-norm without weights**, after RoPE, on RoPE layers only, and only
   when `use_qk_norm` is true (Scout: true; Maverick: false). It is a plain
   RMSNorm with eps and no gain -- jitllm's QK norm always carries a weight.
4. **Interleaved dense/MoE layers**: `interleave_moe_layer_step` = 1 (Scout:
   every layer MoE) or 2 (Maverick: layer i is MoE when i+1 is even, with expert
   FFN 8192; the others are dense with FFN 16384). NDenseLead expresses a dense PREFIX, not
   a period.
5. **Router**: top-1 (`expert_used_count` = 1) of 16 or 128, gating by
   SIGMOID of the selected logit, **no renormalisation**, and the weight is
   applied to the expert's INPUT before the FFN (`weight_before_ffn`, the one
   architecture in llama.cpp that does this) rather than to its output. With
   SiLU experts that is not the same function. Builds on: FlagExpertSigmoid,
   the generated top-k kernel.
6. **Shared expert** on every MoE layer, ungated, same width as an expert,
   summed with the routed output. Exists (DeepSeek's ungated shared expert).
7. **RoPE**: base 500000, llama3-style frequency factors where the file ships
   `rope_freqs` (the Maverick header ships none; check Scout's before
   assuming). Exists (RoleRopeFreqs).
8. **Context**: 1M (Maverick) / 10M (Scout) positions. The chunked layers need
   at most 8192 positions of KV each, so 3/4 of the cache can be a ring --
   worth doing at the same time as (1), because a 10M-position dense cache is
   the thing that makes this model unusable on the hardware jitllm targets.

**Tokenizer**
9. `tokenizer.ggml.model = gpt2`, 202,048 tokens, BOS 200000, and
   `tokenizer.ggml.pre = "default"` -- which jitllm resolves to llama3's split.
   Llama 4's pattern is the o200k-style case-splitting one llama.cpp names
   `llama4`; `pretok.Table` has no entry because `scripts/gentok` could not
   fetch the gated repo ("gated repo: needs HF_TOKEN"). RULE 11: fetch it with
   a token, do not record the absence. The same entry unblocks muse-glimmer.

**Conversion**
10. The Ollama blob is single-file: 34 vision blocks (`v.blk.N.*`,
    `attn_norm`/`ffn_norm` with biases, `mlp.fc1`/`fc2`) and `mm.linear_1`
    live beside the text tensors, with `llama4.vision.*` keys (C2).

**Vision (V6)**
11. ViT: 336 px, patch 14, 1408 wide, 16 heads, LayerNorm with biases, GELU
    MLP, 2-D rotary over the patch grid (`vision.rope.freq_base` 10000 --
    FlagVisionRope exists for Qwen2-VL), pixel shuffle with ratio 0.5 (the
    idefics3 projector's shuffle exists), then an MLP projector into 5120.
    Images are tiled; each tile is 144 tokens after the shuffle.

**Placement**
12. 108.6B / 401.6B at Q4_K_M is 67 / 245 GB: both are over-committed on every
    machine this project has, so they run through the v26 expert pages. Top-1
    routing is the friendliest case that pager has seen (1 of 16 or 128 per
    token, against qwen3next's 10 of 512).

Cost: text graph M (items 1-7 are each S; the chunked mask and the per-layer
NoPE are the only new attention behaviour), tokenizer S, conversion S (C2),
vision M. Every text item has a second consumer on the site, which is why it is
cheaper than it looks: chunked/NoPE (cohere2, muse-glimmer), temperature
(mistral3), sigmoid top-1 (none -- unique), weightless QK norm (none).

## 7. Cloud-only families

Ollama serves only `:cloud` tags for these: no local weights to pull. The
architecture comes from the model's own `config.json` on HuggingFace; weights
(and community GGUFs) exist there, so "run it locally" means bringing them.
All are 229B-2.8T.

| family | pulls | arch (config.json) | missing |
|---|---|---|---|
| kimi-k2.6 | 481.7K | `kimi_k25`: text is `kimi_k2` = DeepseekV3 (MLA, V3 router) -- a graph jitllm runs -- plus MoonViT; 1.04T | vision (MoonViT, llama.cpp `kimik25`) |
| kimi-k2.7-code | 243.3K | same as kimi-k2.6 | vision |
| mistral-large-3 | 120.8K | `mistral4` in llama.cpp, which reuses deepseek2's loader and graph; 675B | C23 (alias), vision |
| minimax-m2.7 | 2.4M | `minimax_m2`: full-attention MoE, 256 experts top-8, per-layer QK norm, sigmoid+bias router; 229B | C22 |
| nemotron-3-ultra | 107.6K | `nemotron_h`: Mamba-2 + MoE; 550B | C11 |
| kimi-k3 | 90.4K | `kimi_k3` = kimi_linear (KDA + MLA -- jitllm runs Kimi-Linear) + 896 experts + additions; 2.81T | C21 |
| glm-5.2, glm-5.3 | 355.7K, 61.8K | `glm_moe_dsa`: MLA + DeepSeek sparse-attention indexer; 756B | C20 |
| glm-5.3-flash | 154.3K | `glm5_next`: KDA + MLA + DSA indexer + hyper-connections; 321B | C20, C21 |
| deepseek-v4-pro | 426K | `deepseek_v4`: hyper-connections, compressed attention, indexer, SWA 128, sinks; 1.65T | C20 |
| deepseek-v4.1-flash | 52.8K | `deepseek_v41`: v4 + compressed "engram" vocabulary; 763B | C20 |
| minimax-m3 | 515.8K | `minimax_m3_vl`: sparse attention + vision | C20 |

## 8. Where this could be wrong

- **Light-mode tags.** Only each family's first short tag had its full tensor
  table read; the others had their KV section only. A size that differs in
  TENSORS but not in keys (a tied head, a missing bias) is not seen. The
  architecture and tokenizer of every short tag are read, not inferred.
- **"SUPPORTED" is a statement about lists.** It says every name the file uses
  is one the converter accepts. It does not say this checkpoint's logits match
  a reference; RULE 7's goldens are per architecture. The first act of any
  item here is to convert the default tag of each family it claims and run
  `TestGreedyMatchesLlamaCpp`-style teacher forcing on it -- and that should
  include a sample of the 122, because Ollama's converter has already been
  shown to spell things differently from llama.cpp's.
- **Tokenizer labels are claims** (RULE 7m). Ollama writes `default` where
  llama.cpp writes a name, and some labels look wrong for their vocabulary
  (`mistral-small` 24b says `qwen2` over a Tekken vocabulary; `sailor2` says
  `llama-bpe` over a Qwen vocabulary; `olmo-3` and `glm-ocr` say `llama-bpe`).
  C3 is priced S on the assumption that the vocabulary corroborates a fix;
  where it cannot, `tok.WithTokenizer` with the model's own tokenizer.json is
  the answer the project already has.
- **Costs are estimates from reading graphs**, anchored to what the earlier
  items actually cost (qwen3next, deepseek2, kimilinear). The device tiers
  are included in every M/L: RULE 8a says a graph feature on one tier is an
  absence on the other.
- **Pull counts are lifetime**, per family, and favour old models (section 1).
- **The 2026 architectures** (gemma4, qwen35, qwen4exp, muse-glimmer, laguna,
  cohere2moe, deepseek4, glm-dsa, kimi-k3, minimax-m3) were read from
  llama.cpp master at the time of the survey; several of those graphs are weeks old and
  may still move.

## 9. The Hugging Face artefacts behind the re-classification

Read with HTTP range requests -- KV section and tensor table, never a
weight byte -- and fed to the converter's own `configOf`, `identify` and
`typeOf` through `meta.New` (a throwaway test, not checked in). "converts"
means all three accepted the header; it is a statement about the lists, exactly
as section 1 says of SUPPORTED, and it is not a teacher-forced run.

| family | artefact | arch / pre / mmproj projector | converter says |
|---|---|---|---|
| gemma3 | ggml-org/gemma-3-4b-it-GGUF | gemma3 / SPM / `gemma3` | text converts; projector not in visionOf (V2) |
| translategemma | mradermacher/translategemma-4b-it-GGUF | gemma3 / SPM / `gemma3` | text converts (V2) |
| medgemma, medgemma1.5 | unsloth/medgemma-4b-it-GGUF, unsloth/medgemma-1.5-4b-it-GGUF | gemma3 / SPM / `gemma3` | text converts (V2) |
| gpt-oss | ggml-org/gpt-oss-20b-GGUF (the MXFP4 file on this box) | gpt-oss / gpt-4o | converts; runs on host and every device |
| gpt-oss-safeguard | unsloth/gpt-oss-safeguard-20b-GGUF | gpt-oss / gpt-4o | converts |
| qwen2.5vl | ggml-org/Qwen2.5-VL-3B-Instruct-GGUF | qwen2vl / qwen2 / `qwen2.5vl_merger` | text converts; projector not in visionOf (V3) |
| qwen3-vl | Qwen/Qwen3-VL-2B-Instruct-GGUF, Qwen/Qwen3-VL-30B-A3B-Instruct-GGUF | qwen3vl, qwen3vlmoe / qwen2 / `qwen3vl_merger` | refused: architecture name (C1) |
| qwen3-coder-next | unsloth/Qwen3-Coder-Next-GGUF | qwen3next / qwen2 | converts (`ssm_dt.bias`, split attn_qkv/attn_gate, MXFP4 shared expert) |
| qwen3-next | unsloth/Qwen3-Next-80B-A3B-Thinking-GGUF | qwen3next / qwen2 | converts |
| glm-4.7-flash | unsloth/GLM-4.7-Flash-GGUF | deepseek2 / glm4 | converts (key_length 576, key_length_mla 256: llama.cpp's order) |
| mistral-small3.2 | bartowski/mistralai_Mistral-Small-3.2-24B-Instruct-2506-GGUF | llama / tekken / `pixtral` | text converts (V5) |
| mistral-small3.1 | bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF | llama / tekken / `pixtral` | text converts (V5) |
| devstral-2 | ggml-org/Devstral-2-123B-Instruct-2512-GGUF | llama / tekken | converts (YaRN x64 with yarn_log_multiplier 1.0 -- the multiplier's reading is unverified on this model) |
| ministral-3 | mistralai/Ministral-3-3B-Instruct-2512-GGUF | mistral3 / tekken / `pixtral`; attention.temperature_scale 0.1 | refused: name (C1) and temperature (C8) |
| devstral-small-2 | bartowski/mistralai_Devstral-Small-2-24B-Instruct-2512-GGUF | mistral3 / tekken; temperature_scale 0.1 | refused (C1, C8) |
| mistral-medium-3.5 | bartowski/mistralai_Mistral-Medium-3.5-128B-GGUF | mistral3 / pixtral / `pixtral`; temperature_scale 0.0 | refused: name only (C1) |
| internlm2 | bartowski/internlm2_5-1_8b-chat-GGUF | internlm2 / SPM | refused: name (C1) |
| exaone3.5, exaone-deep | bartowski/EXAONE-3.5-2.4B-Instruct-GGUF, bartowski/LGAI-EXAONE_EXAONE-Deep-2.4B-GGUF | exaone / exaone | refused: name (C1); the pre resolves through the legacy starcoder family |
| rnj-1 | bartowski/EssentialAI_rnj-1-instruct-GGUF | gemma3 / llama-bpe; no sliding_window key, YaRN x4 | **converted and was wrong**; fixed -- an absent window key is no sliding layers, as in llama.cpp |
| olmo-3 | unsloth/Olmo-3-7B-Instruct-GGUF | olmo2 / dbrx | refused: C7 |
| qwen3.5, qwen3.6, qwen3.8, ornith-1.5, minicpm-v4.6 | unsloth/Qwen3.5-2B-GGUF, unsloth/Qwen3.6-35B-A3B-GGUF, unsloth/Qwen3.8-27B-GGUF, ornith-ai/Ornith-1.5-9B-GGUF, openbmb/MiniCPM-V-4.6-gguf | qwen35, qwen35moe / qwen35 / `qwen3vl_merger`, `minicpmv4_6` | refused: architecture, and `ssm_alpha`/`ssm_beta` have no role (C9) |
| llama4 | unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF | llama4 / llama4 / `llama4` | refused (C13); pre `llama4` has no entry (C3) |
| gemma4 | ggml-org/gemma-4-E4B-it-GGUF | gemma4 / tokenizer model `gemma4` | refused (C14, C3) |
| glm-ocr | ggml-org/GLM-OCR-GGUF | glm4 / chatglm-bpe / `glm4v` | refused (C16, C3) |
| deepseek-ocr | ggml-org/DeepSeek-OCR-GGUF | deepseek2-ocr / deepseek-v3 / `deepseekocr` | refused (C25) |
| nemotron-3-nano (and the rest of the nemotron_h line) | unsloth/Nemotron-3-Nano-30B-A3B-GGUF | nemotron_h_moe / pixtral | refused (C11); the pre resolves |
| glm4, codegeex4 | bartowski/glm-4-9b-chat-GGUF | chatglm / chatglm-bpe | refused (C16, C3) |
| muse-glimmer | meta-models/Muse-Glimmer-30B-GGUF | muse-glimmer / llama4 | refused (C17a, C3) |
| bge-large | CompendiumLabs/bge-large-en-v1.5-gguf | bert / WordPiece | refused (C12a/b) |

bespoke-minicheck has no HF GGUF; its safetensors are InternLM2 with remote
code, which hfArchTable does not carry, and a GGUF of it says `internlm2` (C1).
nemotron3 (omni), nemotron-3-super, nemotron-3.5-lightning and
nemotron-cascade-2 were graded from nemotron-3-nano's artefact: same line, same
tokenizer.
