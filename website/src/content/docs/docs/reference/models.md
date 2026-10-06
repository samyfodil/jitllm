---
title: Supported models
description: The architectures jitllm runs, the weight formats it reads, and the catalog convert fetches by name.
---

jitllm supports a fixed list of architectures, each one checked against a reference implementation rather than assumed from matching tensor names. Any GGUF or Hugging Face checkpoint of a listed architecture converts, not only the ones in the catalog.

## Architectures

### Dense transformers

| Architecture | Models |
|---|---|
| `llama` | Llama 2, 3, 3.1, 3.2, 3.3; Mistral 7B; SmolLM2; TinyLlama; the text models of SmolVLM and LLaVA; ERNIE 4.5 0.3B (`ernie4_5`) |
| `mistral3` | Mistral Small 3.1/3.2, Magistral, Devstral, Ministral 3 |
| `qwen2`, `qwen3` | Qwen2, Qwen2.5, Qwen3; Qwen3-Embedding |
| `gemma`, `gemma2`, `gemma3`, `gemma3n`, `gemma4` | Gemma 1, 2, 3, 3n and 4; EmbeddingGemma |
| `phi2`, `phi3` | Phi-2; Phi-3, Phi-3.5, Phi-4 |
| `glm4` | GLM-4-0414 (9B, 32B), GLM-Z1, and the text models of GLM-4.1V and GLM-4.6V-Flash |
| `granite` | IBM Granite 3.x and 4.0 |
| `command-r`, `cohere2` | Command-R, Aya, Command-R7B, Command-A |
| `smollm3`, `arcee`, `apertus` | SmolLM3, Arcee AFM-4.5B, Apertus |
| `olmo2`, `exaone4`, `seed_oss` | OLMo 2 and OLMo 3, EXAONE 4.0, Seed-OSS |
| `hunyuan-dense` | Tencent Hunyuan (0.5B to 7B) |
| `starcoder`, `starcoder2` | StarCoder, StarCoderBase, StarCoder2 |
| `stablelm`, `falcon`, `nemotron` | StableLM 2, Falcon, Nemotron |

### Mixtures of experts

| Architecture | Models |
|---|---|
| `qwen3moe`, `qwen2moe` | Qwen3 MoE; Qwen1.5-MoE, Qwen2-57B-A14B |
| `olmoe`, `dbrx`, `phimoe` | OLMoE, DBRX, Phi-3.5-MoE; Mixtral 8x7B through the llama graph |
| `gpt-oss` | gpt-oss 20B and 120B |
| `llama4` | Llama 4 Scout (text) |
| `granitemoe` | IBM Granite mixtures |
| `glm4moe` | GLM-4.5, GLM-4.6, GLM-4.5-Air, and the text models of GLM-4.5V and GLM-4.6V |
| `ernie4_5-moe`, `hunyuan-moe` | ERNIE 4.5 (21B-A3B, 300B-A47B), Hunyuan mixtures |
| `minimax-m2`, `minimax-m3` | MiniMax-M2, M2.1, M2.5 and M3 (text) |
| `bailingmoe2`, `dots1` | Ling and Ring 2.0, dots.llm1 |
| `deepseek2`, `deepseek32`, `deepseek4` | DeepSeek-V2, V3, R1, V3.2 and V4; DeepSeek-Coder-V2; Kimi K2; Moonlight; GLM-4.7-Flash; Kimi-VL (latent attention) |

### Hybrids and state-space models

| Architecture | Models |
|---|---|
| `qwen3next`, `qwen35`, `qwen35moe` | Qwen3-Next, Qwen3.5, Qwen3.6: gated linear attention beside attention |
| Kimi-Linear, `kimi-k3` | Kimi-Linear (from safetensors), Kimi-K3 (text): Kimi Delta Attention beside latent attention |
| `mamba`, `mamba2` | Mamba-1 and FalconMamba; Mamba-2 and Mamba-Codestral |
| `jamba`, `falcon-h1` | Jamba; Falcon-H1 |
| `granitehybrid` | IBM Granite 4.0-H |
| `nemotron_h`, `nemotron_h_moe` | Nemotron-H, Nemotron Nano 2, Nemotron 3 Nano |
| `lfm2`, `lfm2moe` | Liquid LFM2 and LFM2-8B-A1B |

### Vision and embeddings

| Architecture | Models |
|---|---|
| `qwen2vl`, `qwen3vl`, `qwen3vlmoe` | Qwen2-VL, Qwen2.5-VL, Qwen3-VL (2B to 235B-A22B) |
| `bert`, `nomic-bert` | sentence-embedding encoders (all-MiniLM, mxbai-embed, nomic-embed) |

Vision towers convert beside their language model into one container, and their blocks page and place like any other block.

| Projector | Families |
|---|---|
| `idefics3` | SmolVLM, Idefics3 |
| `mlp` | LLaVA-style models (a CLIP or SigLIP tower and a two-layer MLP) |
| `qwen2vl_merger`, `qwen2.5vl_merger`, `qwen3vl_merger` | Qwen2-VL, Qwen2.5-VL, Qwen3-VL |
| `glm4v` | GLM-4.1V, GLM-4.5V, GLM-4.6V |
| `gemma3`, `gemma3nv`, `gemma4v` | Gemma 3, Gemma 3n (MobileNet-V5), Gemma 4 |
| `internvl` | InternVL 2.5 and 3 |
| `resampler` | MiniCPM-V 2.6, 4.0, 4.5 |
| `kimivl` | Kimi-VL (MoonViT) |
| `janus_pro`, `pixtral`, `phi4`, `llama4` | Janus-Pro; Pixtral and Mistral 3; Phi-4 reasoning vision; Llama 4 |

From Hugging Face safetensors, these model classes convert directly: Llama, Mixtral, Qwen2, Qwen2Moe, Qwen3, Qwen3Moe, Olmoe, Olmo2, Olmo3, Gemma, Phi3, PhiMoE, DeepseekV2, DeepseekV3, Glm4Moe, Glm4MoeLite, KimiLinear, Ernie4_5, Ernie4_5_Moe, HunYuanDenseV1, HunYuanMoEV1, Dots1, Apertus, SmolLM3, Arcee, SeedOss, Exaone4 and Ministral3. Every other architecture converts from GGUF.

An architecture that is not on the list is refused at conversion with its name, not converted into something that runs and produces wrong text.

## Weight formats

| Format | Notes |
|---|---|
| F32, F16, BF16 | unquantized; `-q8` turns a safetensors model's matrices into Q8_0 |
| Q8_0, Q4_0, Q5_0, Q5_1 | legacy block formats |
| Q3_K, Q4_K, Q5_K, Q6_K | K-quants, as in every `Q4_K_M` or `Q5_K_M` file |
| MXFP4 | gpt-oss's expert weights |

Conversion keeps the source's quantization and changes only its layout, so a `.jlm` is about the size of the GGUF it came from.

## The catalog

`jitllm convert NAME out.jlm` fetches these by name, pinned to a revision checked against this engine. `jitllm library` prints the same list.

| Name | Parameters | Architecture | Weights | Download | |
|---|---|---|---|---|---|
| `smollm2-360m-instruct` | 360M | llama | Q8_0 | 386 MB |  |
| `llama-3.2-1b-instruct` | 1B | llama | Q4_K_M | 808 MB |  |
| `smollm2-1.7b-instruct` | 1.7B | llama | Q4_K_M | 1.1 GB |  |
| `llama-3.2-3b-instruct` | 3B | llama | Q4_K_M | 2.0 GB |  |
| `mistral-7b-instruct-v0.3` | 7B | llama | Q4_K_M | 4.4 GB |  |
| `llama-3.1-8b-instruct` | 8B | llama | Q4_K_M | 4.9 GB |  |
| `llama-3.3-70b-instruct` | 70B | llama | Q4_K_M | 42.5 GB |  |
| `qwen2.5-1.5b-instruct` | 1.5B | qwen2 | Q4_K_M | 1.1 GB |  |
| `qwen2.5-7b-instruct` | 7B | qwen2 | Q4_K_M | 4.7 GB |  |
| `qwen2-vl-2b-instruct` | 2B | qwen2vl | Q4_K_M | 1.7 GB | images |
| `qwen3-0.6b` | 0.6B | qwen3 | Q8_0 | 639 MB |  |
| `qwen3-1.7b` | 1.7B | qwen3 | Q8_0 | 1.8 GB |  |
| `qwen3-4b` | 4B | qwen3 | Q4_K_M | 2.5 GB |  |
| `qwen3-8b` | 8B | qwen3 | Q4_K_M | 5.0 GB |  |
| `qwen3-14b` | 14B | qwen3 | Q4_K_M | 9.0 GB |  |
| `qwen3-30b-a3b` | 30B-A3B | qwen3moe | Q4_K_M | 18.6 GB |  |
| `qwen3-next-80b-a3b-instruct` | 80B-A3B | qwen3next | Q4_K_M | 48.5 GB |  |
| `qwen3.5-2b` | 2B | qwen35 | Q4_K_M | 1.3 GB |  |
| `qwen3.5-4b` | 4B | qwen35 | Q4_K_M | 2.7 GB |  |
| `qwen3.6-35b-a3b` | 35B-A3B | qwen35moe | Q4_K_M | 22.1 GB |  |
| `phi-2` | 2.7B | phi2 | Q4_K_M | 1.8 GB |  |
| `starcoderbase-1b` | 1B | starcoder | Q4_K_M | 793 MB |  |
| `aya-expanse-8b` | 8B | command-r | Q4_K_M | 5.1 GB |  |
| `command-r7b` | 7B | cohere2 | Q4_K_M | 5.1 GB |  |
| `starcoder2-3b` | 3B | starcoder2 | Q4_K_M | 1.9 GB |  |
| `falcon-7b-instruct` | 7B | falcon | Q4_K_M | 5.0 GB |  |
| `qwen1.5-moe-a2.7b-chat` | 14.3B-A2.7B | qwen2moe | Q4_K_M | 9.5 GB |  |
| `ernie-4.5-21b-a3b` | 21B-A3B | ernie4_5-moe | Q4_K_M | 13.3 GB |  |
| `ernie-4.5-0.3b` | 0.3B | ernie4_5 | Q4_K_M | 241 MB |  |
| `hunyuan-0.5b-instruct` | 0.5B | hunyuan-dense | Q4_K_M | 355 MB |  |
| `ling-mini-2.0` | 16B-A1.4B | bailingmoe2 | Q4_K_M | 9.9 GB |  |
| `phi-mini-moe-instruct` | 7.6B-A2.4B | phimoe | Q4_K_M | 5.0 GB |  |
| `apertus-8b-instruct` | 8B | apertus | Q4_K_M | 5.1 GB |  |
| `granite-4.0-h-350m` | 350M | granitehybrid | Q8_0 | 366 MB |  |
| `nemotron-nano-9b-v2` | 9B | nemotron_h | Q4_K_M | 6.5 GB |  |
| `nemotron-3-nano-30b-a3b` | 30B-A3B | nemotron_h_moe | Q4_K_M | 24.7 GB |  |
| `falcon-h1-0.5b-instruct` | 0.5B | falcon-h1 | Q4_K_M | 315 MB |  |
| `lfm2-350m` | 350M | lfm2 | Q4_K_M | 229 MB |  |
| `lfm2-8b-a1b` | 8B-A1B | lfm2moe | Q4_K_M | 5.0 GB |  |
| `mamba-codestral-7b` | 7B | mamba2 | Q4_K_M | 4.1 GB |  |
| `falcon-mamba-7b` | 7B | mamba | Q4_K_M | 4.2 GB |  |
| `jamba-reasoning-3b` | 3B | jamba | F16 | 6.4 GB |  |
| `nemotron-mini-4b` | 4B | nemotron | Q4_K_M | 2.7 GB |  |
| `smollm3-3b` | 3B | smollm3 | Q4_K_M | 1.9 GB |  |
| `afm-4.5b` | 4.5B | arcee | Q4_K_M | 2.9 GB |  |
| `seed-oss-36b` | 36B | seed_oss | Q4_K_M | 21.8 GB |  |
| `olmo-2-1b` | 1B | olmo2 | Q4_K_M | 936 MB |  |
| `exaone-4.0-1.2b` | 1.2B | exaone4 | Q4_K_M | 812 MB |  |
| `ministral-3-3b` | 3B | mistral3 | Q4_K_M | 2.1 GB |  |
| `stablelm-2-1.6b` | 1.6B | stablelm | Q4_K_M | 1.0 GB |  |
| `granite-3.3-2b` | 2B | granite | Q4_K_M | 1.5 GB |  |
| `granite-4.0-micro` | 3B | granite | Q4_K_M | 2.1 GB |  |
| `granite-3.1-1b-a400m` | 1B-A400M | granitemoe | Q4_K_M | 822 MB |  |
| `deepseek-v2-lite` | 16B-A2.4B | deepseek2 | Q4_K_M | 10.4 GB |  |
| `deepseek-coder-v2-lite-instruct` | 16B-A2.4B | deepseek2 | Q4_K_M | 10.4 GB |  |
| `moonlight-16b-a3b-instruct` | 16B-A3B | deepseek2 | Q4_K_M | 10.5 GB |  |
| `olmoe-1b-7b-instruct` | 7B-A1B | olmoe | Q4_K_M | 4.2 GB |  |
| `gemma-2-2b-it` | 2B | gemma2 | Q4_K_M | 1.7 GB |  |
| `gemma-2-9b-it` | 9B | gemma2 | Q4_K_M | 5.8 GB |  |
| `gemma-3-1b-it` | 1B | gemma3 | Q4_K_M | 806 MB |  |
| `gemma-3-4b-it` | 4B | gemma3 | Q4_K_M | 2.5 GB |  |
| `gemma-3-12b-it` | 12B | gemma3 | Q4_K_M | 7.3 GB |  |
| `gemma-3n-E2B-it` | 5B (E2B) | gemma3n | Q4_K_M | 3.0 GB |  |
| `phi-3.5-mini-instruct` | 3.8B | phi3 | Q4_K_M | 2.4 GB |  |
| `phi-4-mini-instruct` | 3.8B | phi3 | Q4_K_M | 2.5 GB |  |
| `phi-4` | 14B | phi3 | Q4_K_M | 9.1 GB |  |
| `llama-4-scout-17b-16e-instruct` | 109B-A17B | llama4 | Q3_K_S | 46.7 GB |  |
| `gpt-oss-20b` | 21B-A3.6B | gpt-oss | MXFP4 | 12.1 GB |  |
| `gpt-oss-120b` | 117B-A5.1B | gpt-oss | MXFP4 | 63.4 GB |  |
| `all-minilm-l6-v2` | 22M | bert | F16 | 46 MB | embeddings |
| `nomic-embed-text-v1.5` | 137M | nomic-bert | F16 | 274 MB | embeddings |
| `mxbai-embed-large-v1` | 335M | bert | F16 | 670 MB | embeddings |
| `qwen3-embedding-0.6b` | 0.6B | qwen3 | F16 | 1.2 GB | embeddings |
| `smolvlm-256m-instruct` | 256M | llama | Q8_0 | 279 MB | images |
| `smolvlm-500m-instruct` | 500M | llama | Q8_0 | 546 MB | images |
| `llava-phi-3-mini` | 3.8B | llama | int4 | 2.9 GB | images |
| `qwen3-vl-2b-instruct` | 2B | qwen3vl | Q4_K_M | 1.6 GB | images |
| `qwen3-vl-30b-a3b-instruct` | 30B-A3B | qwen3vlmoe | Q4_K_M | 19.3 GB | images |
| `glm-4.6v-flash` | 9B | glm4 | Q4_K_M | 7.1 GB | images |
| `glm-4-9b-0414` | 9B | glm4 | Q4_K_M | 6.2 GB |  |
| `kimi-vl-a3b-thinking-2506` | 16B-A3B | deepseek2 | Q4_K_M | 11.2 GB | images |

The largest (Llama 3.3 70B, Qwen3-Next 80B, gpt-oss 120B) are larger than the memory of an ordinary desktop. They run anyway, [paged from disk](/docs/guides/memory/#running-larger-than-ram), at a speed set by the drive.
