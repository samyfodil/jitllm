# Model and weight format reference

[Back to the README](../README.md)

## Supported models

`convert` reads 65 GGUF text architecture names (57 graphs) and 29 Hugging Face
safetensors class names. The converter's list is the list: an architecture not on it
is refused at conversion, by name.

| Family | GGUF architectures | Notes |
|---|---|---|
| Llama and its relatives | `llama`, `smollm3`, `arcee`, `seed_oss`, `olmo2`, `exaone4`, `mistral3`, `apertus`, `ernie4_5`, `glm4`, `granite` | Llama 2/3/3.x, Mistral, TinyLlama, SmolLM2/3, AFM, Seed-OSS, OLMo 2/3, EXAONE 4, Ministral 3, Apertus, ERNIE 4.5 dense, GLM-4, Granite 3/4 |
| Qwen | `qwen2`, `qwen3`, `qwen3moe`, `qwen2moe`, `qwen2vl`, `qwen3vl`, `qwen3vlmoe` | Qwen2/2.5, Qwen3, Qwen3 MoE, Qwen1.5-MoE, Qwen2-VL and Qwen3-VL text models |
| Qwen hybrids | `qwen3next`, `qwen35`, `qwen35moe` | gated delta-rule linear attention beside attention, Qwen3-Next and Qwen3.5/3.6 |
| Gemma | `gemma`, `gemma2`, `gemma3`, `gemma3n`, `gemma4` | Gemma 1 to 4, Gemma 3n |
| Phi | `phi2`, `phi3`, `phimoe` | Phi-2, Phi-3/3.5/4, Phi-3.5-MoE |
| Classic transformers | `starcoder`, `starcoder2`, `stablelm`, `falcon`, `nemotron`, `command-r`, `cohere2` | LayerNorm, parallel and ungated blocks; Command-R, Aya, Command-R7B, Command-A |
| Mixtures of experts | `olmoe`, `dbrx`, `glm4moe`, `ernie4_5-moe`, `hunyuan-moe`, `hunyuan-dense`, `hunyuan_vl`, `minimax-m2`, `minimax-m3`, `bailingmoe2`, `dots1`, `granitemoe`, `llama4`, `gpt-oss` | Mixtral (as `llama`), OLMoE, DBRX, GLM-4.5/4.6, ERNIE 4.5, Hunyuan, HunyuanOCR's text model (XD-RoPE), MiniMax-M2/M3, Ling 2.0, dots.llm1, Llama 4, GPT-OSS (MXFP4) |
| DeepSeek and Kimi | `deepseek2`, `deepseek32`, `deepseek4`, `kimi-k3` | latent attention (MLA): DeepSeek V2/V3/R1/V3.2, Kimi-K2, Moonlight; DeepSeek V4; Kimi-K3 (Kimi Delta Attention beside MLA, held to its own reference code on the trained Kimi-K3-0.40B) |
| State-space and convolution hybrids | `mamba`, `mamba2`, `jamba`, `falcon-h1`, `granitehybrid`, `nemotron_h`, `nemotron_h_moe`, `lfm2`, `lfm2moe` | Mamba, FalconMamba, Mamba-2, Jamba, Falcon-H1, Granite 4 hybrid, Nemotron-H, Nemotron 3 Nano, LFM2 |
| Embeddings | `bert`, `nomic-bert` | plus Qwen3-Embedding and EmbeddingGemma through their text graphs; `jitllm embed`, `/v1/embeddings` |

Hugging Face safetensors classes convert directly:
`LlamaForCausalLM`, `Qwen2ForCausalLM`, `Qwen3ForCausalLM`,
`Qwen3MoeForCausalLM`, `Qwen2MoeForCausalLM`, `OlmoeForCausalLM`,
`MixtralForCausalLM`, `GemmaForCausalLM`, `Phi3ForCausalLM`,
`PhiMoEForCausalLM`, `DeepseekV2ForCausalLM`, `DeepseekV3ForCausalLM`,
`Glm4MoeLiteForCausalLM` (GLM-4.7-Flash), `Glm4MoeForCausalLM`,
`KimiLinearForCausalLM` (Kimi Linear), `Ernie4_5ForCausalLM`,
`Ernie4_5_MoeForCausalLM`, `HunYuanDenseV1ForCausalLM`,
`HunYuanMoEV1ForCausalLM`, `Dots1ForCausalLM`, `ApertusForCausalLM`,
`SmolLM3ForCausalLM`, `ArceeForCausalLM`, `SeedOssForCausalLM`,
`Olmo2ForCausalLM`, `Olmo3ForCausalLM`, `Exaone4ForCausalLM`,
`Ministral3ForCausalLM`.

Weight formats: F32, F16, BF16, Q4_0, Q5_0, Q5_1, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K
and MXFP4. GGUF quantization is preserved; `-q8` stores a float safetensors
model's matrices as Q8_0.
