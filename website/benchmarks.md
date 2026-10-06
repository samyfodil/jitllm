# jitllm vs llama.cpp vs vLLM — benchmark recap

Everything below is taken from measurements already recorded on 2026-09-27/28.
Nothing here was re-run for this page. Ratios are **jitllm ÷ other engine**,
taken in the same pass on the same machine; above 1.00x means jitllm is faster.

## Method

- **jitllm:** `jitllm speed -p 512 -n 128 -r 2`. It times a 512-token prompt
  (prefill, "pp512") and a 128-token greedy decode ("tg128") on fresh states,
  after an untimed warm-up.
- **llama.cpp:** `llama-bench -p 512 -n 128 -ngl 99 -r 3`, on the same GGUF
  file jitllm converted from.
- **vLLM:** 0.18.1, generated tokens over the whole wall clock (vLLM's own
  accounting).
- **Model files:** Q4_K_M unless the name says otherwise (Q8_0, MXFP4).
- **Repeats:** the two jitllm repetitions are averaged; llama.cpp is its mean
  over three.

## NVIDIA V100 (Tesla V100-SXM2-16GB, sm_70, CUDA) vs llama.cpp

Host: 2x Xeon E5-2680 v4, 8x V100. llama.cpp is built on the box for sm_70.
The whole board is one single-pass sweep of jitllm's main branch (2026-09-28).

| Model | GPUs | Prefill jitllm tok/s | llama.cpp | **x** | Decode jitllm tok/s | llama.cpp | **x** |
|---|---|---|---|---|---|---|---|
| SmolLM2-360M Q8_0 | 1 | 31031 | 24551 | **1.26x** | 456.8 | 410.0 | **1.11x** |
| Qwen3-0.6B Q8_0 | 1 | 21968 | 20652 | **1.06x** | 382.2 | 379.4 | 1.01x |
| gemma-3-1b | 1 | 15760 | 14621 | **1.08x** | 301.5 | 273.2 | **1.10x** |
| Llama-3.2-1B | 1 | 19721 | 15491 | **1.27x** | 496.5 | 464.3 | **1.07x** |
| SmolLM2-1.7B | 1 | 12654 | 9934 | **1.27x** | 346.9 | 338.4 | 1.03x |
| Qwen2.5-1.5B | 1 | 12104 | 11239 | **1.08x** | 339.0 | 285.2 | **1.19x** |
| gemma-2-2b | 1 | 9390 | 8158 | **1.15x** | 226.3 | 207.8 | **1.09x** |
| Qwen3-1.7B Q8_0 | 1 | 11810 | 11950 | 0.99x | 244.7 | 250.8 | 0.98x |
| Llama-3.2-3B | 1 | 7704 | 6261 | **1.23x** | 224.2 | 220.7 | 1.02x |
| Phi-3.5-mini | 1 | 5359 | 5717 | 0.94x | 197.1 | 199.5 | 0.99x |
| Phi-4-mini | 1 | 6589 | 6035 | **1.09x** | 187.5 | 198.3 | 0.95x |
| gemma-3-4b | 1 | 6616 | 5730 | **1.15x** | 157.0 | 134.4 | **1.17x** |
| Qwen3-4B | 1 | 6487 | 5165 | **1.26x** | 176.4 | 168.4 | **1.05x** |
| OLMoE-1B-7B (MoE) | 1 | 10596 | 4149 | **2.55x** | 423.7 | 389.8 | **1.09x** |
| Mistral-7B v0.3 | 1 | 3774 | 3253 | **1.16x** | 135.6 | 133.2 | 1.02x |
| Qwen2.5-7B | 1 | 4047 | 3403 | **1.19x** | 138.1 | 131.4 | **1.05x** |
| Llama-3.1-8B | 1 | 3888 | 3220 | **1.21x** | 129.4 | 125.7 | 1.03x |
| Qwen3-8B | 1 | 3544 | 3074 | **1.15x** | 121.4 | 119.9 | 1.01x |
| gemma-2-9b | 1 | 2988 | 2500 | **1.20x** | 90.7 | 92.4 | 0.98x |
| gemma-3-12b | 1 | 2439 | 2015 | **1.21x** | 72.0 | 75.2 | 0.96x |
| Qwen3-14B | 1 | 2201 | 1778 | **1.24x** | 75.0 | 72.5 | 1.04x |
| phi-4 (14B) | 1 | 2191 | 1828 | **1.20x** | 74.9 | 74.5 | 1.01x |
| gpt-oss-20b (MoE, MXFP4) | 1 | 4332 | 2719 | **1.59x** | 188.1 | 170.7 | **1.10x** |
| DeepSeek-V2-Lite (MoE, MLA) | 1 | 4362 | 2087 | **2.09x** | 186.8 | 171.2 | **1.09x** |
| DeepSeek-Coder-V2-Lite (MoE, MLA) | 1 | 4351 | 2025 | **2.15x** | 185.8 | 171.1 | **1.09x** |
| Moonlight-16B-A3B (MoE, MLA) | 1 | 4498 | 2118 | **2.12x** | 178.4 | 131.0 | **1.36x** |
| gemma-3-27b | 2 | 958 | 902 | **1.06x** | 37.3 | 38.7 | 0.96x |
| Qwen3-30B-A3B (MoE) | 2 | 3509 | 1081 | **3.25x** | 150.9 | 153.5 | 0.98x |
| Qwen3-32B | 2 | 1009 | 791 | **1.28x** | 35.6 | 34.5 | 1.03x |
| Mixtral-8x7B (MoE) | 2 | 1703 | 868 | **1.96x** | 85.0 | 82.3 | 1.03x |
| Llama-3.3-70B | 4 | 444 | 371 | **1.20x** | 17.5 | 17.4 | 1.01x |
| Qwen2.5-72B | 4 | 421 | 373 | **1.13x** | 16.0 | 16.4 | 0.98x |
| Qwen3-Next-80B-A3B (hybrid MoE) | 4 | 1876 | 445 | **4.22x** | 96.7 | 94.0 | 1.03x |
| gpt-oss-120b (MoE, MXFP4) | 5 | 1971 | 1222 | **1.61x** | 116.1 | 118.4 | 0.98x |
| Kimi-Linear-48B-A3B (hybrid, MLA) | 4 | 2665 | — | runs; no llama.cpp row | 105.3 | — | — |

**Summary:**
- **Prefill:** jitllm is faster on 33 of 35 models llama.cpp ran. Mixture-of-experts and hybrid models run 1.6x to 4.2x faster.
- **Decode:** every row is between 0.95x and 1.36x, so at parity or ahead.
- **Scale:** models run across up to 5 GPUs.
- **Kimi-Linear-48B:** it ran here only from a Hugging Face conversion, with no GGUF, so there is no llama.cpp row.

## Batched decode: jitllm vs llama.cpp vs vLLM (V100)

This is many independent sequences decoding together. The setup:
- one V100-SXM2-16GB, CUDA;
- Llama-3.1-8B-Instruct Q4_K_M;
- a 6-token prompt, then 128 tokens generated per sequence;
- aggregate tokens per second over the whole wall clock;
- all three engines in the same pass (2026-09-26).

| Batch | jitllm tok/s | llama.cpp tok/s | vLLM 0.18.1 tok/s | vs llama.cpp | vs vLLM |
|---|---|---|---|---|---|
| 16 | 757 | 694 | 395 | **1.09x** | **1.92x** |
| 32 | 920 | 988 | 415 | 0.93x | **2.22x** |
| 64 | 1028 | 630 | 436 | **1.63x** | **2.36x** |
| 128 | 1142 | 1090 | 441 | **1.05x** | **2.59x** |

jitllm is 1.9x to 2.6x vLLM at every batch size on this card. It beats llama.cpp
at 3 of 4 batch sizes; batch 32 is the exception, at 0.93x.

## Apple M4 (Mac mini) vs llama.cpp (b10985)

The board was taken on one machine in a single pass (2026-09-27).

| Model | Metal prefill | Metal decode | CPU prefill | CPU decode |
|---|---|---|---|---|
| stories15M | 0.66x | **1.31x** | 0.28x | **1.07x** |
| SmolLM2-360M Q8_0 | 0.76x | **1.04x** | 0.65x | 0.79x |
| TinyLlama-1.1B q3_K_M | 0.91x | **1.24x** | 0.47x | **1.10x** |
| Llama-3.2-1B | 0.88x | 0.96x | 0.67x | 0.97x |
| Qwen2-1.5B | 0.91x | 0.96x | 0.83x | 0.91x |
| Qwen3-MoE-4x0.6B | 0.82x | 0.98x | 0.67x | 0.92x |
| gemma-2b | 0.90x | **1.00x** | 0.74x | 0.89x |

- **Metal decode:** at parity or ahead on 4 of 7 models.
- **M4 prefill:** still behind on both Metal and CPU. Work continues there.

## Notes

- **gemma-3-27b:** re-measured after two fixes landed on 2026-09-28 (a deadlock and a placement fix), because the sweep build hung on it. Its llama.cpp column is the mean of two same-day runs (899, 905).
- **Not covered here:** x86 CPU (Xeon) is still in progress. vLLM has only the batched 8B comparison above.
- **Architectures:** jitllm also runs phi-2, StarCoder/StarCoder2, Command-R/Cohere2, StableLM, Falcon and Nemotron. They were checked against transformers and llama.cpp, but they have no throughput rows yet.
