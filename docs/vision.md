# Vision models

[Back to the README](../README.md)

## Vision

A vision model is one container: the language model and its tower convert
together, and the tower's blocks page and place like any other block, on the CPU
or a GPU. Image preprocessing (decoding, resizing, normalising, gathering
patches) is generated code too, on AVX2, SSE and NEON, held to each family's
reference processor.

| Projector | Families |
|---|---|
| `idefics3` | SmolVLM, Idefics3 |
| `mlp` | LLaVA-style models (CLIP or SigLIP tower and a two-layer MLP) |
| `qwen2vl_merger` | Qwen2-VL |
| `qwen2.5vl_merger` | Qwen2.5-VL |
| `qwen3vl_merger` | Qwen3-VL |
| `glm4v` | GLM-4.1V, GLM-4.5V, GLM-4.6V |
| `kimivl` | Kimi-VL (MoonViT) |
| `hunyuanvl` | HunyuanVL, HunyuanOCR |
| `gemma3` | Gemma 3 |
| `gemma3nv` | Gemma 3n (MobileNet-V5) |
| `gemma4v` | Gemma 4 |
| `internvl` | InternVL 2.5 / 3 (InternViT-300M) |
| `resampler` | MiniCPM-V 2.6, 4.0, 4.5 |
| `janus_pro` | Janus-Pro |
| `pixtral` | Pixtral, Mistral 3 |
| `phi4` | Phi-4 reasoning vision |
| `llama4` | Llama 4 |

```sh
./jitllm convert -o models smolvlm-256m-instruct models/smolvlm.jlm
./jitllm run -image picture.png -n 128 models/smolvlm.jlm "What is in this image?"

# Your own GGUF pair: the model and its mmproj become one container.
./jitllm convert model.gguf mmproj.gguf models/model.jlm
```
