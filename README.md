<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/jitllm.svg"><img src="docs/assets/jitllm-light.svg" alt="jitllm" width="318"></picture></p>

<p align="center">
<a href="#performance"><img src="docs/assets/readme/perf.png" width="400" alt="Performance: 4.62x llama.cpp's prefill on Qwen3-Next-80B (4x V100); 35 of 35 models decode and 34 of 35 prefill faster than on llama.cpp (V100); up to 4.08x sooner to the first token; up to 5.53x vLLM 0.18.1's batched decode; up to 1.53x llama.cpp's decode on an Apple M4"></a>
<a href="#the-five-principles"><img src="docs/assets/readme/schedule.gif" width="400" alt="A simulation of placement: model blocks page from the .jlm onto a GPU and the CPU, and relocate, KV with them, as a GPU is added, the placement changes and the GPU is removed"></a>
</p>
<p align="center">
<a href="#how-it-works"><img src="docs/assets/readme/jit.gif" width="400" alt="The JIT: the inner loop of a Q4_K matrix-vector product emitted for an AVX2+VNNI CPU, then for CUDA"></a>
<a href="#memory-and-placement"><img src="docs/assets/readme/experts.gif" width="400" alt="Expert pages: each expert of a mixture is its own page, so a token reads only the routed experts that are not resident"></a>
</p>

# jitllm

**An operating system for LLM inference: generate the compute for your hardware,
page models larger than memory, and move execution without losing the conversation.**

- **Every kernel JIT-generated at load time** for your CPU (AVX2, SSE, NEON)
  or GPU (CUDA, Vulkan, Metal), specialised to the model's shapes and weight
  formats. No precompiled kernels or interpreted compute fallback.
- **Disk, RAM and VRAM form the memory system.** Weights, routed MoE experts
  and KV state page as needed. Blocks move between CPU and GPUs at run time,
  carrying their attention and recurrent state with them.
- **Runs everywhere, depends on nothing.** One binary per platform (Linux, macOS,
  Windows; x86 and Arm) that finds CUDA, Vulkan or Metal at run time and
  otherwise runs on the CPU.

Use the engine through an **OpenAI- or Anthropic-compatible server** with streaming
and tool calling, an embeddable Go library, [desktop](#desktop-app) and [terminal](#terminal-app) apps, or Docker.

## Quick start

Build with **Go 1.26 or newer**.

```sh
git clone https://github.com/samyfodil/jitllm.git
cd jitllm
go build ./cmd/jitllm
(cd server && go build -o ../jitllmd ./cmd/jitllmd)

# Fetch a small instruct model from the catalog and convert it (about 386 MB).
mkdir -p models
./jitllm convert -o models smollm2-360m-instruct models/smollm2.jlm

# Ask it something from the command line.
./jitllm run -chat -n 128 models/smollm2.jlm "Why is the sky blue?"

# Or serve it.
./jitllmd serve -addr 127.0.0.1:8080 -models ./models \
  -load smollm2.jlm -id smollm2 -devices auto
```

From another terminal:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"smollm2","messages":[{"role":"user","content":"Why is the sky blue?"}],"max_tokens":128,"stream":true}'
```

Flags go before the model path. `jitllm <command>` with no arguments prints the
usage.

Models run from `.jlm` containers, converted once from GGUF or Hugging Face
weights. See the [CLI and conversion guide](docs/cli.md) for other inputs and flags,
and the [server reference](docs/server.md) for endpoints and daemon commands.

### Docker

```sh
docker build -t jitllm .
docker volume create jitllm-models
docker run --rm -v jitllm-models:/models --entrypoint jitllm jitllm convert smollm2-360m-instruct
docker run -d -p 8080:8080 -v jitllm-models:/models jitllm -load SmolLM2-360M-Instruct-Q8_0.jlm -id smollm2
```

Add `--gpus all` (with the NVIDIA Container Toolkit) to run on an NVIDIA GPU. The image is
built for linux/amd64 and linux/arm64; release images are published as
`ghcr.io/samyfodil/jitllm`. See [docs/docker.md](docs/docker.md).

### Desktop app

Model downloads, conversion, text and image chat, and a live view of where each
block of the model sits, in one window:

```sh
(cd ui && CGO_ENABLED=0 go build -o ../jitllm-desktop .)
./jitllm-desktop
```

<p align="center">
<picture><source media="(prefers-color-scheme: light)" srcset="website/public/shots/chat-light.png"><img src="website/public/shots/chat-dark.png" width="400" alt="The desktop app's Chat screen: a reply with its thinking folded, sampling controls, and the rate and bandwidth of the reply"></picture>
<picture><source media="(prefers-color-scheme: light)" srcset="website/public/shots/machine-light.png"><img src="website/public/shots/machine-dark.png" width="400" alt="The Machine screen: which blocks run on the GPU and which on the CPU, the KV cache, and the pages held in each memory"></picture>
</p>
<p align="center">
<picture><source media="(prefers-color-scheme: light)" srcset="website/public/shots/discover-smartest-light.png"><img src="website/public/shots/discover-smartest-dark.png" width="400" alt="The Discover screen: the catalog ranked for this machine, with an expected rate and whether each model fits"></picture>
<picture><source media="(prefers-color-scheme: light)" srcset="website/public/shots/models-light.png"><img src="website/public/shots/models-dark.png" width="400" alt="The Models screen: the converted models, each with whether it fits on the GPU, fits in memory or pages from disk"></picture>
</p>

### Terminal app

The same screens in a terminal, over SSH too, sharing the desktop app's settings
and chats. The arrow keys move the model's blocks between the CPU and the GPU,
and the conversation keeps its history across the move:

```sh
(cd tui && go build -o ../jitllm-tui .)
./jitllm-tui models/smollm2.jlm
```

<p align="center">
<img src="website/public/shots/tui-chat.png" width="400" alt="The terminal app's Chat screen: Qwen3-30B-A3B answering at 16.3 tokens a second, the engine panel beside it">
<img src="website/public/shots/tui-engine.png" width="400" alt="The terminal app's Engine screen: every block and where it runs, the rate over time, memory and the pager">
</p>

## Supported models

`convert` reads 65 GGUF architecture names (57 graphs), 30 Hugging Face safetensors
classes and 17 vision projectors; anything else is refused at conversion, by name.

- **Text:** Llama 2/3, Mistral, Mixtral, SmolLM2/3, Qwen2 to Qwen3.6 (dense, MoE, Next and VL text),
  Gemma 1 to 4 and 3n, Phi-2/3/4 and Phi-3.5-MoE, DeepSeek V2/V3/R1/V3.2/V4, Kimi-K2, Kimi Linear,
  Kimi-K3, GLM-4/4.5/4.6/4.7-Flash, gpt-oss, Llama 4, Granite, OLMo 2/3, OLMoE, ERNIE 4.5, Hunyuan,
  MiniMax-M2/M3, Ling 2.0, dots.llm1, Apertus, EXAONE 4, Seed-OSS, Command-R/A, DBRX, Falcon,
  StarCoder 1/2, StableLM, Nemotron
- **State-space hybrids:** Mamba, Mamba-2, Jamba, Falcon-H1, Granite 4 hybrid, Nemotron-H, LFM2
- **Vision:** SmolVLM, LLaVA, Qwen2/2.5/3-VL, Qwen3.5, GLM-4.xV, Kimi-VL, HunyuanOCR, Gemma 3/3n/4,
  InternVL, MiniCPM-V, Janus-Pro, Pixtral/Mistral 3, Phi-4 vision, Llama 4
- **Embeddings:** BERT family, nomic-embed-text, Qwen3-Embedding, EmbeddingGemma

Weights: F32, F16, BF16, Q4_0, Q5_0, Q5_1, Q8_0, Q3_K-Q6_K and MXFP4. The full list, generated
from the code, is [docs/models.md](docs/models.md); the [vision guide](docs/vision.md) covers pictures.

## Performance

Results vary by model, hardware and workload. The [scoreboard](docs/perf/current.md)
compares engines measured in the same pass on the same machine, with hardware,
backends, prompt lengths and generation lengths alongside the results.
The [performance overview](docs/performance.md) preserves the comparison tables
and their limitations; [scripts/vs-llamacpp.sh](scripts/vs-llamacpp.sh) measures your machine.

## Documentation

- [CLI and conversion](docs/cli.md) · [Server](docs/server.md) · [Devices](docs/devices.md)
- [Memory and placement](docs/placement.md) · [Vision](docs/vision.md)
- [Go library and desktop app](docs/embedding-go.md) · [Terminal app](docs/tui.md) · [Model API](engine/model/)
- [Project docs](docs/) · [Testing](docs/testing.md) · [Roadmap](ROADMAP.md)
- [Contributing](CONTRIBUTING.md) · [Project rules](AGENTS.md)

<a id="the-five-principles"></a>
<a id="how-it-works"></a>
<a id="memory-and-placement"></a>
The [five principles](docs/runtime.md#the-five-principles) govern every execution
path. See [how it works](docs/runtime.md#how-it-works) for the runtime and source
layout, or [memory and placement](docs/placement.md) for budgets and relocation.

## Embed it in Go

The CLI, server and desktop app use the same engine. This example applies the
model's chat template and generates a greedy response:

```go
package main

import (
    "fmt"
    "log"

    "github.com/samyfodil/jitllm/engine/model"
)

func main() {
    m, err := model.Open("models/smollm2.jlm")
    if err != nil {
        log.Fatal(err)
    }
    defer m.Close()

    ids, err := m.ChatIDs([]model.ChatMessage{
        {Role: "user", Content: "Why is the sky blue?"},
    }, true)
    if err != nil {
        log.Fatal(err)
    }
    state := m.NewState(2048)
    defer state.Close()

    logits, err := state.Prefill(ids)
    for i := 0; i < 128 && err == nil; i++ {
        id := model.Greedy(logits)
        if m.Vocab.IsEOG(id) {
            break
        }
        fmt.Print(m.Vocab.Decode([]int32{id}))
        logits, err = state.Forward(id)
    }
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println()
}
```

See the [Go library guide](docs/embedding-go.md) for budgets, devices, KV caching,
embeddings and batching.
