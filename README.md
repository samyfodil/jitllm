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

**A Go LLM inference engine that generates kernels for your hardware and pages models through CPU and GPU memory.**

- **JIT kernels:** generated at run time for the CPU or GPU in front of it.
- **Models larger than memory:** weights page from disk through RAM and VRAM;
  mixtures of experts read only the experts each token routes to.
- **Move blocks while serving:** blocks, attention state and recurrent state
  move between the CPU and GPUs without losing the conversation.
- **No cgo or GPU SDK:** one Go executable per platform, using installed
  CUDA, Vulkan or Metal drivers at run time, or running on the CPU.
- **Compatible server:** OpenAI and Anthropic APIs, with streaming and tool calling.

## Quick start

Build with **Go 1.26 or newer**. GPU backends use the drivers already installed;
nothing GPU-specific is needed at build time.

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

## Supported models

Llama, Qwen, Gemma, Phi, DeepSeek, Kimi, GPT-OSS and other dense, mixture-of-experts
and state-space families; vision and embedding models are also supported.
See the [full model list](docs/models.md), [model and weight format reference](docs/supported-model-reference.md)
and [vision guide](docs/vision.md). Architectures outside the converter's list are refused.

## Performance

Results vary by model, hardware and workload. The [scoreboard](docs/perf/current.md)
compares engines measured in the same pass on the same machine, with hardware,
backends, prompt lengths and generation lengths alongside the results.
The [performance overview](docs/performance.md) preserves the comparison tables
and their limitations; [scripts/vs-llamacpp.sh](scripts/vs-llamacpp.sh) measures your machine.

## Documentation

- [CLI and conversion](docs/cli.md) · [Server](docs/server.md) · [Devices](docs/devices.md)
- [Memory and placement](docs/placement.md) · [Vision](docs/vision.md)
- [Go library and desktop app](docs/embedding-go.md) · [Model API](engine/model/)
- [Project docs](docs/) · [Testing](docs/testing.md) · [Roadmap](ROADMAP.md)
- [Contributing](CONTRIBUTING.md) · [Project rules](AGENTS.md)

<a id="the-five-principles"></a>
<a id="how-it-works"></a>
<a id="memory-and-placement"></a>
Kernels, paging, relocation, generated compute and low allocation are the
[five principles](docs/runtime.md#the-five-principles). Read
[how it works](docs/runtime.md#how-it-works) for the runtime and source layout,
or [memory and placement](docs/placement.md) for budgets and moving blocks.

Try a model and [share your results](https://github.com/samyfodil/jitllm/issues)
with `jitllm hardware`, the model and quantization, the command and what you saw.

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
embeddings and batching. The core module's one dependency is `goffi`.

## License

Apache License 2.0; see [LICENSE](LICENSE). Copyright Bunshin Labs LLC
([bunshinlabs.com](https://bunshinlabs.com)); see [COPYRIGHT](COPYRIGHT).
