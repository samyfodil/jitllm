---
title: Quickstart
description: Fetch a 30B mixture of experts, serve it over the OpenAI API and send it a request.
---

This takes an 18.6 GB download. It assumes you have [installed `jitllm` and `jitllmd`](/docs/install/) and they are on your `PATH`.

## 1. Fetch and convert a model

`convert` takes a name from the built-in catalog, downloads the GGUF from Hugging Face, checks it against the Hub's SHA-256 and writes a `.jlm` container, the one format the engine runs.

```sh
mkdir -p models
jitllm convert -o models qwen3-30b-a3b models/qwen3.jlm
```

Qwen3-30B-A3B is a mixture of experts: 30B weights, of which each token uses 3B, so it decodes like a much smaller model. It wants about 20 GB of memory; with less, the engine pages its experts from disk and it still runs, at a speed set by the drive. A smaller model is a name away: `qwen3-8b`, or `smollm2-360m-instruct` for a 386 MB first try.

`jitllm library` lists every name it accepts. You can also convert a GGUF or a Hugging Face model you already have; see [Convert models](/docs/guides/convert/).

## 2. Try it from the command line

```sh
jitllm run -chat models/qwen3.jlm "Why is the sky blue?"
```

`-chat` applies the model's own chat template. Qwen3 thinks before it answers, and the reply runs until the model ends it; `-n N` caps it at N tokens. Flags go before the model path; everything after it is the prompt.

## 3. Serve it

```sh
jitllmd serve -addr 127.0.0.1:8080 -models ./models \
  -load qwen3.jlm -id qwen3 -devices auto
```

`-devices auto` measures each GPU backend and picks one; a machine with no GPU runs on the CPU. Leave `-devices` out and the server runs on the CPU only. The server is ready when it prints `jitllmd on 127.0.0.1:8080` and the three endpoint addresses.

## 4. Send it a request

From another terminal:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3","messages":[{"role":"user","content":"Why is the sky blue?"}],"stream":true}'
```

Any OpenAI client works with the base URL `http://127.0.0.1:8080/v1`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
for chunk in client.chat.completions.create(
    model="qwen3",
    messages=[{"role": "user", "content": "Why is the sky blue?"}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="")
```

The Anthropic messages API is at `POST /v1/messages` on the same address.

## 5. Look inside

`jitllmd` is also a client of a running server:

```sh
jitllmd models          # what is loaded
jitllmd devices         # the hardware, and the memory that is spendable
jitllmd stats -watch    # engine counters, live
```

## Where next

- [Serve an API](/docs/guides/serve/) covers streaming, tool calls, sessions and the Connect API.
- [Devices and placement](/docs/guides/devices/) shows how to pick GPUs and budget their memory.
- [Memory and paging](/docs/guides/memory/) explains how a model larger than RAM runs.
