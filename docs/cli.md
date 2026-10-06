# CLI and model conversion

[Back to the README](../README.md)

| Command | What it does |
|---|---|
| `jitllm hardware` | the CPU tier, the GPUs each backend sees, and their memory |
| `jitllm library` | the models `convert` can fetch by name |
| `jitllm convert` | GGUF, a Hugging Face directory, a URL or a catalog name to `.jlm` |
| `jitllm run` | generate, with `-chat`, `-image`, sampling, devices and placement |
| `jitllm embed` | an embedding model's normalised vector |
| `jitllm tokenize` / `info` | the token ids of a text / a GGUF's metadata |
| `jitllm speed` / `batch` | prompt and decode rates, and batched decode |
| `jitllm verify` | compare a device run with the host, position by position |
| `jitllm asm Q4_K` | disassemble a generated CPU kernel |

## Convert a model

```sh
# A model from the catalog (jitllm library lists them).
./jitllm convert -o models llama-3.2-1b-instruct models/llama.jlm

# A GGUF you already have.
./jitllm convert path/to/model.gguf models/model.jlm

# A Hugging Face directory, optionally quantized to Q8_0.
./jitllm convert -q8 path/to/hf-model-dir models/model-q8.jlm

# A Hugging Face repository, read by range without downloading every shard.
./jitllm convert hf://HuggingFaceTB/SmolLM2-360M-Instruct models/smollm2-hf.jlm

# Replace the source's chat template: a tokenizer_config.json, a
# chat_template.json, a .jinja file or a model directory.
./jitllm convert -chat-template path/to/tokenizer_config.json model.gguf models/model.jlm
```

| Flag | Meaning |
|---|---|
| `-o DIR` | where a downloaded model lands |
| `-q8` | safetensors only: store weight matrices as Q8_0 |
| `-chat-template FILE` | store this file's chat template(s) in place of the source's |

The container carries the configuration, tokenizer, stop tokens, chat templates
and any vision tower, in the layout the kernels read, so a page-in is a copy and
not a repack. Set `HF_TOKEN` for gated repositories.

Use the model's own chat template with `-chat`. Generation stops at any of the
model's stop tokens, which the container takes from its source (a GGUF's
end-of-turn ids, a Hugging Face model's `generation_config` list), so a chat
ends at the end of its turn:

```sh
./jitllm run -chat -system "Explain things with simple examples." \
  -temp 0.7 -top-p 0.9 -n 256 models/smollm2.jlm "What is recursion?"
```
