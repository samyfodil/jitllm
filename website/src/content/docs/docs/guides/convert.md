---
title: Convert models
description: Turn a GGUF, a Hugging Face model or a catalog name into a .jlm container.
---

The engine runs one weight format, [`.jlm`](/docs/reference/jlm/). GGUF and Hugging Face safetensors are what you convert *from*. Conversion lays the weights out once, in the form every backend reads, so paging a block in during inference is a plain read with no repacking.

Give `jitllm model.gguf` to the engine directly and it refuses, printing the `convert` command that fixes it.

## From the catalog

```sh
jitllm library                       # every name, with its download size
jitllm convert -o models qwen3-8b models/qwen3-8b.jlm
```

A catalog name is pinned to a Hugging Face revision that was checked against this engine. For a vision model the entry names both the language model and its vision tower, and both go into one container. See the [catalog](/docs/reference/models/#the-catalog).

## From a GGUF

```sh
jitllm convert path/to/model.gguf models/model.jlm
```

The GGUF's quantization is kept as it is, K-quants and MXFP4 included; conversion changes the storage layout, not the numbers. The formats it reads are F32, F16, BF16, Q4_0, Q5_0, Q5_1, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K and MXFP4.

A vision model in GGUF comes as two files, the model and an `mmproj`. Pass both and they become one container:

```sh
jitllm convert model.gguf mmproj.gguf models/vlm.jlm
```

## From Hugging Face safetensors

A safetensors model is a directory: `config.json` and the tokenizer files sit beside the weights, and sharded models come through `model.safetensors.index.json`.

```sh
jitllm convert path/to/hf-model-dir models/model.jlm

# Quantize every weight matrix to Q8_0 on the way in: half the bytes of F16.
jitllm convert -q8 path/to/hf-model-dir models/model-q8.jlm
```

`-q8` applies to safetensors only; a GGUF keeps the quantization it was published with.

## From a URL

`convert` downloads a Hugging Face reference first. A download resumes if interrupted, is checked against the Hub's SHA-256, and is skipped when the file is already present and complete.

| Form | Meaning |
|---|---|
| `hf://OWNER/REPO` | a safetensors repository, or the repository's only GGUF |
| `hf://OWNER/REPO/FILE.gguf` | one GGUF in a repository |
| `hf://OWNER/REPO/FILE.gguf@REV` | pinned to a branch, tag or commit |
| `https://huggingface.co/OWNER/REPO/resolve/REV/FILE.gguf` | the download link |
| `https://huggingface.co/OWNER/REPO/blob/REV/FILE.gguf` | the address-bar link |

```sh
jitllm convert hf://HuggingFaceTB/SmolLM2-360M-Instruct models/smollm2-hf.jlm
```

A safetensors repository converts where it lives: tensor ranges are streamed from the Hub, so you do not download every shard first.

`-o DIR` is where a downloaded GGUF lands; without it, `JITLLM_MODELS` sets the default. For a private or gated repository set `HF_TOKEN` (`HUGGING_FACE_HUB_TOKEN` and `~/.cache/huggingface/token` are read too).

## What goes into a container

- The weights, in a layout shared by the CPU, CUDA, Vulkan and Metal.
- The model's configuration, as a typed record rather than loose keys.
- The tokenizer: its vocabulary, merges and pre-tokenizer stages.
- The chat templates, which `-chat` and the server apply.
- The vision tower, when the model has one.

Everything inference needs is in the one file; the source can be deleted.

## When to convert again

A container carries a version number. When a new jitllm changes the layout, it refuses an old container by name instead of misreading it; convert again from the source, which takes seconds to minutes. There is deliberately no backward compatibility: a container is a cache of its source, not an archive.

## Tokenizer overrides

A GGUF names its pre-tokenizer with a label. If you have the model's own `tokenizer.json` and want its pre-tokenizer instead, pass it at run time:

```sh
jitllm run -tokenizer path/to/tokenizer.json models/model.jlm "Hello"
```

The GGUF still supplies the vocabulary; only the splitting rules are replaced.
