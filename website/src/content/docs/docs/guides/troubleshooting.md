---
title: Troubleshooting
description: Common errors and what to do about them.
---

## "is not a .jlm container" or a convert command is printed

The engine runs `.jlm` files only. Convert the GGUF or Hugging Face model first; the error prints the exact command:

```sh
jitllm convert model.gguf model.jlm
```

## A container is refused after updating jitllm

The container's layout version is older than this build reads. Convert it again from its source. Containers are caches of their source, so there is no migration step.

## My flags were read as the prompt

Flags must come before the model path. Everything after the path is prompt text. jitllm refuses a known flag in that position rather than silently generating from it, so move the flag forward:

```sh
jitllm run -chat -n 64 models/m.jlm "Hello"     # right
jitllm run models/m.jlm -chat "Hello"           # refused
```

## No GPU was used

Run `jitllm hardware`. If no device is listed, jitllm could not load the driver library: `libcuda.so.1` or `nvcuda.dll` for CUDA, the Vulkan loader for Vulkan. Install or repair the GPU driver; no toolkit is needed. With `-devices auto` a machine with no usable GPU runs on the CPU, and the run's summary line says so. Name a device (`-devices cuda:0`) to get an error instead.

If the device is listed but few blocks were placed, the card is short of memory: other processes hold it, or the model is large. The summary line prints how many blocks went where. See [Devices and placement](/docs/guides/devices/#when-the-model-does-not-fit-on-the-card).

## A long conversation runs out of memory

The KV cache grows a page at a time, and by default all of it stays in memory. Give the run a directory and a cap, and older history spills to disk and comes back when attention reads it:

```sh
jitllm run -kv-cache ~/.cache/jitllm-kv -kv-budget 2G models/m.jlm "..."
```

On a GPU, a context that outgrows the card can also hand blocks back to the CPU (`-relocate`, on by default when the host is allowed). See [Memory and paging](/docs/guides/memory/#long-context).

## A placement entry was ignored

`-placement` is best effort: a block that cannot go where it is named, because the device is full, is placed where the engine would have put it, and the run reports the decline. Add `-placement-strict` to get an error instead. A device the map names must also appear in `-devices`; naming one that does not is always an error.

## The first answer is slow

The first prompt of a process generates and loads the prompt kernels, and the tuners measure this machine. Later prompts are faster. On a server this happens once per model; use `jitllm speed`, which warms up first, for timings.

## A model runs much slower than expected

- Check that it fits. If `-maxmem` or the machine leaves it a page short, every token reads from disk. The summary line reports page-ins; a larger budget, or a smaller quantization, fixes it.
- Close other heavy work. Decode uses every performance core and waits for the slowest, so one busy core slows the whole token.
- On a laptop, heat throttling lowers every rate. Compare runs taken in the same conditions.

## CPU and GPU answers differ

The CPU and GPU add floating-point numbers in different orders, so logits differ slightly, and greedy decoding can take a different branch at a near-tie. `jitllm verify -devices cuda:0 models/m.jlm` reports every position where they differ and by how much. Small differences at near-ties are expected; a large one is a bug worth [reporting](https://github.com/jitllm/jitllm/issues).

## Reporting a problem

Include the output of `jitllm hardware`, the model and its quantization, the exact command, and what you saw. That is almost always enough to reproduce it.
