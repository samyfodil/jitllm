---
title: Install
description: Install jitllm and jitllmd from source on Linux, macOS or Windows.
---

jitllm is built from source with the Go toolchain. There is nothing else to install: no Python, no C compiler, no CUDA toolkit. GPU backends are loaded at runtime from the driver that is already on the machine.

## Requirements

- **Go 1.26 or newer** and **Git**.
- **Linux, macOS or Windows**, on **x86-64 or ARM64**.
- For a GPU, only its driver:

| Backend | What jitllm loads | Where it comes from |
|---|---|---|
| CUDA | `libcuda.so.1` (Linux, WSL), `nvcuda.dll` (Windows) | the NVIDIA driver |
| Vulkan | `libvulkan.so.1`, `vulkan-1.dll`, or `libvulkan.1.dylib` / MoltenVK | the GPU driver or Vulkan loader |
| Metal | the system Metal framework | macOS on Apple silicon |

jitllm generates its own PTX, SPIR-V and Metal shaders, so the CUDA toolkit, `nvcc` and a shader compiler are never needed. With no usable GPU, automatic mode runs on the CPU.

## Build

```sh
git clone https://github.com/samyfodil/jitllm.git
cd jitllm

# The engine and its command line.
go install ./cmd/jitllm

# The API server. It is a separate module, so the core keeps one dependency.
(cd server && go install ./cmd/jitllmd)
```

`go install` puts `jitllm` and `jitllmd` in Go's bin directory, `$(go env GOPATH)/bin` unless `GOBIN` says otherwise. Make sure it is on your `PATH`:

```sh
export PATH="$PATH:$(go env GOPATH)/bin"
```

On Windows the same commands produce `jitllm.exe` and `jitllmd.exe`; in PowerShell run the second as `cd server; go install .\cmd\jitllmd; cd ..`.

The core module has a single dependency, [`goffi`](https://github.com/go-webgpu/goffi), which is how it calls the GPU drivers without cgo. Builds are static and cross-compile like any Go program:

```sh
GOOS=windows GOARCH=amd64 go build ./cmd/jitllm
GOOS=darwin  GOARCH=arm64 go build ./cmd/jitllm
```

## Check what jitllm sees

```sh
jitllm hardware
```

`hardware` prints the CPU's instruction-set tier and the kernels it will generate, every GPU each backend can open, and how much memory is actually spendable on each. One card that answers to two backends (CUDA and Vulkan, say) is listed under both and counted once. Use the device names it prints with [`-devices`](/docs/guides/devices/).

## The desktop app

The desktop app is a third module. It runs the engine in process and brings model downloads, conversion, text and image chat, sampling controls and memory and device views into one window. See [The desktop app](/docs/guides/desktop/) for what each screen does.

```sh
(cd ui && CGO_ENABLED=0 go build -o "$(go env GOPATH)/bin/jitllm-desktop" .)
jitllm-desktop
```

## Next

- [Quickstart](/docs/get-started/): fetch a model and serve it.
- [Hardware support](/docs/reference/hardware/): what each CPU tier and GPU backend runs.
