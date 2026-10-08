---
title: Install
description: Install jitllm, jitllmd and the two apps from a release on Linux, macOS or Windows, or build them from source.
---

Every program is one self-contained binary. There is nothing else to install: no Python, no C library, no CUDA toolkit. GPU backends are loaded at run time from the driver already on the machine.

## Install a release

On Linux and macOS:

```sh
curl -fsSL https://jitllm.org/install.sh | sh
```

On Windows, in PowerShell:

```powershell
irm https://jitllm.org/install.ps1 | iex
```

The script finds the latest release, downloads the archives for this machine, checks each against the release's `checksums.txt`, and installs the binaries:

| | Linux and macOS | Windows |
|---|---|---|
| where | `~/.local/bin`, or `/usr/local/bin` as root | `%LOCALAPPDATA%\Programs\jitllm`, added to your user `PATH` |
| which programs | arguments: `sh -s -- all` | `$env:JITLLM_PROGRAMS = "all"` before the line |
| a version | `JITLLM_VERSION=v0.1.0` | `$env:JITLLM_VERSION = "v0.1.0"` |
| another directory | `JITLLM_INSTALL_DIR=...` | `$env:JITLLM_INSTALL_DIR = "..."` |

The programs are `jitllm` (the CLI) and `jitllmd` (the server), installed by default, and `desktop` (the [desktop app](/docs/guides/desktop/)) and `tui` (the [terminal app](/docs/guides/terminal/)); `all` is the four. For example, the CLI and the terminal app:

```sh
curl -fsSL https://jitllm.org/install.sh | sh -s -- jitllm tui
```

Run it again to update. To install by hand, the [latest release](https://github.com/samyfodil/jitllm/releases/latest) has one archive per program and platform, named `<program>_<version>_<os>_<arch>`: `jitllm`, `jitllmd`, `jitllm-desktop` and `jitllm-tui`, for `linux`, `darwin` and `windows` on `amd64` and `arm64` (`.zip` on Windows, `.tar.gz` elsewhere), with `checksums.txt` beside them. Unpack one and put the binary on your `PATH`. The server also ships as a container image; see [Docker](https://github.com/samyfodil/jitllm/blob/main/docs/docker.md).

## GPUs

A GPU needs only its driver:

| Backend | What jitllm loads | Where it comes from |
|---|---|---|
| CUDA | `libcuda.so.1` (Linux, WSL), `nvcuda.dll` (Windows) | the NVIDIA driver |
| Vulkan | `libvulkan.so.1`, `vulkan-1.dll`, or `libvulkan.1.dylib` / MoltenVK | the GPU driver or Vulkan loader |
| Metal | the system Metal framework | macOS on Apple silicon |

jitllm generates its own PTX, SPIR-V and Metal shaders, so the CUDA toolkit, `nvcc` and a shader compiler are never needed. With no usable GPU, automatic mode runs on the CPU.

## Build from source

With **Go 1.26 or newer** and **Git**, and no C toolchain (there is no cgo):

```sh
git clone https://github.com/samyfodil/jitllm.git
cd jitllm

# The engine and its command line.
go install ./cmd/jitllm

# The API server. It is a separate module, so the core keeps one dependency.
(cd server && go install ./cmd/jitllmd)

# The desktop app and the terminal app, modules of their own too.
(cd ui && CGO_ENABLED=0 go build -o "$(go env GOPATH)/bin/jitllm-desktop" .)
(cd tui && go build -o "$(go env GOPATH)/bin/jitllm-tui" .)
```

These put the binaries in Go's bin directory, `$(go env GOPATH)/bin` unless `GOBIN` says otherwise. Make sure it is on your `PATH`:

```sh
export PATH="$PATH:$(go env GOPATH)/bin"
```

On Windows the same commands produce `.exe` files; in PowerShell run the second as `cd server; go install .\cmd\jitllmd; cd ..`.

The core module has a single dependency, [`goffi`](https://github.com/go-webgpu/goffi), which is how it calls the GPU drivers without cgo. Builds are static and cross-compile like any Go program:

```sh
GOOS=windows GOARCH=arm64 go build ./cmd/jitllm
GOOS=darwin  GOARCH=arm64 go build ./cmd/jitllm
```

## Check what jitllm sees

```sh
jitllm hardware
```

`hardware` prints the CPU's instruction-set tier and the kernels it will generate, every GPU each backend can open, and how much memory is actually spendable on each. One card that answers to two backends (CUDA and Vulkan, say) is listed under both and counted once. Use the device names it prints with [`-devices`](/docs/guides/devices/).

## The apps

The [desktop app](/docs/guides/desktop/) and the [terminal app](/docs/guides/terminal/) run the engine in their own process and share their settings and saved chats. Install them with the script's `desktop` and `tui`, or build them as above.

## Next

- [Quickstart](/docs/get-started/): fetch a model and serve it.
- [Hardware support](/docs/reference/hardware/): what each CPU tier and GPU backend runs.
