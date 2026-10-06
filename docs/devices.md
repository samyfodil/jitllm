# Supported devices

[Back to the README](../README.md)

## Devices

| Target | What runs |
|---|---|
| x86-64 with AVX2, FMA and F16C | 256-bit generated kernels; AVX-VNNI where the CPU has it |
| x86-64 with SSE4.1 and SSSE3 only | 128-bit legacy-encoded kernels (Atom-class CPUs); below that tier a model is refused by name |
| ARM64 (NEON) | generated A64 kernels, including the SDOT dot product |
| NVIDIA (CUDA) | PTX kernels, tensor-core matrix multiply, captured decode graphs |
| Vulkan | SPIR-V kernels (a software Vulkan device is declined) |
| Apple (Metal) | MSL kernels on unified memory |

GPU libraries are loaded at run time through
[`goffi`](https://github.com/go-webgpu/goffi), with no cgo. One card seen by two
backends is one device (matched by its UUID). `-devices auto` picks one device by
measuring; `all` (or a bare `gpu`) uses every distinct device, and a bare `cuda`
or `vulkan` every device of that backend, the fastest filled first; `cuda:1`,
`vulkan:1` or `gpu:1` pins one; `metal` is the system's Metal device; `cpu` uses
none. Releases are built for Linux and macOS on x86-64 and ARM64, and Windows on
x86-64.
