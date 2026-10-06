# syntax=docker/dockerfile:1
#
# jitllm in a container: jitllmd serving on :8080, with the jitllm CLI beside
# it. docs/docker.md has the run commands, CPU and NVIDIA.
#
#   docker buildx build --platform linux/amd64,linux/arm64 -t jitllm .
#
# Both binaries are pure Go (CGO_ENABLED=0) and are cross-compiled on the
# build host for every platform, so no emulation runs the compiler. They are
# not static: goffi links them against glibc's dynamic loader, which is how
# CUDA and Vulkan are opened at run time (dlopen of libcuda.so.1 and
# libvulkan.so.1). So the runtime base needs glibc and cannot be
# distroless/static, and the GPU libraries arrive when the container starts,
# never at build time.
#
# BIN picks where the binaries come from: `source` (the default) compiles this
# checkout; `dist` takes the release binaries goreleaser places in the build
# context at <os>/<arch>/ (dockers_v2 in .goreleaser.yaml), so a published
# image carries the very binaries the release archives carry. The stage not
# chosen is never built.

ARG GO_VERSION=1.27
ARG DEBIAN=trixie
ARG BIN=source

FROM --platform=$BUILDPLATFORM golang:${GO_VERSION}-${DEBIAN} AS build
WORKDIR /src
# The module files alone first, so a source edit does not re-download.
COPY go.mod go.sum ./
COPY server/go.mod server/go.sum server/
RUN --mount=type=cache,target=/go/pkg/mod \
    go mod download && go -C server mod download
COPY . .
ARG TARGETOS
ARG TARGETARCH
ARG VERSION=dev
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    export CGO_ENABLED=0 GOOS=$TARGETOS GOARCH=$TARGETARCH && \
    flags="-s -w -X main.version=$VERSION" && \
    go build -trimpath -ldflags "$flags" -o /out/jitllm ./cmd/jitllm && \
    go -C server build -trimpath -ldflags "$flags" -o /out/jitllmd ./cmd/jitllmd

FROM scratch AS bin-source
COPY --from=build /out/ /

FROM scratch AS bin-dist
ARG TARGETPLATFORM
COPY $TARGETPLATFORM/jitllm $TARGETPLATFORM/jitllmd /

FROM bin-${BIN} AS bin

FROM debian:${DEBIAN}-slim
# ca-certificates: `jitllm convert NAME|URL` downloads from the Hugging Face
# Hub. libvulkan1: the Vulkan loader. It finds no device by itself: the ICD
# comes with the GPU (NVIDIA's from the Container Toolkit under the `graphics`
# capability; Mesa's for AMD and Intel is not shipped, see docs/docker.md).
# CUDA needs nothing here: the toolkit mounts libcuda.so.1 and the PTX JIT
# into the library path and refreshes the loader cache.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates libvulkan1 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --uid 10001 --user-group --create-home --home-dir /var/lib/jitllm jitllm \
 && mkdir /models && chown jitllm:jitllm /models
COPY --from=bin /jitllm /jitllmd /usr/local/bin/

# JITLLM_MODELS is where `jitllm convert` writes and `jitllmd serve -models`
# scans. The NVIDIA variables are read only by the NVIDIA Container Toolkit:
# compute brings libcuda and the PTX compiler, utility nvidia-smi, graphics
# the Vulkan ICD.
ENV JITLLM_MODELS=/models \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
VOLUME /models
WORKDIR /models
# Numeric, so a runtime that must prove the user is not root (Kubernetes'
# runAsNonRoot) can.
USER 10001:10001
EXPOSE 8080

# The daemon's own client asks the control plane for its loaded models: a
# real round trip through the server, with no curl in the image.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["jitllmd", "models", "-loaded", "-addr", "localhost:8080"]

# Flags given to `docker run` are appended: `-load model.jlm -devices cuda`.
# The CLI is `docker run --entrypoint jitllm ...`.
ENTRYPOINT ["jitllmd", "serve", "-addr", ":8080"]
