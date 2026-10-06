#!/usr/bin/env bash
# Run the container image the way docs/docker.md tells a user to: convert the
# committed stories260K inside it into a mounted model directory, serve that
# with the image's own entrypoint, and ask the OpenAI surface for a greedy
# completion. A started server or a 200 is not evidence the engine computed
# the model, so the reply's text is checked.
#
#   .github/scripts/docker-smoke.sh IMAGE [WORKDIR]
#
# WORKDIR (default: a fresh directory under the system temp dir) receives the
# converted container and is mounted at /models. The server listens on
# JITLLM_SMOKE_PORT (default 18080) on the loopback interface.
set -euo pipefail
cd "$(dirname "$0")/../.."

image=${1:?usage: docker-smoke.sh IMAGE [WORKDIR]}
work=${2:-$(mktemp -d)}
port=${JITLLM_SMOKE_PORT:-18080}
name=jitllm-smoke-$$

mkdir -p "$work"
# The image runs as an unprivileged user that does not own the directory.
chmod 777 "$work"

cleanup() { docker rm -f "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT

docker run --rm -v "$work:/models" -v "$PWD/testdata/models:/src:ro" \
  --entrypoint jitllm "$image" convert -o /models /src/stories260K.gguf

docker run -d --name "$name" -p "127.0.0.1:$port:8080" -v "$work:/models" \
  "$image" -load stories260K.jlm -devices cpu >/dev/null

for _ in $(seq 60); do
  if curl -sf "http://127.0.0.1:$port/healthz" >/dev/null; then break; fi
  sleep 1
done

curl -sf "http://127.0.0.1:$port/v1/models"
echo
out=$(curl -sf "http://127.0.0.1:$port/v1/completions" -H 'Content-Type: application/json' \
  -d '{"model":"stories260K","prompt":"Once upon a time","max_tokens":16,"temperature":0}') || {
  docker logs "$name"
  exit 1
}
echo "$out"
if ! grep -q 'Lily' <<<"$out"; then
  docker logs "$name"
  echo "::error::the image's completion did not continue the story"
  exit 1
fi
