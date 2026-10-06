# Server reference

[Back to the README](../README.md)

## The server

`jitllmd serve` serves three surfaces on one address.

| Surface | Endpoints |
|---|---|
| OpenAI-compatible | `POST /v1/chat/completions`, `POST /v1/completions`, `GET /v1/models`, `POST /v1/embeddings` (floats or base64) |
| Anthropic-compatible | `POST /v1/messages` (named SSE events) |
| Connect RPC | `ModelService`, `InferenceService` (`Generate`, `Complete`, `Cancel`, `Embed`), `SessionService`, `PlacementService`, `DeviceService`, `TelemetryService` |
| Health | `GET /healthz` |

Both compatible APIs stream and take `tools`: the tool list is rendered by the
model's own chat template, and calls come back as OpenAI `tool_calls` or
Anthropic `tool_use` blocks. `ignore_eos` on the OpenAI endpoints and the
Connect `GenerateRequest` (`jitllmd run -ignore-eos`) keeps generating past the
stop tokens until `max_tokens`. Generates of one model placed wholly on a device
decode together as rows of one step (`-max-batch`).

| `jitllmd` verb | What it does |
|---|---|
| `serve` | run the daemon (`-addr`, `-models`, `-load`, `-id`, `-devices`, `-maxmem`, `-max-batch`) |
| `run` | stream a generation from a running daemon |
| `models` | list, describe, load and unload models |
| `devices` | the hardware and the memory that is actually spendable |
| `sessions` | create, list, describe, reset and close sessions |
| `place` | read and move the CPU/device seam and the pager budget |
| `stats` | engine counters, once or streamed (`-watch`) |

The Connect API is defined in [`proto/jitllm/v1`](../proto/jitllm/v1/).
