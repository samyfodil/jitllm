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
stop tokens until `max_tokens` or the end of the session's context. A request
with no `max_tokens` runs until the model ends its reply or fills the context;
either way a reply that reaches the end of the context finishes as `length`. Generates of one model placed wholly on a device
decode together as rows of one step (`-max-batch`).

Structured output: OpenAI's `response_format` (`json_object`, or
`json_schema` with its `schema`) on `/v1/chat/completions` and
`/v1/completions`, and `grammar` (GBNF) or `json_schema` on the Connect
`GenerateRequest`. Every sampled token keeps the reply inside the grammar and
the reply ends where the grammar does. A schema keyword the converter does not
build (`pattern`, `format`, numeric bounds, `allOf`, ...) is a 400 naming it;
the supported subset is in
[model-correctness.md](engineering-history/model-correctness.md),
"engine/grammar". A constrained generate decodes alone and without
speculation, and refuses `ignore_eos`.

Speculative decoding is off by default and asked for per request
(`speculation` on `GenerateRequest`, `jitllm_speculate` on the OpenAI
endpoints) or per session (`speculation` on `CreateSessionRequest`). It
drafts with the model's own prediction block where it carries one and by
prompt lookup otherwise, and verifies the drafts in one pass. It is for
greedy decoding: greedy output is plain greedy's token for token, while a
sampled request with prompt lookup drafts nothing. A speculative generate
prefills alone (not through the step loop or the session's prompt store), a
`continue_session` generate decodes plainly, and a session whose speculative
reply was cut by a stop string inside a round refuses `continue_session`.

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
