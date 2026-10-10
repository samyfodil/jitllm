# Server reference

[Back to the README](../README.md)

## The server

`jitllmd serve` serves three surfaces on one address.

| Surface | Endpoints |
|---|---|
| OpenAI-compatible | `POST /v1/chat/completions`, `POST /v1/completions`, `GET /v1/models`, `POST /v1/embeddings` (floats or base64), `/v1/files`, `/v1/batches` |
| Anthropic-compatible | `POST /v1/messages` (named SSE events) |
| Connect RPC | `ModelService`, `InferenceService` (`Generate`, `Complete`, `Cancel`, `Embed`), `SessionService`, `PlacementService`, `DeviceService`, `TelemetryService` |
| Health | `GET /healthz` |

Both compatible APIs stream and take `tools`: the tool list is rendered by the
model's own chat template, and calls come back as OpenAI `tool_calls` or
Anthropic `tool_use` blocks. `ignore_eos` on the OpenAI endpoints and the
Connect `GenerateRequest` (`jitllmd run -ignore-eos`) keeps generating past the
stop tokens until `max_tokens` or the end of the session's context. A request
with no `max_tokens` runs until the model ends its reply or fills the context;
either way a reply that reaches the end of the context finishes as `length`. Generates of one model placed wholly on one device, or wholly on the host,
decode together as rows of one step (`-max-batch`). A model's queue holds
`-max-queue` requests (64 by default) before it answers 429 with a
`Retry-After`. Under a model's KV budget (`Engine.SetKVBudget`, an embedding
API with no flag yet) a generate parks idle sessions of the model, lowest
priority then least recently stepped, into its memory cache, and a parked
session resumes byte for byte at its next generate.

`logprobs` is OpenAI's on both endpoints: on chat `logprobs: true` with
`top_logprobs` 0..20 (`choices[].logprobs.content`), on `/v1/completions` an
alternative count 0..20 (`tokens`, `token_logprobs`, `top_logprobs`,
`text_offset`), streamed or not. The values are the model's raw distribution
-- the log-softmax of the logits, before temperature, the repeat penalty and
the top-k/top-p/min-p cuts -- which is what vLLM returns by default
(`logprobs_mode=raw_logprobs`); they do not move with the request's
temperature. The log-softmax and the top-N selection are generated kernels on
the host, where the logits of every sampled token are. `logprobs` with `echo`
(the prompt's own log-probabilities) is refused.

`n` (1..128) asks for several continuations of one prompt. The prompt is
prefilled once: the first choice seals its pages and final logits into a
request-scoped prompt store and every other choice restores them, so usage
counts the prompt once (`jitllm.prompt_restored` lists the positions each
choice restored). A seeded request's choice `i` uses `seed + i` and equals the
same request sent alone with that seed; an unseeded one draws distinct seeds
(`jitllm.seeds`). The choices decode one after another; streamed, each
choice's chunks carry its `index`, and the usage comes in a last chunk of its
own.

The Files and Batch APIs are OpenAI's: `POST /v1/files` (multipart, `purpose`
`batch`), `GET /v1/files[/{id}[/content]]`, `DELETE /v1/files/{id}`, and
`POST /v1/batches`, `GET /v1/batches[/{id}]`, `POST /v1/batches/{id}/cancel`.
A batch validates every line first (`custom_id`, `POST`, the batch's
`endpoint`: `/v1/chat/completions`, `/v1/completions` or `/v1/embeddings`),
then runs each line in the background through the same handler the live
route uses, one after another, and writes the output and error files as jsonl.
Files and batch records are kept under `.jitllm-batch` in the model directory
(`Config.BatchDir`), so they survive a restart; a batch the restart
interrupted is reported failed.

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

With each other: `logprobs` on a constrained reply are the raw distribution's,
before the grammar's mask, as vLLM's default does; `n` with a grammar
constrains every choice; speculation with `logprobs` or with `n` is refused
by name (a round decides several tokens from one verification and reads back
no per-token distribution, and the choices of `n` prefill through a shared
prompt store a Speculator does not use).

| `jitllmd` verb | What it does |
|---|---|
| `serve` | run the daemon (`-addr`, `-models`, `-load`, `-id`, `-devices`, `-maxmem`, `-max-batch`, `-max-queue`, `-kv-cache`; `jitllmd serve -h` lists every flag) |
| `run` | stream a generation from a running daemon |
| `models` | list, describe, load and unload models |
| `devices` | the hardware and the memory that is actually spendable |
| `sessions` | create, list, describe, reset and close sessions |
| `place` | read and move the CPU/device seam and the pager budget |
| `stats` | engine counters, once or streamed (`-watch`) |

The Connect API is defined in [`proto/jitllm/v1`](../proto/jitllm/v1/).
