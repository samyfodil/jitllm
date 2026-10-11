# Connect an OpenAI-compatible client

Point the official `openai` SDKs, or plain curl, at `jitllmd`. Nothing in the
client changes but its base URL.

**Prerequisites:** `jitllm` and `jitllmd` on your `PATH`, `curl`; for Python,
`pip install openai`; for JavaScript, Node 18 or newer and `npm install openai`.

## Start a server with a chat model

<!-- test: needs Qwen3-0.6B-Q8_0.jlm -->
<!-- test: instead mkdir -p models && ln -s "$JITLLM_TEST_MODELS/Qwen3-0.6B-Q8_0.jlm" models/Qwen3-0.6B-Q8_0.jlm -->
```sh
jitllm convert -o models qwen3-0.6b
```

<!-- test: background -->
```sh
jitllmd serve -addr 127.0.0.1:8080 -models models -load Qwen3-0.6B-Q8_0.jlm -id qwen3-0.6b -devices cpu
```

```sh
for i in $(seq 60); do curl -sf http://127.0.0.1:8080/healthz && break; sleep 1; done
```
<!-- test: expect ok -->

The server asks for no API key. The SDKs insist on one, so pass any string.

## curl

Non-streaming, then streaming (`"stream": true` sends server-sent events, one
`data:` line per chunk, and `data: [DONE]` at the end):

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3-0.6b",
  "messages": [{"role": "system", "content": "You answer in one short sentence."},
               {"role": "user", "content": "What is the capital of France? /no_think"}],
  "max_tokens": 48, "temperature": 0}'
```
<!-- test: expect Paris -->

```sh
curl -sN http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3-0.6b", "stream": true,
  "messages": [{"role": "user", "content": "Count to three. /no_think"}],
  "max_tokens": 32, "temperature": 0}'
```
<!-- test: expect data: [DONE] -->

## Python

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")

reply = client.chat.completions.create(
    model="qwen3-0.6b",
    messages=[{"role": "user", "content": "What is the capital of France? /no_think"}],
    max_tokens=48,
    temperature=0,
)
print(reply.choices[0].message.content)

stream = client.chat.completions.create(
    model="qwen3-0.6b",
    messages=[{"role": "user", "content": "Count to three. /no_think"}],
    max_tokens=32,
    temperature=0,
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
print()
```
<!-- test: expect Paris -->

## JavaScript

Save as `chat.mjs` and run `node chat.mjs`:

```js
import OpenAI from 'openai'

const client = new OpenAI({ baseURL: 'http://127.0.0.1:8080/v1', apiKey: 'unused' })

const reply = await client.chat.completions.create({
  model: 'qwen3-0.6b',
  messages: [{ role: 'user', content: 'What is the capital of France? /no_think' }],
  max_tokens: 48,
  temperature: 0,
})
console.log(reply.choices[0].message.content)

const stream = await client.chat.completions.create({
  model: 'qwen3-0.6b',
  messages: [{ role: 'user', content: 'Count to three. /no_think' }],
  max_tokens: 32,
  temperature: 0,
  stream: true,
})
for await (const chunk of stream) process.stdout.write(chunk.choices[0]?.delta?.content ?? '')
console.log()
```
<!-- test: expect Paris -->

## Anthropic clients

The same server answers Anthropic's `POST /v1/messages`; an Anthropic SDK takes
`base_url="http://127.0.0.1:8080"`:

```sh
curl -s http://127.0.0.1:8080/v1/messages -H 'Content-Type: application/json' -d '{
  "model": "qwen3-0.6b", "max_tokens": 48,
  "messages": [{"role": "user", "content": "What is the capital of France? /no_think"}]}'
```
<!-- test: expect "type":"message" -->

## When it does not work

| You see | Cause | Do this |
|---|---|---|
| `openai.APIConnectionError` / `ECONNREFUSED` | wrong base URL or the server is down | the base URL ends in `/v1` for OpenAI SDKs and has no `/v1` for Anthropic ones; check `curl http://127.0.0.1:8080/healthz` |
| `404` with `no loaded model` | the `model` string is not a loaded id | `client.models.list()` lists the ids |
| `429` with `Retry-After` | the model's queue (`-max-queue`, 64 by default) is full | retry after the header's delay, or raise `-max-queue` |
| a field you sent is ignored or refused | not every OpenAI field is implemented | the 400 names the field; `docs/server.md` lists what is supported |
| the SDK times out on a long reply | a slow first token on a large model | raise the client timeout; [diagnose](diagnose.md) where the time goes |
