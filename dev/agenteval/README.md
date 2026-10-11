# agenteval

Measures whether an agent can use and change jitllm: each task is set up in a
fresh workspace, handed to the agent command you name, and checked by machine
afterwards. Run it before and after a change to the docs, the recipes or the
CLI's messages, and compare the reports.

It calls no model API and no remote service. The agent is whatever command you
give it; with none, it does nothing and says so.

## The tasks

| id | the agent is asked to | completed when | required validation |
|---|---|---|---|
| `cpu-server` | start a CPU-only server from `stories260K.gguf` as model `stories` on a given port, and leave it running | `/v1/models` lists `stories`; a greedy completion continues the story; every block ran on the host | it ran `jitllm convert`, and sent the server a request with curl |
| `connect-client` | write `client.sh`, `client.py` or `client.mjs` that prints a greedy completion from a running server | the client prints the expected continuation and only the text | it ran a client (curl, python3 or node) |
| `unsupported-model` | say which of `alpha.gguf` and `beta.gguf` jitllm runs, in `verdict.json` | alpha supported; beta unsupported, its reason naming the architecture `ollmo`; neither file modified | it ran `jitllm` on `beta.gguf` |
| `placement` | find why a live server reads weights from disk every token, fix it without a restart, write `diagnosis.txt` | `jitllmd stats` reports the model `FITS`; the same server process still runs and answers; the diagnosis names the budget | it ran `jitllmd place`, `stats` or `models` |
| `server-bug` | fix a seeded bug (`parallel_tool_calls` read inverted) in a copy of the tree, and run the tests | a hidden test of `oaToolChoice` passes; the server package vets | it ran `go test` |

"Required validation" is read from the commands the agent ran: `jitllm`,
`jitllmd`, `go`, `curl`, `python3` and `node` are logging wrappers first on its
`PATH`.

## Running it

Under the cap, from the repository root:

    ./scripts/cap 12G -- taskset -c 0,2,4,6,8,10 go run ./dev/agenteval -agent 'COMMAND'

`COMMAND` runs with `bash -c` in the task's workspace, once per task. The prompt
is on its stdin, in `$AGENTEVAL_PROMPT`, and in the file `$AGENTEVAL_PROMPT_FILE`;
`$AGENTEVAL_TASK` is the task id. `JITLLM_AGENTEVAL_CMD` names the command when
`-agent` does not. A non-interactive agent CLI that takes its prompt from stdin
is used as it is, for example `-agent 'my-agent --print --allow-shell'`.

| flag | meaning |
|---|---|
| `-tasks a,b` | run only these tasks (`-list` lists them) |
| `-runs N` | run each task N times |
| `-out FILE` | the JSON report (default `agenteval-report.json`) |
| `-timeout D` | the agent's time per task (default 20m) |
| `-tool-call-pattern S` | a string counted once per tool call in the agent's stdout (default `"type":"tool_use"`, what stream-JSON agent transcripts carry); with no match the count is reported as unknown |
| `-work DIR`, `-keep` | where the workspaces go, and keep them afterwards |
| `-bin DIR` | use these `jitllm` and `jitllmd` instead of building them from the tree |

`-agent reference` runs each task's built-in solution, which shows that the task
can be completed and its checks pass. `TestTasksDiscriminate` runs that and an
agent that does nothing (`-agent true`), which must complete nothing.

## The report

```json
{
  "agent": "...", "revision": "<git HEAD>", "started": "...",
  "completed": 4, "required_checks_ran": 3,
  "results": [{
    "task": "placement", "workspace": "...",
    "completed": true, "checks": [{"name": "...", "passed": true, "detail": "..."}],
    "required_checks_ran": true, "required": [{"name": "...", "passed": true, "detail": "jitllmd\tplace -addr ..."}],
    "wall_seconds": 84.2, "timed_out": false, "agent_exit": 0,
    "tool_calls": 23, "commands": ["jitllmd\tstats -addr 127.0.0.1:41233", "..."]
  }]
}
```

Each workspace keeps the agent's stdout and stderr, the prompt, and the command
log under `.agenteval/` (with `-keep`). Incorrect assumptions are not scored by
machine; read them from the transcripts.

On Linux every process left running in a workspace is killed when its task ends, so a
task cannot leak a server into the next.
