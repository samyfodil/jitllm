# Recipes

Short, tested paths from an intent to a working result. Each recipe lists its
prerequisites, the commands, what they print, and what to do when they print
something else. Every command block in these files is run by
`internal/srcgate.TestRecipesRun` against a real `jitllmd`, so a recipe that
stops working fails a test instead of a reader.

| Recipe | What you end up with |
|---|---|
| [Start a server and make the first request](start-a-server.md) | `jitllmd` serving a model, and a completion and a chat reply from it |
| [Connect an OpenAI-compatible client](openai-clients.md) | the Python and JavaScript `openai` SDKs, and curl, talking to `jitllmd` |
| [Tools and structured output](tools-and-structured-output.md) | tool calls (chosen and forced) and replies held to a JSON schema |
| [Choose a supported checkpoint and convert it](convert-a-checkpoint.md) | a `.jlm` container, and a way to tell an unsupported model before downloading it |
| [Memory budgets and device placement](memory-and-placement.md) | a model held to a memory budget, and blocks placed where you ask |
| [Diagnose a slow or failed request](diagnose.md) | the server's own account of where a request's time and memory went |

All of them need the two binaries, `jitllm` and `jitllmd`, on your `PATH`
([install](../../README.md#install)), and `curl`. They run on a machine with no
GPU. Every server in them listens on `127.0.0.1:8080`.

The capability manifest (`docs/capabilities.json`, `jitllm capabilities`) and
`jitllm doctor` answer "does this build support my model and workload" and "what
is wrong with this machine" in machine-readable form, where a build has them.

## How the test runs a recipe

`TestRecipesRun` builds `jitllm` and `jitllmd` from the tree, then for each
recipe makes an empty working directory, puts the two binaries first on `PATH`,
and runs the fenced blocks in order:

- ` ```sh ` blocks run with `bash -euo pipefail`, each in a fresh shell (a block
  does not inherit another's variables).
- ` ```python ` blocks run with `JITLLM_RECIPE_PYTHON` (default `python3`) when
  it can `import openai`, and are skipped by name when it cannot.
- ` ```js ` blocks run as ES modules with `node` when it can `import 'openai'`
  (from the working directory, which links `JITLLM_RECIPE_NODE_MODULES` as its
  `node_modules` when that is set), and are skipped by name when it cannot.
- ` ```text ` and ` ```json ` blocks are shown, not run.

Every `127.0.0.1:80NN` address is rewritten to a free port, the same one for the
same address throughout a recipe. An HTML comment directly before or after a
block steers it:

| Comment | Meaning |
|---|---|
| `<!-- test: background -->` | before a block: start it and leave it running; it is stopped when the recipe ends |
| `<!-- test: expect TEXT -->` | after a block: its output must contain TEXT (several may follow one block) |
| `<!-- test: exit N -->` | before a block: it must exit with status N |
| `<!-- test: instead COMMAND -->` | before a block: run COMMAND in its place. Only for downloads: COMMAND copies the same file from the repository (`$JITLLM_REPO`) or the test model directory (`$JITLLM_TEST_MODELS`) |
| `<!-- test: needs FILE -->` | every later block needs FILE in the test model directory (`JITLLM_MODELS`); without it they are reported missing, which fails the test except in the model-free run, where it is a named skip |
| `<!-- test: skip REASON -->` | before a block: not run, and the reason is logged |

Run them:

    ./scripts/cap 8G -- taskset -c 0,2,4,6,8,10 go test ./internal/srcgate -count=1 -run TestRecipesRun -v

`JITLLM_RECIPE=diagnose` runs one recipe. CI's model-free run executes every
block that needs only the repository's own `testdata/models/stories260K.gguf`.
