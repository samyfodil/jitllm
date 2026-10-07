# Terminal app

The terminal app ([tui/](../tui/)) is the desktop app's screens in a terminal:
chat, models, discover, the engine and settings, over the same engine as
`jitllm` and `jitllmd`. It runs the engine in its own process, works over SSH,
and shares its settings and saved chats with the desktop app through
[common/](../common/), so a chat started in one is in the other's list.

Install it from a release with `scripts/install.sh tui` ([README](../README.md#install)),
or build it:

```sh
(cd tui && go build -o ../jitllm-tui .)
./jitllm-tui [-chat=true|false] [-device auto] [-ctx 8192] [-models DIR] [-mouse] [model.jlm]
```

With no model it opens on the Models screen. Each flag overrides its setting
for this run only; `-mouse` scrolls with the wheel, and the terminal's own text
selection then needs shift. The app paints its own background, so use a
terminal with 24-bit colour.

![The Chat screen: Qwen3-30B-A3B answering on a laptop with a 4 GB GPU, the engine panel beside the conversation](../website/public/shots/tui-chat.png)

## Screens

| Key | Screen | What it does |
|---|---|---|
| `F1` | Chat | saved chats, the transcript and the prompt, with the engine at the right: decode rate, where each block runs, memory |
| `F2` | Models | the `.jlm` files in your folders and the GGUF and safetensors sources beside them; `enter` loads a container, converts a source or rebuilds an outdated container, `ctrl+x` closes it |
| `F3` | Discover | the catalog ranked for this machine, fastest to smartest (`<` `>`), with whether each model fits; `enter` downloads and converts, or loads one already on disk |
| `F4` | Engine | every block and where it runs, the rate over time, memory and the pager; `→` `←` move one block to the device or home, `home` and `end` all of them |
| `F5` | Settings | the desktop app's settings file as a form; `enter` on the last field saves, `esc` discards |

`alt+1` to `alt+5` work as well, `esc` goes back to the chat, `ctrl+k` opens a
command palette over every action, and `ctrl+c` quits. The bottom line lists
the keys the current screen takes.

![The Engine screen: 1 of 48 blocks on the GPU, the decode rate over time, memory and the pager](../website/public/shots/tui-engine.png)

## Chat

`enter` sends, `ctrl+j` or `alt+enter` adds a line, `↑` `↓` walk the prompt
history, `esc` stops the reply or clears the prompt, `ctrl+t` folds the
reasoning, `ctrl+y` copies the last reply, `ctrl+n` starts a chat and
`ctrl+↑` `ctrl+↓` move between chats.

A prompt that starts with `/` is a command, completed with `tab`: `/new`,
`/delete`, `/chat`, `/completion`, `/system <text>`, `/temp <value>`,
`/seam <blocks>` (that many blocks on the device), `/think`, `/copy`,
`/models`, `/engine` and `/quit`.

A block moved between the CPU and the GPU takes its KV cache and recurrent
state with it, so the conversation keeps its history ([devices](devices.md),
[placement](placement.md)).
