#!/usr/bin/env python3
"""Dump greedy continuations from llama.cpp, once. -> testdata/golden/run/*.json

  JITLLM_LCPP=... JITLLM_MODELS=... python3 scripts/rungold.py

-no-cnv is mandatory: without it llama-completion wraps the prompt in the
model's chat template (tinyllama has one) and the comparison is against a
different prompt entirely.
"""
import json, os, subprocess, sys

import refenv  # where things are: environment variables only (docs/testing.md)

LIB = refenv.lcpp_bin()
BIN = os.path.join(LIB, "llama-completion")
OUT = os.path.join(os.path.dirname(__file__), "..", "testdata", "golden", "run")
HERE = os.path.dirname(__file__)
MODELS = {
    "stories260K": (os.path.join(HERE, "..", "testdata", "models", "stories260K.gguf"), 32),
    "stories15M":  (os.path.join(HERE, "..", "models", "stories15M-q4_0.gguf"), 24),
    "tinyllama":   (os.path.join(refenv.models(), "tinyllama-1.1b-q3_K_M.gguf"), 12),
    "qwen3moe":    (os.path.join(HERE, "..", "models", "Qwen3-MOE-4x0.6B-Q4_K_M.gguf"), 20),
    "qwen2vl":     (os.path.join(HERE, "..", "models", "Qwen2-VL-2B-Instruct-Q4_K_M.gguf"), 20),
}
PROMPTS = ["Once upon a time", "The capital of France is", "One day, a little"]

def gen(model, prompt, n):
    cmd = [BIN, "-m", model, "-p", prompt, "--temp", "0", "--top-k", "1", "-s", "1",
           "-fa", "off", "-dev", "none", "-n", str(n), "--no-warmup", "-no-cnv", "-t", "6"]
    r = subprocess.run(cmd, capture_output=True, env=dict(os.environ, LD_LIBRARY_PATH=LIB))
    if r.returncode != 0:
        sys.exit("llama-completion failed: %s" % r.stderr.decode()[-800:])
    text = r.stdout.decode()
    # llama.cpp echoes the prompt (with the tokenizer's leading space) then the
    # continuation; strip a trailing "> EOF by user" line if present.
    text = text.split("\n> EOF by user")[0]
    # ★ THE LEADING SPACE IS THE TOKENIZER'S, NOT llama.cpp's, SO IT IS
    # OPTIONAL. A SentencePiece vocabulary renders the prompt back with a
    # leading space and a BPE one (qwen2, qwen2vl, qwen3moe) does not -- this
    # exited with "unexpected echo" on every BPE model rather than on a real
    # problem, which is a harness that only knew one model family.
    for lead in (" " + prompt, prompt):
        if text.startswith(lead):
            return text[len(lead):]
    sys.exit("unexpected echo: %r" % text[:120])

def main():
    if not os.path.exists(BIN): sys.exit("no llama-completion")
    os.makedirs(OUT, exist_ok=True)
    for name, (path, n) in MODELS.items():
        if not os.path.exists(path):
            print("%-12s SKIP" % name); continue
        rows = [{"prompt": p, "n": n, "out": gen(path, p, n)} for p in PROMPTS]
        with open(os.path.join(OUT, name + ".json"), "w") as fh:
            json.dump(rows, fh, ensure_ascii=False, indent=1)
        print("%-12s %d prompts x %d tokens" % (name, len(rows), n))
        for r in rows: print("     %-28r -> %r" % (r["prompt"], r["out"][:60]))

if __name__ == "__main__":
    main()
