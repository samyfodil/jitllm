#!/usr/bin/env python3
"""Dump golden tokenizations from llama.cpp's own tokenizer, once.

  JITLLM_LCPP=... JITLLM_MODELS=... python3 scripts/tokgold.py    # -> testdata/golden/tokens/<model>.json

Uses -f (a file) rather than -p so tabs, newlines and multi-byte text survive
argv intact. Test strings deliberately have no trailing newline: llama-tokenize
strips one, and a golden that depends on that is a golden that breaks silently.
"""
import json, os, subprocess, sys, tempfile

import refenv  # where things are: environment variables only (docs/testing.md)

LIB = refenv.lcpp_bin()
BIN = os.path.join(LIB, "llama-tokenize")
OUT = os.path.join(os.path.dirname(__file__), "..", "testdata", "golden", "tokens")
MODELS = {
    "tinyllama": os.path.join(refenv.models(), "tinyllama-1.1b-q3_K_M.gguf"),
    "stories15M": os.path.join(os.path.dirname(__file__), "..", "models", "stories15M-q4_0.gguf"),
    "stories260K": os.path.join(os.path.dirname(__file__), "..", "testdata", "models", "stories260K.gguf"),
}
CASES = [
    "Hello world! The quick brown fox — 42 café.",
    "Hello",
    "",
    " ",
    "   leading spaces",
    "trailing spaces   ",
    "tab\there and\ttwo",
    "line one\nline two",
    "日本語のテキストです",
    "emoji 🎉🔥 mixed with ascii",
    "MixedCASE_123-456 snake_case camelCase",
    "a",
    "  ",
    "the the the the",
    "\x00\x01\x7f binary-ish",
    "Ω≈ç√∫˜µ≤≥÷",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "<s>not really bos</s>",
]

def ids(model, text, no_bos=False):
    with tempfile.NamedTemporaryFile("wb", delete=False) as fh:
        fh.write(text.encode())
        path = fh.name
    try:
        cmd = [BIN, "-m", model, "-f", path, "--ids", "--no-parse-special"]
        if no_bos:
            cmd.append("--no-bos")
        env = dict(os.environ, LD_LIBRARY_PATH=LIB)
        r = subprocess.run(cmd, capture_output=True, env=env)
        if r.returncode != 0:
            sys.exit("llama-tokenize failed: %s" % r.stderr.decode()[-500:])
        line = [l for l in r.stdout.decode().splitlines() if l.startswith("[")][-1]
        return json.loads(line)
    finally:
        os.unlink(path)

def main():
    if not os.path.exists(BIN):
        sys.exit("no llama-tokenize at %s" % BIN)
    os.makedirs(OUT, exist_ok=True)
    for name, path in MODELS.items():
        if not os.path.exists(path):
            print("%-12s SKIP (no model)" % name); continue
        rows = []
        for t in CASES:
            rows.append({"text": t, "bos": ids(path, t), "nobos": ids(path, t, no_bos=True)})
        with open(os.path.join(OUT, name + ".json"), "w") as fh:
            json.dump(rows, fh, ensure_ascii=False, indent=1)
        print("%-12s %d cases" % (name, len(rows)))

if __name__ == "__main__":
    main()
