#!/usr/bin/env python3
"""Record how Gemma 4's own tokenizer.json encodes a corpus.

    $JITLLM_HF_PY scripts/gemma4tokgold.py [tokenizer.json]

-> tok/testdata/gemma4_encode.json

The authority is HuggingFace `tokenizers` running the model's own
tokenizer.json (RULE 7m), with add_special_tokens off. The corpus carries
what its algorithm is about: spaces turned into U+2581, characters the
vocabulary lacks (byte fallback), newlines next to other characters (where
llama.cpp's gemma4 pre-split on newlines would part from the model's
tokenizer), runs of spaces, and a long stretch of prose and code.
"""
import json
import os
import sys

from tokenizers import Tokenizer

import refenv  # where things are: environment variables only (docs/testing.md)

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(refenv.under("CV_TOK", "tok"), "gemma4", "tokenizer.json")
OUT = os.path.join(HERE, "..", "tok", "testdata", "gemma4_encode.json")

CASES = [
    "The capital of France is",
    "def getUserName(self):\n    return 12345",
    " leading space and  double  spaces\nline two\ttab",
    "end.\nnext line.\n\nafter a blank line\n",
    "\n\n\nthree newlines\r\n\r\ncrlf  \n  \n mixed",
    "a b  c   d    e     f      g       h        i         j",
    "   leading and trailing   ",
    "日本語のテキストです。カタカナとひらがな",
    "中文abc混合English词 한국어 텍스트입니다",
    "emoji 🎉🔥 mixed 👍🏽 with ascii 🫠",
    "café naïve résumé é combining à́ marks",
    "Ω≈ç√∫˜µ≤≥÷ ©®™ §¶ ∑_{i=1}^{n} x_i² ≥ 0",
    "Привет, мир! Γειά σου κόσμε مرحبا שלום",
    "rare: 𓀀 𝔘𝔫𝔦𝔠𝔬𝔡𝔢 ꙮ   ­ soft",
    "for (int i = 0; i < n; ++i) {\n\tsum += a[i];\n}",
    "<html><body><p>text</p></body></html>",
    "12345 1234567 3.14159 (42) -7 x86_64 2025年10月3日",
    "don't won't I'm we'll they've she'd it's",
    "",
    "a",
    " ",
    "\n",
    "▁already▁escaped",
]


def main():
    tk = Tokenizer.from_file(SRC)
    agents = open(os.path.join(HERE, "..", "AGENTS.md"), encoding="utf-8").read()
    cases = CASES + [agents[:20000], agents[20000:26000]]
    out = [{"text": t, "ids": tk.encode(t, add_special_tokens=False).ids} for t in cases]
    json.dump(out, open(OUT, "w"), ensure_ascii=False)
    print("wrote", OUT, len(out), "cases,", sum(len(c["ids"]) for c in out), "ids")


if __name__ == "__main__":
    main()
