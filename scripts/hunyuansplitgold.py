#!/usr/bin/env python3
"""Record how Hunyuan's own tokenizer.json pre-tokenizes a corpus.

    $JITLLM_HF_PY scripts/hunyuansplitgold.py [tokenizer.json [name]]

-> tok/testdata/<name>_split.json and tok/testdata/pre/<name>.json, name
hunyuan-dense by default ("-" becomes "_" in the first). Spark-X2.5's
tokenizer.json, with name spark2_5, is the same three whole-pattern Splits
with a variant third pattern and a Digits stage after them.

The authority is HuggingFace `tokenizers` running the model's own
pre_tokenizer (RULE 7m), not llama.cpp's re-implementation. The pieces are the
ORIGINAL text sliced at the offsets tokenizers reports, i.e. before the
ByteLevel stage maps bytes to its alphabet, which is what tok's pipeline
compares at.
"""
import json
import os
import sys

from tokenizers import Tokenizer

import refenv  # where things are: environment variables only (docs/testing.md)

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = refenv.arg(1, "hf", "Hunyuan-0.5B-Instruct", "tokenizer.json")
OUT = os.path.join(HERE, "..", "tok", "testdata")
NAME = sys.argv[2] if len(sys.argv) > 2 else "hunyuan-dense"

CASES = [
    "def getUserName(self):\n    return 12345",
    "The capital of France is",
    " leading space and  double  spaces\nline two\ttab",
    "12345 1234567 3.14159 (42) -7 x86_64 2025年10月3日",
    "价格是12345元，日期2025-10-03。",
    "日本語のテキストです。カタカナとひらがな",
    "中文abc混合English词",
    "(foo) [bar] {baz} 'quote' \"double\" `tick`",
    "don't won't I'm we'll they've she'd it's",
    "x+=1; y-->0 && z!=2 || w<=3",
    "a/b/c http://example.com/path?q=1&r=2#frag",
    "\n\n\nthree newlines\r\n\r\ncrlf  \n  \n mixed",
    "trailing spaces   ",
    "   ",
    "tab\t\tdouble tab\t",
    "emoji 🎉🔥 mixed 👍🏽 with ascii",
    "café naïve résumé",
    "é combining à́ marks ́alone",
    "nbsp here em-space　ideographic",
    "zero​width‍joiner﻿bom",
    "Ω≈ç√∫˜µ≤≥÷ ©®™ §¶",
    "math: ∑_{i=1}^{n} x_i² ≥ 0",
    "Ⅻ ½ ① ٣ ४ ൬",
    "Привет, мир! Γειά σου κόσμε",
    "مرحبا بالعالم שלום עולם",
    "!!!??? ... --- ___ ***",
    ".hidden -flag --long-flag @user #tag $var %mod",
    "C++ C# F# .NET node.js",
    "MixedCASE_123-456 snake_case camelCase",
    "\x00\x01\x7f binary-ish",
    "",
    "a",
    "1",
    " 1",
    "한국어 텍스트입니다",
    "ﾊﾝｶｸｶﾀｶﾅ and ｆｕｌｌｗｉｄｔｈ",
    "end.\n\nnext!\r\n\r\nthird?\n",
    "a,\nb;\r\n c:\n\n d",
    "  \n  \n\t\nindented\n    block",
    "x)\n}\n]\n",
    "line sep para",
]


def main():
    tk = Tokenizer.from_file(SRC)
    pre = tk.pre_tokenizer
    out = []
    for t in CASES:
        pieces = [t[a:b] for _, (a, b) in pre.pre_tokenize_str(t)]
        if "".join(pieces) != t:
            raise SystemExit("pieces of %r do not reassemble it: %r" % (t, pieces))
        out.append({"text": t, "pieces": pieces})
    with open(os.path.join(OUT, NAME.replace("-", "_") + "_split.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    obj = json.load(open(SRC))["pre_tokenizer"]
    with open(os.path.join(OUT, "pre", NAME + ".json"), "w") as f:
        json.dump({"pre_tokenizer": obj}, f, ensure_ascii=False, indent=1)
    print("wrote", len(out), "cases")


if __name__ == "__main__":
    main()
