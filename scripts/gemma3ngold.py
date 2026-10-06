#!/usr/bin/env python3
"""Build the Gemma 3n fixtures and record transformers' logits.

    $JITLLM_HF_PY scripts/gemma3ngold.py [name ...]
    JITLLM_MODELS=/path/to/models ... (where the GGUFs go)

-> $JITLLM_MODELS/<name>.gguf and testdata/golden/gemma3n/<name>.json

gemma4gold.py's method: the oracle is transformers' OWN Gemma3nForCausalLM, the
GGUF is written by llama.cpp's OWN converter, every tensor is F32, and every
parameter is randomised. The coefficient matrices AltUp initialises to zero
(correction and prediction) would make its predict and correct steps the
identity, and correct_output_scale's zeros would erase the per-layer input, so
the randomisation is what makes the gate see them.

The fixture carries every piece Gemma 3n has: four AltUp streams, LAuReL at
rank 32, per-layer inputs 32 wide (a device quantizes an activation in blocks
of 32), the gaussian top-k on the first three FFNs, the last two blocks
reading the KV of the last earlier block of their own kind, two periods of the
5-block sliding pattern with a window of 4 keys, and an attention scale of one
over q, k and an unweighted v norm.
"""
import json
import os
import shutil
import subprocess
import sys

import torch
from tokenizers import Tokenizer

import refenv  # where things are: environment variables only (docs/testing.md)

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = refenv.models()
TOKDIR = refenv.under("CV_TOK", "tok")
HFDIR = refenv.under("CV_HF", "hf")
LCPP = refenv.need("LCPP_SRC", "a llama.cpp source checkout")
PY = sys.executable
OUT = os.path.join(HERE, "..", "testdata", "golden", "gemma3n")
NPOS, NLOGIT = 12, 512
TEXTS = [
    "The capital of France is",
    "def getUserName(self):\n    return 12345",
    " leading space and  double  spaces\nline two\ttab",
    "naïve café — 東京 🙂 \n\n\n end",
]

S, F = "sliding_attention", "full_attention"


def gemma3n():
    from transformers import Gemma3nForCausalLM, Gemma3nTextConfig
    n = 10
    cfg = Gemma3nTextConfig(
        vocab_size=262400, vocab_size_per_layer_input=262144, hidden_size=64,
        intermediate_size=[128] * n, num_hidden_layers=n, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, layer_types=[S, S, S, S, F] * 2,
        sliding_window=4, final_logit_softcapping=30.0, hidden_size_per_layer_input=32,
        altup_num_inputs=4, altup_active_idx=0, altup_correct_scale=True, laurel_rank=32,
        num_kv_shared_layers=2, activation_sparsity_pattern=[0.95] * 3 + [0.0] * (n - 3),
        rms_norm_eps=1e-6, max_position_embeddings=4096, tie_word_embeddings=True,
        rope_parameters={S: {"rope_type": "default", "rope_theta": 10000.0},
                         F: {"rope_type": "default", "rope_theta": 1000000.0}},
        pad_token_id=0, eos_token_id=1, bos_token_id=2)
    return Gemma3nForCausalLM, cfg


SPECS = {
    "synth-gemma3n": gemma3n,
}


def perturb(m):
    with torch.no_grad():
        for n, p in m.named_parameters():
            last = n.split(".")[-2] if "." in n else ""
            if n.endswith("weight") and "norm" in last:
                p.copy_(torch.randn_like(p) * 0.3 + 1)
            elif n.endswith("correct_output_scale"):
                p.copy_(torch.randn_like(p) * 0.3 + 1)
            elif "correction_coefs" in n or "prediction_coefs" in n:
                p.copy_(torch.randn_like(p) * 0.3)
            else:
                p.copy_(torch.randn_like(p) * 0.2)


def build(name):
    cls, cfg = SPECS[name]()
    cfg._attn_implementation = "eager"
    torch.manual_seed(20261004)
    m = cls(cfg).float().eval()
    perturb(m)
    d = os.path.join(HFDIR, name)
    shutil.rmtree(d, ignore_errors=True)
    m.save_pretrained(d, safe_serialization=True)
    for f in os.listdir(os.path.join(TOKDIR, "gemma3n")):
        if f not in ("config.json", "full_config.json"):
            shutil.copy(os.path.join(TOKDIR, "gemma3n", f), d)
    dst = os.path.join(MODELS, name + ".gguf")
    subprocess.run([PY, os.path.join(LCPP, "convert_hf_to_gguf.py"), d, "--outtype", "f32",
                    "--outfile", dst], check=True)
    ids = [(i * 7919 + 13) % 50000 for i in range(NPOS)]
    with torch.no_grad():
        lg = m(torch.tensor([ids])).logits[0]
    rows = [{"argmax": int(r.argmax()), "max": float(r.max()),
             "head": [round(float(x), 7) for x in r[:NLOGIT]]} for r in lg]
    tk = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
    texts = [{"text": t, "ids": tk.encode(t, add_special_tokens=False).ids} for t in TEXTS]
    os.makedirs(OUT, exist_ok=True)
    json.dump({"ids": ids, "vocab": cfg.vocab_size, "pos": rows, "texts": texts},
              open(os.path.join(OUT, name + ".json"), "w"))
    print("wrote", dst, "argmax", [r["argmax"] for r in rows])


if __name__ == "__main__":
    for n in sys.argv[1:] or list(SPECS):
        build(n)
