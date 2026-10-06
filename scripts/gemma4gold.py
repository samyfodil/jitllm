#!/usr/bin/env python3
"""Build the Gemma 4 fixtures and record transformers' logits.

    $JITLLM_HF_PY scripts/gemma4gold.py [name ...]
    JITLLM_MODELS=/path/to/models ... (where the GGUFs go)

-> $JITLLM_MODELS/<name>.gguf and testdata/golden/gemma4/<name>.json

moegold.py's method: the oracle is transformers' OWN Gemma4ForCausalLM, the
GGUF is written by llama.cpp's OWN converter, every tensor is F32, and every
parameter and every buffer that is a weight is randomised -- layer_scalar is a
buffer transformers initialises to one, which is the identity.

Every fixture puts the global layers' geometry apart from the sliding layers'
on every axis Gemma 4 does: head_dim 32 against 16, one kv head against two,
and the proportional rotary (a quarter of 32 rotated at base^(-2i/32)) against
the plain one (all of 16 at base^(-2i/16)). The window is 4 keys, so twelve
positions cross it.
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
OUT = os.path.join(HERE, "..", "testdata", "golden", "gemma4")
NPOS, NLOGIT = 12, 512
TEXTS = [
    "The capital of France is",
    "def getUserName(self):\n    return 12345",
    " leading space and  double  spaces\nline two\ttab",
    "naïve café — 東京 🙂 \n\n\n end",
]

S, F = "sliding_attention", "full_attention"


def base(**kw):
    from transformers import Gemma4TextConfig
    c = dict(vocab_size=262144, hidden_size=64, intermediate_size=128, num_hidden_layers=6,
             num_attention_heads=4, num_key_value_heads=2, head_dim=16, global_head_dim=32,
             num_global_key_value_heads=1, attention_k_eq_v=True,
             layer_types=[S, S, F, S, S, F], sliding_window=4, final_logit_softcapping=30.0,
             hidden_size_per_layer_input=0, num_kv_shared_layers=0, rms_norm_eps=1e-6,
             max_position_embeddings=4096, tie_word_embeddings=True,
             rope_parameters={S: {"rope_type": "default", "rope_theta": 10000.0},
                              F: {"rope_type": "proportional", "partial_rotary_factor": 0.25,
                                  "rope_theta": 1000000.0}},
             pad_token_id=0, eos_token_id=1, bos_token_id=2)
    c.update(kw)
    return Gemma4TextConfig(**c)


def gemma4():
    # The 31B's shape: dense, K = V on the global layers (no v_proj there),
    # one global kv head against two sliding ones.
    from transformers import Gemma4ForCausalLM
    return Gemma4ForCausalLM, base()


def gemma4_kv():
    # The E-models' attention: a v_proj on every layer (attention_k_eq_v
    # false), the same kv head count everywhere.
    from transformers import Gemma4ForCausalLM
    return Gemma4ForCausalLM, base(attention_k_eq_v=False, num_global_key_value_heads=None)


def gemma4_moe():
    # The 26B-A4B's block: a dense MLP and a mixture in parallel on every
    # layer, each with its own pre and post norms, a router on an unweighted
    # RMSNorm times router.scale over sqrt(d), and per_expert_scale on the
    # renormalised weights.
    from transformers import Gemma4ForCausalLM
    return Gemma4ForCausalLM, base(enable_moe_block=True, num_experts=8, top_k_experts=2,
                                   moe_intermediate_size=32)


def gemma4_e():
    # The E2B/E4B's block: per-layer embeddings (a second token table, a
    # projection of the scaled embedding, and a gated residual per block),
    # the last two layers reading the KV of the last earlier layer of their
    # own type, and those two layers' MLP double width. A v_proj everywhere
    # and one kv head on both geometries, as the E-models have.
    from transformers import Gemma4ForCausalLM
    return Gemma4ForCausalLM, base(attention_k_eq_v=False, num_global_key_value_heads=None,
                                   num_key_value_heads=1, hidden_size_per_layer_input=32,
                                   vocab_size_per_layer_input=262144, num_kv_shared_layers=2,
                                   use_double_wide_mlp=True)


def gemma4_wide():
    # The real head widths: 256 on the sliding layers and 512 on the global
    # ones, as every Gemma 4 has, so the attention kernels run at the widths a
    # checkpoint asks for rather than at the small fixtures' 16 and 32.
    from transformers import Gemma4ForCausalLM
    return Gemma4ForCausalLM, base(hidden_size=256, intermediate_size=256, num_attention_heads=2,
                                   num_key_value_heads=1, head_dim=256, global_head_dim=512)


SPECS = {
    "synth-gemma4-wide": gemma4_wide,
    "synth-gemma4": gemma4,
    "synth-gemma4-kv": gemma4_kv,
    "synth-gemma4-moe": gemma4_moe,
    "synth-gemma4-e": gemma4_e,
}


def perturb(m):
    # moegold.perturb's classification, plus the buffers that are weights:
    # layer_scalar (initialised to one, the identity).
    with torch.no_grad():
        for n, p in m.named_parameters():
            last = n.split(".")[-2] if "." in n else ""
            if n.endswith("weight") and "norm" in last:
                p.copy_(torch.randn_like(p) * 0.3 + 1)
            elif n.endswith("per_expert_scale") or n.endswith("router.scale"):
                p.copy_(torch.randn_like(p) * 0.3 + 1)
            elif n.endswith("bias"):
                p.copy_(torch.randn_like(p) * 0.3)
            else:
                p.copy_(torch.randn_like(p) * 0.2)
        for n, b in m.named_buffers():
            if n.endswith("layer_scalar"):
                b.copy_(torch.rand_like(b) * 0.8 + 0.6)


def build(name):
    cls, cfg = SPECS[name]()
    cfg._attn_implementation = "eager"
    torch.manual_seed(20261003)
    m = cls(cfg).float().eval()
    perturb(m)
    d = os.path.join(HFDIR, name)
    shutil.rmtree(d, ignore_errors=True)
    m.save_pretrained(d, safe_serialization=True)
    for f in os.listdir(os.path.join(TOKDIR, "gemma4")):
        if f != "config.json":
            shutil.copy(os.path.join(TOKDIR, "gemma4", f), d)
    dst = os.path.join(MODELS, name + ".gguf")
    subprocess.run([PY, os.path.join(LCPP, "convert_hf_to_gguf.py"), d, "--outtype", "f32",
                    "--outfile", dst], check=True)
    ids = [(i * 7919 + 13) % min(cfg.vocab_size, 50000) for i in range(NPOS)]
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
