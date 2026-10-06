#!/usr/bin/env python3
"""Build the DeepSeek V3.2 fixtures and record transformers' logits.

    $JITLLM_HF_PY scripts/ds32gold.py [name ...]
    JITLLM_MODELS=/path/to/models ... (where the GGUFs go)

-> $JITLLM_MODELS/<name>.gguf and testdata/golden/deepseek32/<name>.json

c6gold.py's method: the oracle is transformers' OWN DeepseekV32ForCausalLM,
the GGUF is written by llama.cpp's OWN converter, every tensor is F32, and
every parameter and every buffer that is a weight is randomised -- the
router's e_score_correction_bias is a buffer transformers initialises to
zero, which is the identity.

The indexer keeps index_topk = 4 positions, so from the fifth of twelve
positions on the attention reads a subset and the sparse path is what the
golden measures; a top-k at or above the sequence would make DeepSeek Sparse
Attention exactly deepseek2's dense attention. Eight indexer heads keep an
exact tie at the selection boundary (every head's ReLU zero) unlikely, and
the Go gate's "a top-k one short" violation moving every later logit says the
boundary position is one the attention reads.

The tokenizer is DeepSeek-V3.2-Exp's own ($CV_TOK/deepseek-v32).
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
OUT = os.path.join(HERE, "..", "testdata", "golden", "deepseek32")
NPOS, NLOGIT = 12, 512
TEXTS = [
    "The capital of France is",
    "def getUserName(self):\n    return 12345",
    " leading space and  double  spaces\nline two\ttab",
    "naïve café — 東京 🙂 \n\n\n end",
]


def base(**kw):
    from transformers import DeepseekV32Config
    # MLA with every head width distinct (qk_nope 16, qk_rope 8, v 16, latent
    # 16, query latent 32), the V3 router with a grouped top-k that discards a
    # group, a dense lead block, and YaRN biting inside twelve positions (factor
    # 8 over 4), whose cos and sin the indexer's rotary shares.
    c = dict(vocab_size=129280, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
             num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
             n_shared_experts=1, n_routed_experts=8, num_experts_per_tok=2,
             n_group=2, topk_group=1, routed_scaling_factor=2.5, norm_topk_prob=True,
             kv_lora_rank=16, q_lora_rank=32, qk_rope_head_dim=8, v_head_dim=16,
             qk_nope_head_dim=16, first_k_dense_replace=1,
             index_topk=4, index_head_dim=16, index_n_heads=8,
             rms_norm_eps=1e-6, max_position_embeddings=2048, tie_word_embeddings=False,
             rope_parameters={"rope_type": "yarn", "rope_theta": 10000.0, "factor": 8.0,
                              "original_max_position_embeddings": 4, "beta_fast": 32.0,
                              "beta_slow": 1.0, "mscale": 1.0, "mscale_all_dim": 1.0},
             bos_token_id=0, eos_token_id=1)
    c.update(kw)
    return DeepseekV32Config(**c)


def ds32():
    from transformers import DeepseekV32ForCausalLM
    return DeepseekV32ForCausalLM, base()


SPECS = {
    "synth-deepseek32": ds32,
}


def perturb(m):
    # c6gold.perturb's classification, plus the buffer that is a weight: the
    # router's selection-only bias.
    with torch.no_grad():
        for n, p in m.named_parameters():
            last = n.split(".")[-2] if "." in n else ""
            if n.endswith("weight") and "norm" in last:
                p.copy_(torch.randn_like(p) * 0.3 + 1)
            elif n.endswith("bias"):
                p.copy_(torch.randn_like(p) * 0.3)
            else:
                p.copy_(torch.randn_like(p) * 0.2)
        for n, b in m.named_buffers():
            if n.endswith("e_score_correction_bias"):
                b.copy_(torch.randn_like(b) * 0.5)


def build(name):
    cls, cfg = SPECS[name]()
    cfg._attn_implementation = "eager"
    torch.manual_seed(20261004)
    m = cls(cfg).float().eval()
    perturb(m)
    d = os.path.join(HFDIR, name)
    shutil.rmtree(d, ignore_errors=True)
    m.save_pretrained(d, safe_serialization=True)
    for f in os.listdir(os.path.join(TOKDIR, "deepseek-v32")):
        if f != "config.json":
            shutil.copy(os.path.join(TOKDIR, "deepseek-v32", f), d)
    dst = os.path.join(MODELS, name + ".gguf")
    # llama.cpp's DeepseekV32Model.set_vocab asserts the tokenizer's
    # add_bos_token, which tokenizer_config.json sets and transformers 5's
    # TokenizersBackend does not surface; the shim sets it on the loaded
    # tokenizer and runs the converter unchanged.
    shim = ("import sys, runpy, transformers\n"
            "orig = transformers.AutoTokenizer.from_pretrained\n"
            "def load(*a, **k):\n"
            "    t = orig(*a, **k)\n"
            "    t.add_bos_token = True\n"
            "    return t\n"
            "transformers.AutoTokenizer.from_pretrained = load\n"
            "sys.argv = sys.argv[1:]\n"
            "sys.path.insert(0, __import__('os').path.dirname(sys.argv[0]))\n"
            "runpy.run_path(sys.argv[0], run_name='__main__')\n")
    subprocess.run([PY, "-c", shim, os.path.join(LCPP, "convert_hf_to_gguf.py"), d,
                    "--outtype", "f32", "--outfile", dst], check=True)
    ids = [(i * 7919 + 13) % min(cfg.vocab_size, 50000) for i in range(NPOS)]
    with torch.no_grad():
        lg = m(torch.tensor([ids]), use_cache=False).logits[0]
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
