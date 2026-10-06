#!/usr/bin/env python3
"""Build the MiniMax-M3 fixture and record transformers' logits.

    $JITLLM_HF_PY scripts/minimaxm3gold.py [name ...]
    JITLLM_MODELS=/path/to/models ... (where the GGUFs go)

-> $JITLLM_MODELS/<name>.gguf and testdata/golden/minimaxm3/<name>.json

c6gold.py's method: the oracle is transformers' OWN MiniMaxM3VLForCausalLM
(the text model of MiniMax-M3), the GGUF is written by llama.cpp's OWN
converter, every tensor is F32, and every parameter and every buffer that is a
weight is randomised -- the router's e_score_correction_bias is a buffer
transformers initialises to zero, which is the identity.

llama.cpp's converter reads the checkpoint as MiniMax publishes it
(MiniMaxM3SparseForCausalLM: per-expert w1/w2/w3, block_sparse_moe, the
indexer as self_attn.index_*, sparse_attention_config), which transformers 5
renames on load and does not write back; save_original() writes that layout
from the class's own tensors, so both references read the same weights.

MiniMax Sparse Attention selects key BLOCKS: blocks of 4 positions, the top 2
by each GQA group's max indexer score, the query's own block forced in. Over
24 positions that is six blocks, so from position 8 on every query reads a
subset and each kv group reads its own -- the sparse path is what the golden
measures. The first block is dense (moe_layer_freq 0): no indexer and dense
attention, as the real model's first three are.

The tokenizer is MiniMax-M3's own ($CV_TOK/minimax-m3).
"""
import json
import os
import shutil
import subprocess
import sys

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer

import refenv  # where things are: environment variables only (docs/testing.md)

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = refenv.models()
TOKDIR = refenv.under("CV_TOK", "tok")
HFDIR = refenv.under("CV_HF", "hf")
LCPP = refenv.need("LCPP_SRC", "a llama.cpp source checkout")
PY = sys.executable
OUT = os.path.join(HERE, "..", "testdata", "golden", "minimaxm3")
NPOS, NLOGIT = 24, 512
TEXTS = [
    "The capital of France is",
    "def getUserName(self):\n    return 12345",
    " leading space and  double  spaces\nline two\ttab",
    "naïve café — 東京 🙂 \n\n\n end",
]

# MiniMax-M3's shape at toy size: GQA 4:2 at head_dim 16, per-head Gemma q/k
# norms, NEOX rotary on half of each head, a dense lead block, then a sigmoid
# top-2 of 8 with the selection bias, renormalised, a routed scale and one
# shared expert, all swigluoai; one indexer head per kv group at head_dim.
SHAPE = dict(vocab_size=200064, hidden_size=64, intermediate_size=32, num_hidden_layers=3,
             num_attention_heads=4, num_key_value_heads=2, head_dim=16,
             dense_intermediate_size=64, shared_intermediate_size=32, num_local_experts=8,
             num_experts_per_tok=2, routed_scaling_factor=2.0, rotary_dim=8,
             swiglu_alpha=1.702, swiglu_limit=7.0, rms_norm_eps=1e-6,
             max_position_embeddings=4096, tie_word_embeddings=False,
             index_n_heads=2, index_head_dim=16, index_block_size=4, index_topk_blocks=2,
             index_local_blocks=1, bos_token_id=200034, eos_token_id=200020)
MOE_FREQ = [0, 1, 1]


def m3():
    from transformers import MiniMaxM3VLForCausalLM
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLTextConfig
    cfg = MiniMaxM3VLTextConfig(
        **SHAPE,
        rope_parameters={"rope_theta": 5000000.0, "rope_type": "default", "partial_rotary_factor": 0.5},
        layer_types=["minimax_m3_sparse" if f else "full_attention" for f in MOE_FREQ],
        mlp_layer_types=["sparse" if f else "dense" for f in MOE_FREQ])
    return MiniMaxM3VLForCausalLM, cfg


SPECS = {
    "synth-minimaxm3": m3,
}


def perturb(m):
    # c6gold.perturb's classification, plus the buffer that is a weight: the
    # router's selection-only bias. A Gemma norm stores w and scales by 1 + w,
    # so its weight is drawn around zero.
    with torch.no_grad():
        for n, p in m.named_parameters():
            last = n.split(".")[-2] if "." in n else ""
            if n.endswith("weight") and "norm" in last:
                p.copy_(torch.randn_like(p) * 0.3)
            else:
                p.copy_(torch.randn_like(p) * 0.2)
        for n, b in m.named_buffers():
            if n.endswith("e_score_correction_bias"):
                b.copy_(torch.randn_like(b) * 0.5)


def save_original(m, cfg, d):
    """MiniMax's own checkpoint layout, which llama.cpp's converter reads."""
    out = {}
    for n, t in m.state_dict().items():
        t = t.contiguous()
        if ".mlp.experts.gate_up_proj" in n:
            p = n.replace(".mlp.experts.gate_up_proj", ".block_sparse_moe.experts")
            i = t.shape[1] // 2
            for e in range(t.shape[0]):
                out[f"{p}.{e}.w1.weight"] = t[e, :i].clone()
                out[f"{p}.{e}.w3.weight"] = t[e, i:].clone()
        elif ".mlp.experts.down_proj" in n:
            p = n.replace(".mlp.experts.down_proj", ".block_sparse_moe.experts")
            for e in range(t.shape[0]):
                out[f"{p}.{e}.w2.weight"] = t[e].clone()
        elif n.endswith("gate_up_proj.weight"):
            i = t.shape[0] // 2
            p = n[:-len("gate_up_proj.weight")]
            p = p.replace(".mlp.shared_experts.", ".block_sparse_moe.shared_experts.")
            out[p + "gate_proj.weight"] = t[:i].clone()
            out[p + "up_proj.weight"] = t[i:].clone()
        elif ".mlp.shared_experts." in n:
            out[n.replace(".mlp.shared_experts.", ".block_sparse_moe.shared_experts.")] = t
        elif ".mlp.gate.e_score_correction_bias" in n:
            out[n.replace(".mlp.gate.e_score_correction_bias", ".block_sparse_moe.e_score_correction_bias")] = t
        elif ".mlp.gate.weight" in n:
            out[n.replace(".mlp.gate.", ".block_sparse_moe.gate.")] = t
        elif ".self_attn.indexer." in n:
            out[n.replace(".self_attn.indexer.", ".self_attn.index_")] = t
        else:
            out[n] = t
    save_file(out, os.path.join(d, "model.safetensors"), metadata={"format": "pt"})
    c = {k: v for k, v in SHAPE.items()}
    c.update({
        "architectures": ["MiniMaxM3SparseForCausalLM"],
        "model_type": "minimax_m3",
        "rope_theta": 5000000.0, "partial_rotary_factor": 0.5,
        "hidden_act": "swigluoai", "use_gemma_norm": True, "use_qk_norm": True,
        "qk_norm_type": "per_head", "n_shared_experts": 1, "scoring_func": "sigmoid",
        "use_routing_bias": True, "moe_layer_freq": MOE_FREQ, "torch_dtype": "float32",
        "sparse_attention_config": {
            "use_sparse_attention": True,
            "sparse_index_dim": SHAPE["index_head_dim"],
            "sparse_num_index_heads": SHAPE["index_n_heads"],
            "sparse_topk_blocks": SHAPE["index_topk_blocks"],
            "sparse_block_size": SHAPE["index_block_size"],
            "sparse_local_block": SHAPE["index_local_blocks"],
            "sparse_init_block": 0, "sparse_score_type": "max",
            "sparse_attention_freq": MOE_FREQ,
            "sparse_disable_index_value": MOE_FREQ,
        },
    })
    for k in ("index_n_heads", "index_head_dim", "index_block_size", "index_topk_blocks",
              "index_local_blocks"):
        c.pop(k)
    json.dump(c, open(os.path.join(d, "config.json"), "w"), indent=2)


def build(name):
    cls, cfg = SPECS[name]()
    cfg._attn_implementation = "eager"
    torch.manual_seed(20261004)
    m = cls(cfg).float().eval()
    perturb(m)
    d = os.path.join(HFDIR, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    save_original(m, cfg, d)
    for f in os.listdir(os.path.join(TOKDIR, "minimax-m3")):
        if f != "config.json":
            shutil.copy(os.path.join(TOKDIR, "minimax-m3", f), d)
    dst = os.path.join(MODELS, name + ".gguf")
    subprocess.run([PY, os.path.join(LCPP, "convert_hf_to_gguf.py"), d,
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
