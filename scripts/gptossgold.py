#!/usr/bin/env python3
"""Build a small gpt-oss, write it as a GGUF, and record transformers' logits.

    $JITLLM_HF_PY scripts/gptossgold.py

-> models/synth-gptoss.gguf and testdata/golden/gptoss/synth-gptoss.json

THE ORACLE FOR THE gpt-oss GRAPH IS transformers' OWN GptOssForCausalLM. The real
gpt-oss-20b is 42 GB in f32 and does not fit this box; this one has gpt-oss's
whole feature set at a size that does, and it is small enough to ship to the M4,
which holds no model large enough otherwise:

  - two layers, the first sliding over a window of 4, so 12 positions reach it;
  - attention sinks, biased q/k/v/o, GQA 4/2 at head_dim 16;
  - YaRN with gpt-oss's own parameters, truncate false;
  - a biased router over 4 experts, top 2, every bank biased;
  - swiglu-oai, with expert weights large enough that the gate crosses 7;
  - the expert banks in MXFP4. They are quantized here with gguf-py and then
    DEQUANTIZED BACK INTO THE TRANSFORMERS MODEL, so both engines run exactly
    the same weights and the comparison is of the graph, not of a quantizer.

The embedding and the head are f16 (rounded the same way on both sides) and
everything else f32. The tokenizer is gpt-oss-20b's own, copied from its GGUF.
"""
import json
import os

import numpy as np
import torch
from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter
from gguf.quants import dequantize, quantize
from transformers import GptOssConfig, GptOssForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "..", "models")
SRC = os.path.join(MODELS, "gpt-oss-20b-Q4_K_M.gguf")
DST = os.path.join(MODELS, "synth-gptoss.gguf")
OUT = os.path.join(HERE, "..", "testdata", "golden", "gptoss", "synth-gptoss.json")
H, F, E, K, NH, NKV, D, L, WIN = 64, 64, 4, 2, 4, 2, 16, 2, 4
NPOS, NLOGIT = 12, 256
MX = GGMLQuantizationType.MXFP4

src = GGUFReader(SRC)
field = lambda k: src.fields[k].contents()
tokens = field("tokenizer.ggml.tokens")
V = len(tokens)

cfg = GptOssConfig(
    num_hidden_layers=L, num_local_experts=E, vocab_size=V, hidden_size=H,
    intermediate_size=F, head_dim=D, num_attention_heads=NH, num_key_value_heads=NKV,
    sliding_window=WIN, num_experts_per_tok=K, max_position_embeddings=131072,
    rope_parameters={"rope_type": "yarn", "rope_theta": 150000.0, "factor": 32.0,
                     "beta_fast": 32.0, "beta_slow": 1.0, "truncate": False,
                     "original_max_position_embeddings": 4096},
)
torch.manual_seed(20260921)
m = GptOssForCausalLM(cfg).float().eval()
P = dict(m.named_parameters())
with torch.no_grad():
    for n, p in P.items():
        if n.endswith("layernorm.weight") or n == "model.norm.weight":
            p.copy_(torch.randn_like(p) * 0.3 + 1)
        elif "experts" in n and not n.endswith("bias"):
            p.copy_(torch.randn_like(p) * 0.5)  # large enough that gate crosses 7
        elif n.endswith("sinks"):
            p.copy_(torch.randn_like(p))
        elif n.endswith("bias"):
            p.copy_(torch.randn_like(p) * 0.2)
        else:
            p.copy_(torch.randn_like(p) * 0.2)
    for n in ("model.embed_tokens.weight", "lm_head.weight"):
        P[n].copy_(P[n].half().float())


def mx(w):
    """Quantize rows of w (rows, k) to MXFP4; return the bytes and the exact values."""
    q = quantize(np.ascontiguousarray(w, dtype=np.float32), MX)
    return q, dequantize(q, MX).reshape(w.shape)


tensors = []  # (name, array, dtype or None, logical shape)
with torch.no_grad():
    tensors.append(("token_embd.weight", P["model.embed_tokens.weight"].numpy().astype(np.float16), None, None))
    tensors.append(("output.weight", P["lm_head.weight"].numpy().astype(np.float16), None, None))
    tensors.append(("output_norm.weight", P["model.norm.weight"].numpy(), None, None))
    for i in range(L):
        pre = "model.layers.%d." % i
        g = lambda s: P[pre + s]
        b = "blk.%d." % i
        tensors += [
            (b + "attn_norm.weight", g("input_layernorm.weight").numpy(), None, None),
            (b + "post_attention_norm.weight", g("post_attention_layernorm.weight").numpy(), None, None),
            (b + "attn_sinks.weight", g("self_attn.sinks").numpy(), None, None),
            (b + "ffn_gate_inp.weight", g("mlp.router.weight").numpy(), None, None),
            (b + "ffn_gate_inp.bias", g("mlp.router.bias").numpy(), None, None),
        ]
        for hf, gg in (("q", "attn_q"), ("k", "attn_k"), ("v", "attn_v"), ("o", "attn_output")):
            tensors.append((b + gg + ".weight", g("self_attn.%s_proj.weight" % hf).numpy(), None, None))
            tensors.append((b + gg + ".bias", g("self_attn.%s_proj.bias" % hf).numpy(), None, None))
        gu, gub = g("mlp.experts.gate_up_proj"), g("mlp.experts.gate_up_proj_bias")
        dn, dnb = g("mlp.experts.down_proj"), g("mlp.experts.down_proj_bias")
        banks = {"gate": [], "up": [], "down": []}
        for e in range(E):
            # transformers: gate_up (H, 2F), gate = [:, ::2], up = [:, 1::2];
            # down (F, H). The GGUF stores each expert as rows x k.
            for name, cols in (("gate", slice(0, None, 2)), ("up", slice(1, None, 2))):
                q, deq = mx(gu[e][:, cols].T.numpy())
                gu[e][:, cols] = torch.from_numpy(deq.T.copy())
                banks[name].append(q)
            q, deq = mx(dn[e].T.numpy())
            dn[e].copy_(torch.from_numpy(deq.T.copy()))
            banks["down"].append(q)
        for name, rows, k in (("gate", F, H), ("up", F, H), ("down", H, F)):
            tensors.append((b + "ffn_%s_exps.weight" % name, np.stack(banks[name]), MX, (E, rows, k)))
        tensors.append((b + "ffn_gate_exps.bias", gub[:, 0::2].numpy().copy(), None, None))
        tensors.append((b + "ffn_up_exps.bias", gub[:, 1::2].numpy().copy(), None, None))
        tensors.append((b + "ffn_down_exps.bias", dnb.numpy().copy(), None, None))

w = GGUFWriter(DST, "gpt-oss")
w.add_block_count(L)
w.add_context_length(131072)
w.add_embedding_length(H)
w.add_feed_forward_length(F)
w.add_head_count(NH)
w.add_head_count_kv(NKV)
w.add_rope_freq_base(150000.0)
w.add_layer_norm_rms_eps(cfg.rms_norm_eps)
w.add_expert_count(E)
w.add_expert_used_count(K)
w.add_key_length(D)
w.add_value_length(D)
w.add_sliding_window(WIN)
w.add_expert_feed_forward_length(F)
w.add_rope_scaling_type(__import__("gguf").RopeScalingType.YARN)
w.add_rope_scaling_factor(32.0)
w.add_rope_scaling_orig_ctx_len(4096)
w.add_tokenizer_model("gpt2")
w.add_tokenizer_pre("gpt-4o")
w.add_token_list(tokens)
w.add_token_types(field("tokenizer.ggml.token_type"))
w.add_token_merges(field("tokenizer.ggml.merges"))
w.add_bos_token_id(field("tokenizer.ggml.bos_token_id"))
w.add_eos_token_id(field("tokenizer.ggml.eos_token_id"))
for name, arr, dt, shape in tensors:
    if dt is None:
        w.add_tensor(name, np.ascontiguousarray(arr))
    else:
        w.add_tensor(name, arr, raw_dtype=dt)  # byte shape; gguf-py derives (E, rows, k)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()

ids = [(i * 7919 + 13) % 199000 for i in range(NPOS)]
with torch.no_grad():
    lg = m(torch.tensor([ids])).logits[0]
rows = [{"argmax": int(r.argmax()), "max": float(r.max()), "lse": float(torch.logsumexp(r, 0)),
         "head": [round(float(x), 7) for x in r[:NLOGIT]]} for r in lg]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump({"ids": ids, "vocab": V, "pos": rows}, open(OUT, "w"))
print("wrote", DST, "argmax", [r["argmax"] for r in rows])
