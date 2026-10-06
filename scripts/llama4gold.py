#!/usr/bin/env python3
"""Build a small Llama 4, write it as a GGUF, and record transformers' logits.

    $JITLLM_HF_PY scripts/llama4gold.py [tokenizer.json]
    Q8=1 $JITLLM_HF_PY scripts/llama4gold.py

-> models/synth-llama4.gguf and testdata/golden/llama4/synth-llama4.json, or
   with Q8=1 models/synth-llama4-q8.gguf and .../synth-llama4-q8.json

Q8=1 writes every matrix -- embedding, head, attention, dense and expert FFNs
-- as Q8_0, quantized here with gguf-py and DEQUANTIZED BACK INTO THE
TRANSFORMERS MODEL before the golden is taken, so both engines run the same
weights. It is the fixture that reaches the device's QUANTIZED paths: a float
expert bank has no grouped (batched-prefill) mixture kernel, so the F32
fixture cannot exercise the input-weighted grouped mixture at all.

THE ORACLE FOR THE llama4 GRAPH IS transformers' OWN Llama4ForCausalLM. The real
Scout is 108B parameters; this one carries every text feature at a size that
fits anywhere, with each one set to a value that BITES inside the twelve golden
positions -- a feature a fixture cannot see is a feature it does not gate:

  - four layers, so layer 3 is the one NoPE layer (no_rope_layer_interval 4);
  - attention_chunk_size 4, so positions 4 and 8 each start a chunk and see
    ONE key on the chunked layers where a sliding window would see four;
  - floor_scale 4, so the NoPE layer's temperature steps at positions 3, 7 and
    11 (Scout's 8192 would leave it 1 for the whole golden);
  - use_qk_norm, the weightless L2 norm after the rotary;
  - interleave_moe_layer_step 2, so layers 0 and 2 are DENSE (FFN 128) and 1
    and 3 are mixtures of 4 experts (FFN 64) -- Scout is step 1, Maverick 2,
    and step 2 exercises both kinds of block;
  - top-1 sigmoid routing with the weight on the expert's INPUT, and the
    ungated shared expert;
  - llama3 rotary scaling over a SHORT original context (32), so most rotary
    pairs are rescaled; the factors are written as rope_freqs.weight exactly as
    llama.cpp's converter computes them.

Every tensor is F32 so the comparison is of the graph, not of a quantizer, and
the q/k projections are NOT permuted: Llama 4 rotates adjacent pairs through
view_as_complex, which is llama.cpp's NORM rotary (its converter sets
undo_permute = False for this class).

The tokenizer is Llama 4's own tokenizer.json, which the gated
meta-llama/Llama-4-Scout-17B-16E-Instruct and the ungated
unsloth/Llama-4-Scout-17B-16E-Instruct share byte for byte (one LFS pointer
blob, b1fde397...). The golden also records its ids for a few texts, through
the `tokenizers` library, so the container's tokenizer is gated against the
model's own artefact.

The non-default keys (attention.chunk_size, attention.temperature_length) are
what jitllm's converter reads for Llama 4; llama.cpp hardcodes both and would
run this file at 8192.
"""
import json
import math
import os
import sys

import numpy as np
import torch
from gguf import GGMLQuantizationType, GGUFWriter
from gguf.quants import dequantize, quantize
from tokenizers import Tokenizer
from transformers import Llama4ForCausalLM, Llama4TextConfig

import refenv  # where things are: environment variables only (docs/testing.md)

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.environ.get("JITLLM_MODELS", os.path.join(HERE, "..", "models"))
Q8 = os.environ.get("Q8") == "1"
STEM = "synth-llama4-q8" if Q8 else "synth-llama4"
DST = os.path.join(MODELS, STEM + ".gguf")
OUT = os.path.join(HERE, "..", "testdata", "golden", "llama4", STEM + ".json")
Q8_0 = GGMLQuantizationType.Q8_0
TOK = sys.argv[1] if len(sys.argv) > 1 else os.path.join(refenv.need("L4_TOK", "the directory holding Llama 4's tokenizer.json"), "tokenizer.json")

H, F, FM, E, K, NH, NKV, D, L = 64, 64, 128, 4, 1, 4, 2, 16, 4
CHUNK, FLOOR, TSCALE, STEP = 4, 4, 0.1, 2
THETA, ROPE = 500000.0, {"factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 8.0,
                         "original_max_position_embeddings": 256}
NPOS, NLOGIT = 12, 512
TEXTS = [
    "The capital of France is",
    "getUserName iPhone helloWorld HTTPServer",
    "I'm sure they'll say it's 12345 or 6789.",
    " leading space and  double  spaces\nline two\ttab",
    "café naïve über \U0001F600 中文",
]

tj = json.load(open(TOK))
vocab = tj["model"]["vocab"]
added = {t["id"]: t for t in tj["added_tokens"]}
V = max(max(vocab.values()), max(added)) + 1
tokens, types = [""] * V, [1] * V
for s, i in vocab.items():
    tokens[i] = s
for i, t in added.items():
    tokens[i] = t["content"]
    types[i] = 3 if t.get("special") else 4
missing = [i for i in range(V) if tokens[i] == ""]
if missing:
    raise SystemExit("tokenizer.json leaves %d ids unnamed, first %d" % (len(missing), missing[0]))
merges = [m if isinstance(m, str) else " ".join(m) for m in tj["model"]["merges"]]
bos = next(i for i, t in added.items() if t["content"] == "<|begin_of_text|>")
eos = next(i for i, t in added.items() if t["content"] == "<|eot|>")

cfg = Llama4TextConfig(
    vocab_size=V, hidden_size=H, intermediate_size=F, intermediate_size_mlp=FM,
    num_hidden_layers=L, num_attention_heads=NH, num_key_value_heads=NKV, head_dim=D,
    num_local_experts=E, num_experts_per_tok=K, interleave_moe_layer_step=STEP,
    attention_chunk_size=CHUNK, floor_scale=FLOOR, attn_scale=TSCALE,
    attn_temperature_tuning=True, use_qk_norm=True, rms_norm_eps=1e-5,
    max_position_embeddings=4096, tie_word_embeddings=False,
    rope_parameters=dict(rope_type="llama3", rope_theta=THETA, **ROPE),
)
cfg._attn_implementation = "eager"
torch.manual_seed(20260927)
m = Llama4ForCausalLM(cfg).float().eval()
P = dict(m.named_parameters())
with torch.no_grad():
    for n, p in P.items():
        if n.endswith("layernorm.weight") or n == "model.norm.weight":
            p.copy_(torch.randn_like(p) * 0.3 + 1)
        elif "router" in n:
            p.copy_(torch.randn_like(p) * 0.5)
        else:
            p.copy_(torch.randn_like(p) * 0.2)
print("moe layers", cfg.moe_layers, "no_rope_layers", cfg.no_rope_layers, "layer_types", cfg.layer_types)


def rt(a):
    """Q8_0 round trip of a (rows, k) matrix in the GGUF orientation."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    return dequantize(quantize(a, Q8_0), Q8_0).reshape(a.shape)


if Q8:
    with torch.no_grad():
        for n, p in P.items():
            if p.dim() == 2 and "router" not in n and "norm" not in n:
                p.copy_(torch.from_numpy(rt(p.numpy())))
            elif n.endswith("experts.gate_up_proj"):          # (E, H, 2F)
                for e in range(E):
                    for cols in (slice(0, F), slice(F, 2 * F)):
                        p[e][:, cols] = torch.from_numpy(rt(p[e][:, cols].T.numpy()).T.copy())
            elif n.endswith("experts.down_proj"):             # (E, F, H)
                for e in range(E):
                    p[e] = torch.from_numpy(rt(p[e].T.numpy()).T.copy())


def rope_factors():
    """llama.cpp's conversion/llama.py generate_extra_tensors, verbatim."""
    freqs = 1.0 / (THETA ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
    factor, lo, hi, old = ROPE["factor"], ROPE["low_freq_factor"], ROPE["high_freq_factor"], \
        ROPE["original_max_position_embeddings"]
    lo_wl, hi_wl = old / lo, old / hi
    out = []
    for f in freqs:
        wl = 2 * math.pi / f
        if wl < hi_wl:
            out.append(1)
        elif wl > lo_wl:
            out.append(factor)
        else:
            s = (old / wl - lo) / (hi - lo)
            out.append(1 / ((1 - s) / factor + s))
    return np.array(out, dtype=np.float32)


rf = rope_factors()
print("rope factors", rf)
T = []
with torch.no_grad():
    np32 = lambda t: t.detach().float().numpy().copy()
    T.append(("token_embd.weight", np32(P["model.embed_tokens.weight"])))
    T.append(("output.weight", np32(P["lm_head.weight"])))
    T.append(("output_norm.weight", np32(P["model.norm.weight"])))
    T.append(("rope_freqs.weight", rf))
    for i in range(L):
        g = lambda s: P["model.layers.%d.%s" % (i, s)]
        b = "blk.%d." % i
        T += [
            (b + "attn_norm.weight", np32(g("input_layernorm.weight"))),
            (b + "ffn_norm.weight", np32(g("post_attention_layernorm.weight"))),
            (b + "attn_q.weight", np32(g("self_attn.q_proj.weight"))),
            (b + "attn_k.weight", np32(g("self_attn.k_proj.weight"))),
            (b + "attn_v.weight", np32(g("self_attn.v_proj.weight"))),
            (b + "attn_output.weight", np32(g("self_attn.o_proj.weight"))),
        ]
        if i in cfg.moe_layers:
            gu = g("feed_forward.experts.gate_up_proj")   # (E, H, 2F)
            dn = g("feed_forward.experts.down_proj")      # (E, F, H)
            T += [
                (b + "ffn_gate_inp.weight", np32(g("feed_forward.router.weight"))),
                (b + "ffn_gate_exps.weight", np32(gu[:, :, :F].transpose(1, 2))),
                (b + "ffn_up_exps.weight", np32(gu[:, :, F:].transpose(1, 2))),
                (b + "ffn_down_exps.weight", np32(dn.transpose(1, 2))),
                (b + "ffn_gate_shexp.weight", np32(g("feed_forward.shared_expert.gate_proj.weight"))),
                (b + "ffn_up_shexp.weight", np32(g("feed_forward.shared_expert.up_proj.weight"))),
                (b + "ffn_down_shexp.weight", np32(g("feed_forward.shared_expert.down_proj.weight"))),
            ]
        else:
            T += [
                (b + "ffn_gate.weight", np32(g("feed_forward.gate_proj.weight"))),
                (b + "ffn_up.weight", np32(g("feed_forward.up_proj.weight"))),
                (b + "ffn_down.weight", np32(g("feed_forward.down_proj.weight"))),
            ]

w = GGUFWriter(DST, "llama4")
w.add_block_count(L)
w.add_context_length(4096)
w.add_embedding_length(H)
w.add_feed_forward_length(FM)
w.add_head_count(NH)
w.add_head_count_kv(NKV)
w.add_key_length(D)
w.add_value_length(D)
w.add_rope_dimension_count(D)
w.add_rope_freq_base(THETA)
w.add_layer_norm_rms_eps(cfg.rms_norm_eps)
w.add_expert_count(E)
w.add_expert_used_count(K)
w.add_expert_feed_forward_length(F)
w.add_uint32("llama4.interleave_moe_layer_step", STEP)
w.add_uint32("llama4.attention.chunk_size", CHUNK)
w.add_uint32("llama4.attention.temperature_length", FLOOR)
w.add_float32("llama4.attention.temperature_scale", TSCALE)
w.add_vocab_size(V)
w.add_tokenizer_model("gpt2")
w.add_tokenizer_pre("llama4")
w.add_token_list(tokens)
w.add_token_types(types)
w.add_token_merges(merges)
w.add_bos_token_id(bos)
w.add_eos_token_id(eos)
w.add_add_bos_token(True)
for name, arr in T:
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if Q8 and arr.ndim >= 2 and "gate_inp" not in name:
        # Quantizing the already round-tripped values reproduces the same
        # bytes; checked rather than assumed.
        rows = arr.reshape(-1, arr.shape[-1])
        q = quantize(rows, Q8_0)
        if not np.array_equal(dequantize(q, Q8_0).reshape(rows.shape), rows):
            raise SystemExit(name + ": Q8_0 is not idempotent on its own output")
        w.add_tensor(name, q.reshape(arr.shape[:-1] + (-1,)), raw_dtype=Q8_0)
    else:
        w.add_tensor(name, arr)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()

ids = [(i * 7919 + 13) % 199000 for i in range(NPOS)]
with torch.no_grad():
    lg = m(torch.tensor([ids])).logits[0]
rows = [{"argmax": int(r.argmax()), "max": float(r.max()), "lse": float(torch.logsumexp(r, 0)),
         "head": [round(float(x), 7) for x in r[:NLOGIT]]} for r in lg]
tk = Tokenizer.from_file(TOK)
texts = [{"text": t, "ids": tk.encode(t, add_special_tokens=False).ids} for t in TEXTS]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump({"ids": ids, "vocab": V, "bos": bos, "pos": rows, "texts": texts}, open(OUT, "w"))
print("wrote", DST, "argmax", [r["argmax"] for r in rows])
for t in texts:
    print(repr(t["text"]), t["ids"])
