#!/usr/bin/env python3
"""Build the Kimi-VL fixture and record Moonshot's own answers.

    $JITLLM_HF_PY scripts/kimivlgold.py

    JITLLM_MODELS, LCPP_SRC   as scripts/refenv.py reads them
    KVL_SRC   a Kimi-VL checkpoint directory: its code, tokenizer and processor
              (config and code files; no weights are read); default
              $JITLLM_MODELS/kimivl/hf/Kimi-VL-A3B-Instruct
    KVL_HF    where the fixture's own checkpoint is written; default
              $JITLLM_MODELS/kimivl/fixtures

-> $JITLLM_MODELS/synth-kimivl.gguf, mmproj-synth-kimivl.gguf and
   testdata/golden/kimivl/synth-kimivl.json

THE ORACLE IS MOONSHOT'S OWN modeling_kimi_vl.py (transformers ships no Kimi-VL
class), loaded with three shims for names transformers 5 dropped, and both
GGUFs are written by llama.cpp's convert_hf_to_gguf.py. The fixture is
KimiVLForConditionalGeneration at toy size with every parameter randomised:
MoonViT (LayerNorm, GELU-tanh MLP with biases, biased fused q|k|v, a learned
8x8 table resampled BICUBIC to the picture's 6x10 patch grid, and the 2-D
rotary in complex pairs: pair 2j turns by the column, 2j+1 by the row, both
at frequency j), then the projector (a LayerNorm on each patch, the 2x2
shuffle, linear, GELU, linear) in front of DeepSeek-V3's text model.

The golden carries the picture, its patches' grid, the tower's rows at f32,
at the converter's Q8_0 and at the engine's int8 activations, and the logits
at the prompt's last row and every continuation row.
"""
import copy
import json
import os
import shutil
import subprocess
import sys
import types
import importlib

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qwenvlgold as q  # noqa: E402  (its picture, quantizers and q8 simulation)
import refenv  # noqa: E402  (where things are: environment variables only)

MODELS = refenv.models()
SRC = refenv.under("KVL_SRC", "kimivl", "hf", "Kimi-VL-A3B-Instruct")
HFDIR = refenv.under("KVL_HF", "kimivl", "fixtures")
LCPP = refenv.lcpp_src()
# Beside the Qwen-VL family's goldens, whose shape it shares.
OUT = os.path.join(HERE, "..", "testdata", "golden", "qwenvl")
# The processor's resize on a picture past a lowered patch limit: 11x17
# patches under 64 scale to 140x93 pixels and pad to 140x112.
PREP_LIMIT = 64
NAME = "synth-kimivl"
NLOGIT = 512


def remote():
    # Moonshot's code was written against transformers 4.50; three names it
    # imports are gone from 5, none of which the forward pass here reaches.
    import transformers.utils.import_utils as iu
    import transformers.pytorch_utils as pu
    import transformers.activations as act
    if not hasattr(iu, "is_torch_fx_available"):
        iu.is_torch_fx_available = lambda: False
    if not hasattr(pu, "is_torch_greater_or_equal_than_1_13"):
        pu.is_torch_greater_or_equal_than_1_13 = True
    if not hasattr(act, "PytorchGELUTanh"):
        class PytorchGELUTanh(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.gelu(x, approximate="tanh")
        act.PytorchGELUTanh = PytorchGELUTanh
    pkg = types.ModuleType("kvl")
    pkg.__path__ = [SRC]
    sys.modules["kvl"] = pkg
    mod = importlib.import_module("kvl.modeling_kimi_vl")
    # And transformers 5 hands tie_weights a keyword the old override lacks.
    tie = mod.KimiVLForConditionalGeneration.tie_weights
    mod.KimiVLForConditionalGeneration.tie_weights = lambda self, *a, **k: tie(self)
    return mod, importlib.import_module("kvl.configuration_kimi_vl")


def build():
    mod, cmod = remote()
    vision = cmod.MoonViTConfig(patch_size=14, init_pos_emb_height=8, init_pos_emb_width=8,
                                num_attention_heads=4, num_hidden_layers=3, hidden_size=64,
                                intermediate_size=100, merge_kernel_size=[2, 2])
    text = cmod.DeepseekV3Config(
        vocab_size=163840, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4, n_shared_experts=1,
        n_routed_experts=4, num_experts_per_tok=2, routed_scaling_factor=2.446, kv_lora_rank=32,
        q_lora_rank=None, qk_rope_head_dim=16, v_head_dim=16, qk_nope_head_dim=16,
        topk_method="noaux_tc", n_group=1, topk_group=1, first_k_dense_replace=1,
        norm_topk_prob=True, scoring_func="sigmoid", rms_norm_eps=1e-5, rope_theta=800000.0,
        max_position_embeddings=4096, bos_token_id=163584, eos_token_id=163585, pad_token_id=163839,
        tie_word_embeddings=False)
    cfg = cmod.KimiVLConfig(vision_config=vision, text_config=text, media_placeholder_token_id=163605)
    # transformers 5 fills rope_scaling with its own default dict, which
    # Moonshot's code reads as a scaling to apply; the checkpoint states none.
    cfg.text_config.rope_scaling = None
    cfg._attn_implementation = "eager"
    cfg.vision_config._attn_implementation = "eager"
    cfg.text_config._attn_implementation = "eager"
    torch.manual_seed(20261004)
    m = mod.KimiVLForConditionalGeneration(cfg).float().eval()
    q.perturb(m)
    with torch.no_grad():
        # The table and the attention at the Qwen fixtures' scales (perturb's
        # notes), which perturb keys on their names.
        # The text model's matrices at half perturb's 0.2, as GLM's fixtures
        # take them: a toy model at 0.2 amplifies the tower's int8 past its
        # own argmax margins.
        for n, p in m.named_parameters():
            mod = n.split(".")[-2]
            if n.endswith("pos_emb.weight"):
                p.copy_(torch.randn_like(p) * 4)
            elif ".wqkv." in n:
                p.copy_(torch.randn_like(p) * 0.3)
            elif n.startswith("language_model") and n.endswith("weight") and "norm" not in mod:
                p.mul_(0.5)
        for n, b in m.named_buffers():
            if n.endswith("e_score_correction_bias"):
                b.copy_(torch.randn_like(b) * 0.3)
    d = os.path.join(HFDIR, NAME)
    shutil.rmtree(d, ignore_errors=True)
    # Written by hand: transformers 5's save_pretrained reads the old code's
    # tied-weight list as a mapping.
    from safetensors.torch import save_file
    os.makedirs(d)
    save_file({k: v.contiguous() for k, v in m.state_dict().items()}, os.path.join(d, "model.safetensors"),
              metadata={"format": "pt"})
    cfg.save_pretrained(d)
    for f in os.listdir(SRC):
        if f.endswith(".py") or f in ("tiktoken.model", "tokenizer_config.json", "chat_template.jinja",
                                      "preprocessor_config.json", "generation_config.json"):
            shutil.copy(os.path.join(SRC, f), d)
    c = json.load(open(os.path.join(d, "config.json")))
    c["auto_map"] = json.load(open(os.path.join(SRC, "config.json")))["auto_map"]
    c["architectures"] = ["KimiVLForConditionalGeneration"]
    json.dump(c, open(os.path.join(d, "config.json"), "w"), indent=2)
    for kind, dst in (("", NAME + ".gguf"), ("--mmproj", "mmproj-" + NAME + ".gguf")):
        args = [sys.executable, os.path.join(LCPP, "convert_hf_to_gguf.py"), d, "--outtype", "f32",
                "--outfile", os.path.join(MODELS, dst)]
        if kind:
            args.append(kind)
        subprocess.run(args, check=True)
    return m, cfg


def processor(**kw):
    # Moonshot's own image processor (pad_input, mean and std 0.5), as the
    # checkpoint configures it.
    proc = importlib.import_module("kvl.image_processing_kimi_vl")
    pc = json.load(open(os.path.join(SRC, "preprocessor_config.json")))
    pc.pop("auto_map", None)
    pc.update(kw)
    return proc.KimiVLImageProcessor(**pc)


def pixels(img, **kw):
    from PIL import Image
    out = processor(**kw).preprocess([Image.fromarray(img)], return_tensors="pt")
    return out["pixel_values"], out["image_grid_hws"]


def unpatch(pv, gh, gw, patch):
    # pixel_values back to [y][x][channel]: raster patches of (c, py, px).
    return pv.reshape(gh, gw, 3, patch, patch).permute(0, 3, 1, 4, 2).reshape(gh * patch, gw * patch, 3)


def features(m, pv, hw):
    with torch.no_grad():
        return m._extract_image_features(pv, hw)


def quantize(m):
    # The tower's and projector's matrices at the converter's Q8_0, the patch
    # embedding flattened as convert.reshapePatchEmbd stores it.
    with torch.no_grad():
        for n, p in m.named_parameters():
            if not (n.startswith("vision_tower") or n.startswith("multi_modal_projector")):
                continue
            if not n.endswith("weight") or "pos_emb" in n or p.ndim < 2:
                continue
            if p.ndim == 4:
                o = p.shape[0]
                p.copy_(q.q8rows(p.reshape(o, -1)).reshape(p.shape))
            else:
                p.copy_(q.q8rows(p))


def features_a8(m, pv, hw):
    hs = [mod_.register_forward_pre_hook(lambda mod_, inp: (q.qact(inp[0]),))
          for n, mod_ in m.named_modules()
          if (n.startswith("vision_tower") or n.startswith("multi_modal_projector"))
          and isinstance(mod_, torch.nn.Linear)]
    try:
        return features(m, pv, hw)
    finally:
        for h in hs:
            h.remove()


def main():
    m, cfg = build()
    img_h, img_w = 84, 140
    img = q.picture(img_h, img_w)
    pv, hw = pixels(img)
    gh, gw = [int(v) for v in hw[0]]
    feats = features(m, pv, hw)
    mq = copy.deepcopy(m)
    quantize(mq)
    feats_q8 = features(mq, pv, hw)
    feats_a8 = features_a8(mq, pv, hw)
    nimg = gh * gw // 4
    # Kimi-VL's chat template around one picture, through its own tokenizer.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(HFDIR, NAME), trust_remote_code=True)
    enc = lambda t: tok.encode(t, add_special_tokens=False, allow_special_tokens=True)
    text = tok.apply_chat_template([{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": "What is in the picture?"}]}],
        add_generation_prompt=True, tokenize=False)
    before, after = text.split("<|media_pad|>")
    pre, post = enc(before), enc(after)
    cont = enc("A disc on a gradient.")
    ids = pre + [cfg.media_placeholder_token_id] * nimg + post + cont
    with torch.no_grad():
        out = m(input_ids=torch.tensor([ids]), pixel_values=pv, image_grid_hws=hw, use_cache=False)
    lg = out.logits[0]
    first = len(pre) + nimg + len(post) - 1
    rows = [{"argmax": int(r.argmax()), "max": float(r.max()),
             "head": [round(float(x), 7) for x in r[:NLOGIT]]} for r in lg[first:]]
    g = {"pre": pre, "post": post, "cont": cont,
         "image": {"h": img_h, "w": img_w, "rgb": q.b64(img), "grid": [1, gh // 2, gw // 2],
                   "embd": [round(float(x), 8) for x in feats.reshape(-1)],
                   "embd_q8": [round(float(x), 8) for x in feats_q8.reshape(-1)],
                   "embd_a8": [round(float(x), 8) for x in feats_a8.reshape(-1)]},
         "logits": rows}
    pimg = q.texture(q.PREP_H, q.PREP_W)
    ppv, phw = pixels(pimg, in_token_limit=PREP_LIMIT)
    ph, pw = [int(v) for v in phw[0]]
    g["prep"] = {"h": q.PREP_H, "w": q.PREP_W, "max": PREP_LIMIT, "rgb": q.b64(pimg),
                 "out_h": ph * 14, "out_w": pw * 14,
                 "px": [round(float(x), 5) for x in unpatch(ppv, ph, pw, 14).reshape(-1)]}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, NAME + ".json"), "w") as f:
        json.dump(g, f)
    print(f"{NAME}: {len(ids)} rows, image {gh // 2}x{gw // 2} merged, {len(rows)} logit rows")


if __name__ == "__main__":
    main()
