#!/usr/bin/env python3
"""Build the HunyuanVL fixture and record transformers' answers.

    $JITLLM_HF_PY scripts/hunyuanvlgold.py

    JITLLM_MODELS, LCPP_SRC   as scripts/refenv.py reads them
    HYVL_SRC  a HunyuanOCR checkpoint directory: its tokenizer, chat template
              and processor config (no weights are read); default
              $JITLLM_MODELS/hunyuanvl/hf/HunyuanOCR
    HYVL_HF   where the fixture's own checkpoint is written; default
              $JITLLM_MODELS/hunyuanvl/fixtures

-> $JITLLM_MODELS/synth-hunyuanvl.gguf, mmproj-synth-hunyuanvl.gguf and
   testdata/golden/qwenvl/synth-hunyuanvl.json

The fixture is transformers' own HunYuanVLForConditionalGeneration at toy
size, every parameter randomised, written to GGUF by llama.cpp's converter. Its
tower is a LayerNorm, GELU ViT with no rotary and a learned table resampled
bilinear (align_corners False) to the grid, added by ROW -- the processor hands
the patches over in merge-group order and the table and the merger both read
the rows as a raster, which is the reference's arithmetic and is what is
recorded. The merger: an RMSNorm, a 2x2 convolution, GELU, a 1x1
convolution, a newline column, a linear, the begin and end rows, an RMSNorm.
The text model is hunyuan-dense's block with XD-RoPE: four position axes
(sequence, width, height, image index) assigned per ELEMENT of the rotary --
transformers' recomposition_frequencies over cat(freq, freq) -- so a NEOX
pair's two halves turn by different axes on an image's rows.

The golden carries the Qwen-VL family's fields (image rows at f32, Q8_0 and
int8; logits at the prompt's last row and each continuation row), the prompt's
four-axis positions (pos4), and a picture the processor resizes (prep).
"""
import copy
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qwenvlgold as q  # noqa: E402  (its picture, quantizers and q8 simulation)
import refenv  # noqa: E402  (where things are: environment variables only)


OUT = os.path.join(HERE, "..", "testdata", "golden", "qwenvl")
NAME = "synth-hunyuanvl"
NLOGIT = 512
IMG_H, IMG_W = 96, 160
# The processor's resize on a picture it must resize, under bounds that keep
# the fixture small: 160x240 to a factor of 32 within [64^2, 192^2].
PREP_MIN, PREP_MAX = 64 * 64, 192 * 192
FIX_MIN, FIX_MAX = 64 * 64, 256 * 256


def build(models, src, hfdir, lcpp):
    from transformers import HunYuanVLConfig, HunYuanVLForConditionalGeneration
    # head_dim 32: 16 rotary pairs, sections [4, 4, 4, 4] -- 32 elements
    # each over cat(freq, freq), so a pair's halves take different axes.
    cfg = HunYuanVLConfig(
        text_config=dict(vocab_size=120818, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=32, rms_norm_eps=1e-5,
                         max_position_embeddings=4096, use_qk_norm=True, bos_token_id=120000,
                         eos_token_id=120020, pad_token_id=120002,
                         rope_parameters={"rope_type": "xdrope", "rope_theta": 10000.0, "alpha": 1000.0,
                                          "xdrope_section": [4, 4, 4, 4], "factor": 1.0}),
        vision_config=dict(hidden_size=64, intermediate_size=100, num_attention_heads=4, num_hidden_layers=3,
                           out_hidden_size=64, text_hidden_size=64, patch_size=16, spatial_merge_size=2,
                           max_image_size=128))
    torch.manual_seed(20261005)
    m = HunYuanVLForConditionalGeneration(cfg).float().eval()
    q.perturb(m)
    with torch.no_grad():
        for n, p in m.named_parameters():
            mod = n.split(".")[-2] if "." in n else ""
            if "position_embedding" in n:
                p.copy_(torch.randn_like(p) * 4)
            elif "patch_embedding.weight" in n:
                # Smaller than the table, so the table's resampling is
                # load-bearing (the aligned-corners violation).
                p.mul_(0.1)
            elif ("image_begin" in n or "image_end" in n or "image_newline" in n or "image_sep" in n):
                p.copy_(torch.randn_like(p))
            elif n.startswith("model.language_model") or n.startswith("lm_head"):
                if n.endswith("weight") and "norm" not in mod:
                    p.mul_(0.5)
    d = os.path.join(hfdir, NAME)
    shutil.rmtree(d, ignore_errors=True)
    m.save_pretrained(d)
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
              "preprocessor_config.json", "generation_config.json"):
        shutil.copy(os.path.join(src, f), d)
    # The processor's bounds at the fixture's scale: the tower's capacity is
    # the ceiling's patches (65536 at HunyuanOCR's own 4096^2), and the golden
    # pictures sit inside both.
    pc = json.load(open(os.path.join(d, "preprocessor_config.json")))
    pc.update(min_pixels=FIX_MIN, max_pixels=FIX_MAX, size={"shortest_edge": FIX_MIN, "longest_edge": FIX_MAX})
    json.dump(pc, open(os.path.join(d, "preprocessor_config.json"), "w"), indent=2)
    for kind, dst in (("", NAME + ".gguf"), ("--mmproj", "mmproj-" + NAME + ".gguf")):
        args = [sys.executable, os.path.join(lcpp, "convert_hf_to_gguf.py"), d, "--outtype", "f32",
                "--outfile", os.path.join(models, dst)]
        if kind:
            args.append(kind)
        subprocess.run(args, check=True)
    return m, cfg, d


def quantize(m):
    # The tower's and merger's matrices at the converter's Q8_0: the 2x2
    # convolution as the matrix over the (dy, dx, channel) row the converter
    # stores, the 1x1 one as a plain matrix.
    with torch.no_grad():
        for n, p in m.named_parameters():
            if not n.startswith("model.vision_tower") or not n.endswith("weight") or p.ndim < 2:
                continue
            if "position_embedding" in n:
                continue
            if p.ndim == 4 and p.shape[2] > 1:
                w = p.permute(0, 2, 3, 1)  # (o, ky, kx, c)
                p.copy_(q.q8rows(w.reshape(w.shape[0], -1)).reshape(w.shape).permute(0, 3, 1, 2))
            elif p.ndim == 4:
                o = p.shape[0]
                p.copy_(q.q8rows(p.reshape(o, -1)).reshape(p.shape))
            else:
                p.copy_(q.q8rows(p))


def features(m, pv, thw):
    with torch.no_grad():
        return m.model.get_image_features(pixel_values=pv, image_grid_thw=thw).pooler_output.reshape(
            -1, m.config.text_config.hidden_size)


def features_a8(m, pv, thw):
    hs = [mod.register_forward_pre_hook(lambda mod, inp: (q.qact(inp[0]),))
          for n, mod in m.named_modules()
          if n.startswith("model.vision_tower") and isinstance(mod, torch.nn.Linear)]

    def qconv(mod, inp):
        x = inp[0]
        if mod.kernel_size[0] == 1:
            y = x.permute(0, 2, 3, 1)
            return (q.qact(y).permute(0, 3, 1, 2),)
        b, c, h, w = x.shape
        k = mod.kernel_size[0]
        g = x.reshape(b, c, h // k, k, w // k, k).permute(0, 2, 4, 3, 5, 1)  # (b, gy, gx, dy, dx, c)
        gq = q.qact(g.reshape(b, h // k, w // k, -1)).reshape(g.shape)
        return (gq.permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w),)
    hs += [mod.register_forward_pre_hook(qconv) for n, mod in m.named_modules()
           if n.startswith("model.vision_tower.patch_merger") and isinstance(mod, torch.nn.Conv2d)]
    try:
        return features(m, pv, thw)
    finally:
        for h in hs:
            h.remove()


def unpatch(pv, gh, gw, merge, patch):
    # pixel_values back to [y][x][channel]: patches row-major over the grid
    # (the processor's order, not Qwen-VL's merge groups) of (channel, py, px).
    p = pv.reshape(gh // merge, merge, gw // merge, merge, 3, patch, patch)
    p = p.permute(0, 1, 5, 2, 3, 6, 4)  # (bh, mh, py, bw, mw, px, c)
    return p.reshape(gh * patch, gw * patch, 3)


def main():
    models, lcpp = refenv.models(), refenv.lcpp_src()
    src = refenv.under("HYVL_SRC", "hunyuanvl", "hf", "HunyuanOCR")
    hfdir = refenv.under("HYVL_HF", "hunyuanvl", "fixtures")
    m, cfg, d = build(models, src, hfdir, lcpp)
    from transformers import AutoProcessor
    from PIL import Image
    proc = AutoProcessor.from_pretrained(d)
    img = q.picture(IMG_H, IMG_W)
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is in the picture?"}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inp = proc(text=[text], images=[Image.fromarray(img)], return_tensors="pt", do_resize=False)
    ids = inp["input_ids"][0].tolist()
    pv, thw = inp["pixel_values"], inp["image_grid_thw"]
    _, gh, gw = [int(v) for v in thw[0]]
    tt = inp["mm_token_type_ids"][0].tolist()
    first, last = tt.index(1), len(tt) - 1 - tt[::-1].index(1)
    pre, rows, post = ids[:first], last - first + 1, ids[last + 1:]
    feats = features(m, pv, thw)
    if feats.shape[0] != rows:
        sys.exit(f"the merger emits {feats.shape[0]} rows and the prompt holds {rows}")
    mq = copy.deepcopy(m)
    quantize(mq)
    feats_q8 = features(mq, pv, thw)
    feats_a8 = features_a8(mq, pv, thw)
    cont = proc.tokenizer.encode("A disc on a gradient.", add_special_tokens=False)
    full = torch.tensor([ids + cont])
    mm = torch.tensor([tt + [0] * len(cont)])
    with torch.no_grad():
        pos, _ = m.model.get_rope_index(full, mm, thw)
        out = m(input_ids=full, pixel_values=pv, image_grid_thw=thw, mm_token_type_ids=mm, use_cache=False)
    lg = out.logits[0]
    at = len(ids) - 1
    lrows = [{"argmax": int(r.argmax()), "max": float(r.max()),
              "head": [round(float(x), 7) for x in r[:NLOGIT]]} for r in lg[at:at + len(cont)]]
    g = {"pre": pre, "post": post, "cont": cont,
         "image": {"h": IMG_H, "w": IMG_W, "rgb": q.b64(img), "grid": [1, gh // 2, gw // 2],
                   "embd": [round(float(x), 8) for x in feats.reshape(-1)],
                   "embd_q8": [round(float(x), 8) for x in feats_q8.reshape(-1)],
                   "embd_a8": [round(float(x), 8) for x in feats_a8.reshape(-1)]},
         "pos4": pos[:, 0, :].T.tolist(),
         "logits": lrows}
    pimg = q.texture(q.PREP_H, q.PREP_W)
    # The PIL processor: HunyuanOCR's own resizes with PIL.Image.resize, and
    # transformers' torchvision one differs from it in the last bits.
    from transformers.models.hunyuan_vl.image_processing_pil_hunyuan_vl import HunYuanVLImageProcessorPil
    pil = HunYuanVLImageProcessorPil.from_pretrained(d)
    pinp = pil(images=[Image.fromarray(pimg)], return_tensors="pt",
               size={"shortest_edge": PREP_MIN, "longest_edge": PREP_MAX})
    _, ph, pw = [int(v) for v in pinp["image_grid_thw"][0]]
    px = unpatch(pinp["pixel_values"], ph, pw, 2, 16)
    g["prep"] = {"h": q.PREP_H, "w": q.PREP_W, "min": PREP_MIN, "max": PREP_MAX, "rgb": q.b64(pimg),
                 "out_h": ph * 16, "out_w": pw * 16, "px": [round(float(x), 5) for x in px.reshape(-1)]}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, NAME + ".json"), "w") as f:
        json.dump(g, f)
    a, b = feats_a8.reshape(-1).double(), feats.reshape(-1).double()
    print(f"{NAME}: {len(ids) + len(cont)} rows, grid {gh}x{gw} patches, {rows} image rows; "
          f"int8 moves the tower {float(((a - b) ** 2).sum() / (b ** 2).sum()):.3e}")


if __name__ == "__main__":
    main()
