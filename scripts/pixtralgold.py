#!/usr/bin/env python3
"""Build the Pixtral / Mistral 3 vision fixture and record transformers' answers.

    $JITLLM_HF_PY scripts/pixtralgold.py
    JITLLM_MODELS=/path/to/models   (where the GGUFs go)
    PIX_TOK=<a Ministral 3 checkpoint dir>         (its tokenizer)

-> $JITLLM_MODELS/synth-pixtral.gguf, $JITLLM_MODELS/mmproj-synth-pixtral.gguf
   and testdata/golden/pixtral/synth-pixtral.json

THE ORACLE IS transformers' OWN Mistral3ForConditionalGeneration (Ministral 3's
text model behind Pixtral's tower and Mistral 3's patch merger), every
parameter randomised, AND BOTH GGUFs ARE WRITTEN BY llama.cpp's OWN
convert_hf_to_gguf.py -- the text model plainly and the tower with --mmproj,
which permutes q and k for interleaved rotary pairs and copies [IMG_BREAK]'s
embedding out of the text table -- so jitllm converts the bytes a real
checkpoint's conversion puts in front of it.

The picture is NOT square and is a whole number of merge units on both sides,
so the processor does not resize it and the merged grid is 5 x 7: a row turned
by the column's frequencies, a merge that reads the unfold's channel-major
order as patch-major, or a break written after the wrong rows each move the
answer. The golden carries the picture, the merged rows transformers' tower
and projector emit (at f32, at the converter's Q8_0 and at the engine's int8
activations), the prompt as transformers' processor lays it out, and the
logits at the prompt's last row and every continuation row. A second picture
the processor must resize records its resize.
"""
import copy
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import torch

import refenv  # where things are: environment variables only (docs/testing.md)

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = refenv.models()
TOK = refenv.under("PIX_TOK", "vlmb", "hf", "Ministral-3-3B")
HFDIR = refenv.under("PIX_HF", "vlmb", "fixtures")
LCPP = refenv.need("LCPP_SRC", "a llama.cpp source checkout")
PY = sys.executable
OUT = os.path.join(HERE, "..", "testdata", "golden", "pixtral")
NAME = "synth-pixtral"
NLOGIT = 512
IMG_H, IMG_W = 140, 196  # 10 x 14 patches, 5 x 7 merged
LONGEST = 224            # the tower's image_size: the processor's longest edge
PREP_H, PREP_W = 170, 300  # resized: 300 is over 224


def config():
    from transformers import Mistral3Config
    # Ministral 3's text model at toy size (its YaRN and temperature as
    # synth-hf-ministral3 has them); Pixtral's tower three blocks deep with
    # an MLP width (100) that is not a whole quantization block, as the
    # converter must pad; a 2x2 merge; biased projector matrices, which the
    # real checkpoint does not carry and llama.cpp reads when present.
    return Mistral3Config(
        text_config=dict(model_type="ministral3", vocab_size=131072, hidden_size=64,
                         intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-5,
                         max_position_embeddings=4096, tie_word_embeddings=True,
                         rope_parameters={"rope_type": "yarn", "rope_theta": 1000000.0, "factor": 16.0,
                                          "original_max_position_embeddings": 4, "beta_fast": 32.0,
                                          "beta_slow": 1.0, "mscale": 1.0, "mscale_all_dim": 0.5,
                                          "llama_4_scaling_beta": 0.5}),
        vision_config=dict(model_type="pixtral", hidden_size=64, intermediate_size=100,
                           num_hidden_layers=3, num_attention_heads=4, head_dim=16, patch_size=14,
                           image_size=LONGEST, hidden_act="silu",
                           rope_parameters={"rope_type": "default", "rope_theta": 10000.0}),
        spatial_merge_size=2, multimodal_projector_bias=True, projector_hidden_act="gelu",
        image_token_index=10, tie_word_embeddings=True)


def perturb(m):
    # Norm weights away from one and every bias away from zero, so a missing
    # norm or bias moves the answer (transformers initialises both to the
    # identity).
    with torch.no_grad():
        for n, p in m.named_parameters():
            mod = n.split(".")[-2] if "." in n else ""
            if n.endswith("weight") and ("norm" in mod or mod.startswith("ln")):
                p.copy_(torch.randn_like(p) * 0.3 + 1)
            elif n.endswith("bias"):
                p.copy_(torch.randn_like(p) * 0.3)
            elif "vision_tower" in n and ("q_proj" in n or "k_proj" in n):
                # Sharper tower attention, so a wrong rotary or a wrong key
                # set moves the output by far more than the int8 activations
                # do. At 0.3 the int8 alone moves transformers' rows by 2e-2
                # and the engine sits 2e-3 from its own arithmetic; at 0.22
                # 6e-3 and 8e-4, with every violation still at 0.4 and up.
                p.copy_(torch.randn_like(p) * 0.22)
            elif "linear_" in n:
                # The projector's two matrices at about the text model's
                # scale: at 0.2 the rows come out at an RMS of 6.3 against an
                # embedding row's 0.17, and the toy text model amplifies any
                # difference in them a hundredfold.
                p.copy_(torch.randn_like(p) * 0.03)
            else:
                p.copy_(torch.randn_like(p) * 0.2)


def picture(h, w):
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    r = 255 * (0.5 + 0.5 * np.sin(x / 9.0))
    g = 255 * (y / h)
    b = 255 * (((x - 70) ** 2 + (y - 40) ** 2) < 30 ** 2)
    return np.clip(np.stack([r, g, b], -1), 0, 255).astype(np.uint8)


def texture(h, w):
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    r = 255 * (0.5 + 0.5 * np.sin(x / 1.7))
    g = 255 * (0.5 + 0.5 * np.cos((x + y) / 2.3))
    b = 255 * (((x - 120) ** 2 + (y - 80) ** 2) < 50 ** 2)
    return np.clip(np.stack([r, g, b], -1), 0, 255).astype(np.uint8)


def q8rows(w):
    # A matrix as the converter stores it: Q8_0 along k (d = amax/127 as f16),
    # k zero-padded to a whole block first, cut back after.
    k = w.shape[-1]
    kp = (k + 31) // 32 * 32
    x = torch.nn.functional.pad(w.reshape(-1, k), (0, kp - k)).reshape(-1, 32)
    d = x.abs().amax(1, keepdim=True) / 127
    idd = torch.where(d == 0, torch.zeros_like(d), 1 / d)
    c = torch.round(x * idd)
    y = (c * d.half().float()).reshape(-1, kp)[:, :k]
    return y.reshape(w.shape)


def qact(x):
    # An activation as the engine's packed matmul reads it: int8 per block of
    # 32 along k, d = amax/127.
    k = x.shape[-1]
    kp = (k + 31) // 32 * 32
    y = torch.nn.functional.pad(x.reshape(-1, k), (0, kp - k)).reshape(-1, 32)
    d = y.abs().amax(1, keepdim=True) / 127
    idd = torch.where(d == 0, torch.zeros_like(d), 1 / d)
    return (torch.round(y * idd) * d).reshape(-1, kp)[:, :k].reshape(x.shape)


def to_patch_major(x, d, s2):
    # The unfold's channel-major columns (c*s2 + p) to the engine's shuffle
    # order (p*d + c): convert.permuteMerger.
    return x.reshape(*x.shape[:-1], d, s2).transpose(-1, -2).reshape(*x.shape[:-1], d * s2)


def to_channel_major(x, d, s2):
    return x.reshape(*x.shape[:-1], s2, d).transpose(-1, -2).reshape(*x.shape[:-1], d * s2)


def quantize_vision(m, d, s2):
    # Every matrix the engine's tower and head multiply by at the converter's
    # Q8_0: the patch conv as its flattened matrix, the merger in the column
    # order the converter stores it.
    with torch.no_grad():
        for n, p in m.named_parameters():
            if not n.endswith("weight") or not ("vision_tower" in n or "multi_modal_projector" in n):
                continue
            if "patch_conv" in n:
                o = p.shape[0]
                p.copy_(q8rows(p.reshape(o, -1)).reshape(p.shape))
            elif "merging_layer" in n:
                p.copy_(to_channel_major(q8rows(to_patch_major(p, d, s2)), d, s2))
            elif p.ndim == 2:
                p.copy_(q8rows(p))


def qpatches(x, p):
    # The patch conv's input as the engine's patch matmul reads it: each
    # patch flattened channel-planar (c, ky, kx), the conv kernel's order,
    # at int8 per block of 32 of that row.
    b, c, h, w = x.shape
    t = x.reshape(b, c, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5).reshape(-1, c * p * p)
    t = qact(t).reshape(b, h // p, w // p, c, p, p).permute(0, 3, 1, 4, 2, 5)
    return t.reshape(b, c, h, w)


def features(m, pv, sizes, a8=False, d=0, s2=0):
    # a8 is the arithmetic the engine's tower and head are made of: every
    # matmul's input at int8 per block of 32 (the patch conv's per patch, the
    # merger's in the column order the converter stores it), and the
    # projector's GELU in its tanh form, which every projector here runs
    # (llama.cpp's ggml_gelu; the reference's is erf, and embd is held to it).
    hs = []
    act = m.model.multi_modal_projector.act
    if a8:
        from transformers.activations import ACT2FN
        m.model.multi_modal_projector.act = ACT2FN["gelu_pytorch_tanh"]
        p = m.config.vision_config.patch_size
        for n, mod in m.named_modules():
            if "vision_tower" not in n and "multi_modal_projector" not in n:
                continue
            if isinstance(mod, torch.nn.Conv2d):
                hs.append(mod.register_forward_pre_hook(lambda mod, inp: (qpatches(inp[0], p),)))
            if not isinstance(mod, torch.nn.Linear):
                continue
            if "merging_layer" in n:
                hs.append(mod.register_forward_pre_hook(
                    lambda mod, inp: (to_channel_major(qact(to_patch_major(inp[0], d, s2)), d, s2),)))
            else:
                hs.append(mod.register_forward_pre_hook(lambda mod, inp: (qact(inp[0]),)))
    try:
        with torch.no_grad():
            f = m.model.get_image_features(pv, sizes)
        f = f.pooler_output if hasattr(f, "pooler_output") else f
        if isinstance(f, (list, tuple)):
            f = torch.cat(list(f), 0)
        return f
    finally:
        m.model.multi_modal_projector.act = act
        for h in hs:
            h.remove()


def b64(img):
    import base64
    return base64.b64encode(img.reshape(-1).tobytes()).decode()


def main():
    from transformers import AutoTokenizer, Mistral3ForConditionalGeneration, PixtralImageProcessorPil
    cfg = config()
    cfg._attn_implementation = "eager"
    cfg.text_config._attn_implementation = "eager"
    cfg.vision_config._attn_implementation = "eager"
    torch.manual_seed(20261004)
    m = Mistral3ForConditionalGeneration(cfg).float().eval()
    perturb(m)
    d = os.path.join(HFDIR, NAME)
    shutil.rmtree(d, ignore_errors=True)
    m.save_pretrained(d, safe_serialization=True)
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        shutil.copy(os.path.join(TOK, f), d)
    improc = PixtralImageProcessorPil(patch_size={"height": 28, "width": 28},
                                      size={"longest_edge": LONGEST})
    improc.save_pretrained(d)
    for kind, dst in (("", NAME + ".gguf"), ("--mmproj", "mmproj-" + NAME + ".gguf")):
        args = [PY, os.path.join(LCPP, "convert_hf_to_gguf.py"), d, "--outtype", "f32",
                "--outfile", os.path.join(MODELS, dst)]
        if kind:
            args.append(kind)
        subprocess.run(args, check=True)

    tok = AutoTokenizer.from_pretrained(d)
    img = picture(IMG_H, IMG_W)
    pv = improc(images=[img], return_tensors="pt")
    sizes = pv["image_sizes"]
    assert [int(v) for v in sizes[0]] == [IMG_H, IMG_W], sizes
    mh, mw = IMG_H // 28, IMG_W // 28
    IMG, BRK, END = 10, 12, 13
    pre = [1] + tok.encode("[INST]", add_special_tokens=False)
    rows = []
    for y in range(mh):
        rows += [IMG] * mw + [BRK]
    rows[-1] = END
    post = tok.encode("What is in the picture?[/INST]", add_special_tokens=False)
    cont = tok.encode(" A disc on a gradient.", add_special_tokens=False)
    ids = pre + rows + post + cont
    t_ids = torch.tensor([ids])
    with torch.no_grad():
        out = m(input_ids=t_ids, pixel_values=pv["pixel_values"], image_sizes=sizes)
    lg = out.logits[0]
    feats = features(m, pv["pixel_values"], sizes)
    vd = cfg.vision_config.hidden_size
    s2 = cfg.spatial_merge_size ** 2
    mq = copy.deepcopy(m)
    quantize_vision(mq, vd, s2)
    feats_q8 = features(mq, pv["pixel_values"], sizes)
    feats_a8 = features(mq, pv["pixel_values"], sizes, a8=True, d=vd, s2=s2)
    brk = m.model.language_model.embed_tokens.weight[BRK].detach()
    first = len(pre) + len(rows) + len(post) - 1

    def logit_rows(lg):
        return [{"argmax": int(r.argmax()), "max": float(r.max()),
                 "head": [round(float(x), 7) for x in r[:NLOGIT]]} for r in lg[first:]]

    lrows = logit_rows(lg)
    # The same prompt with the tower's rows at the engine's arithmetic
    # (feats_a8) spliced where the image tokens are: the toy text model
    # amplifies the int8's 6e-3 on the rows to 0.2 of the logits, so the
    # whole model from the pixels is held to this, and to lrows' argmaxes.
    with torch.no_grad():
        emb = m.model.get_input_embeddings()(t_ids)
        sel = (t_ids[0] == IMG).nonzero().reshape(-1)
        emb[0, sel] = feats_a8.to(emb.dtype)
        lg8 = m(inputs_embeds=emb).logits[0]
    g = {
        "pre": pre, "post": [END] + post, "cont": cont,
        "image": {"h": IMG_H, "w": IMG_W, "rgb": b64(img), "grid": [1, mh, mw],
                  "embd": [round(float(x), 8) for x in feats.reshape(-1)],
                  "embd_q8": [round(float(x), 8) for x in feats_q8.reshape(-1)],
                  "embd_a8": [round(float(x), 8) for x in feats_a8.reshape(-1)],
                  "break": [round(float(x), 8) for x in brk]},
        "logits": lrows,
        "logits_a8": logit_rows(lg8),
    }
    # The processor's resize, on a picture it must resize.
    pimg = texture(PREP_H, PREP_W)
    ppv = improc(images=[pimg], return_tensors="pt")
    oh, ow = [int(v) for v in ppv["image_sizes"][0]]
    px = ppv["pixel_values"][0][:, :oh, :ow].permute(1, 2, 0)
    g["prep"] = {"h": PREP_H, "w": PREP_W, "longest": LONGEST, "rgb": b64(pimg),
                 "out_h": oh, "out_w": ow, "px": [round(float(x), 5) for x in px.reshape(-1)]}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, NAME + ".json"), "w") as f:
        json.dump(g, f)
    print(f"{NAME}: {len(ids)} ids, image {mh}x{mw} merged, prep {PREP_W}x{PREP_H} -> {ow}x{oh}, "
          f"{len(lrows)} logit rows")


if __name__ == "__main__":
    main()
