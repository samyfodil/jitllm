#!/usr/bin/env python3
"""Build the Gemma 3n vision fixture and record transformers' answers.

    $JITLLM_HF_PY scripts/gemma3nvisiongold.py
    JITLLM_MODELS=/path/to/models   (where the GGUFs go)

-> $JITLLM_MODELS/synth-gemma3nv.gguf, $JITLLM_MODELS/mmproj-synth-gemma3nv.gguf,
   $JITLLM_MODELS/synth-gemma3nv.gold/ (the picture's pixels and the tower's
   intermediates, too large for the repository)
   and testdata/golden/gemma3nv/synth-gemma3nv.json

THE ORACLE IS transformers' OWN Gemma3nForConditionalGeneration -- timm's
MobileNetV5 encoder (mobilenetv5_300m_enc, every one of its 84 blocks: edge
residuals, universal inverted residuals, multi-query attention with a
stride-2 key/value downsample, the multi-scale fusion adapter) at a quarter of
its channels, Gemma 3n's embedder, and a Gemma 3n text model -- every
parameter randomised, AND BOTH GGUFs ARE WRITTEN BY llama.cpp's OWN
convert_hf_to_gguf.py, the text model plainly and the tower with --mmproj.

The picture is 200x136; the processor (SiglipImageProcessorFast) resizes it
to 768x768. The golden carries the rows transformers' head emits at f32, at
the converter's Q8_0 and at the engine's int8 activations, the prompt, and the
logits at the prompt's last row and every continuation row.
"""
import copy
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

import refenv  # where things are: environment variables only (docs/testing.md)

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = refenv.models()
TOKDIR = refenv.under("CV_TOK", "tok")
HFDIR = refenv.under("G3NV_HF", "vlmb", "fixtures")
LCPP = refenv.need("LCPP_SRC", "a llama.cpp source checkout")
PY = sys.executable
OUT = os.path.join(HERE, "..", "testdata", "golden", "gemma3nv")
NAME = "synth-gemma3nv"
NLOGIT = 512
IMG_H, IMG_W = 136, 200
S, Fu = "sliding_attention", "full_attention"


def config():
    from transformers import Gemma3nAudioConfig, Gemma3nConfig, Gemma3nTextConfig, Gemma3nVisionConfig
    n = 4
    text = Gemma3nTextConfig(
        vocab_size=262400, vocab_size_per_layer_input=262144, hidden_size=64,
        intermediate_size=[128] * n, num_hidden_layers=n, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, layer_types=[S, S, S, Fu],
        sliding_window=512, final_logit_softcapping=30.0, hidden_size_per_layer_input=32,
        altup_num_inputs=4, altup_active_idx=0, altup_correct_scale=True, laurel_rank=32,
        num_kv_shared_layers=0, activation_sparsity_pattern=[0.0] * n,
        rms_norm_eps=1e-6, max_position_embeddings=4096, tie_word_embeddings=True,
        rope_parameters={S: {"rope_type": "default", "rope_theta": 10000.0},
                         Fu: {"rope_type": "default", "rope_theta": 1000000.0}},
        pad_token_id=0, eos_token_id=1, bos_token_id=2)
    vis = Gemma3nVisionConfig(model_args={"channel_multiplier": 0.25})
    aud = Gemma3nAudioConfig(hidden_size=64, conf_num_hidden_layers=1, conf_num_attention_heads=2,
                             sscp_conv_channel_size=[8, 8], conf_conv_kernel_size=5)
    return Gemma3nConfig(text_config=text, vision_config=vis, audio_config=aud)


def perturb(m):
    with torch.no_grad():
        for n, p in m.named_parameters():
            last = n.split(".")[-2] if "." in n else ""
            if "vision_tower" in n:
                if p.ndim == 1 and ("bn" in n or "norm" in n):
                    p.copy_(torch.randn_like(p) * 0.3 + 1)
                elif n.endswith("layer_scale.gamma"):
                    p.copy_(torch.rand_like(p) * 0.8 + 0.4)
                elif n.endswith("bias"):
                    p.copy_(torch.randn_like(p) * 0.3)
                elif p.ndim == 4 and p.shape[1] == 1:
                    # a depthwise filter: its taps sum to about one
                    p.copy_(torch.randn_like(p) * 0.3 + 1.0 / (p.shape[2] * p.shape[3]))
                else:
                    fan = p[0].numel()
                    p.copy_(torch.randn_like(p) * (1.2 / fan ** 0.5))
            elif "embed_vision" in n:
                if "norm" in last:
                    p.copy_(torch.randn_like(p) * 0.3 + 1)
                else:
                    p.copy_(torch.randn_like(p) * 0.03)
            elif "audio" in n:
                continue
            elif n.endswith("weight") and "norm" in last:
                p.copy_(torch.randn_like(p) * 0.3 + 1)
            elif n.endswith("correct_output_scale"):
                p.copy_(torch.randn_like(p) * 0.3 + 1)
            elif "correction_coefs" in n or "prediction_coefs" in n:
                p.copy_(torch.randn_like(p) * 0.3)
            else:
                p.copy_(torch.randn_like(p) * 0.2)


def picture(h, w):
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    r = 255 * (0.5 + 0.5 * np.sin(x / 9.0))
    g = 255 * (y / h)
    b = 255 * (((x - 70) ** 2 + (y - 40) ** 2) < 30 ** 2)
    return np.clip(np.stack([r, g, b], -1), 0, 255).astype(np.uint8)


def q8rows(w):
    k = w.shape[-1]
    kp = (k + 31) // 32 * 32
    x = F.pad(w.reshape(-1, k), (0, kp - k)).reshape(-1, 32)
    d = x.abs().amax(1, keepdim=True) / 127
    idd = torch.where(d == 0, torch.zeros_like(d), 1 / d)
    y = (torch.round(x * idd) * d.half().float()).reshape(-1, kp)[:, :k]
    return y.reshape(w.shape)


def qact(x):
    k = x.shape[-1]
    kp = (k + 31) // 32 * 32
    y = F.pad(x.reshape(-1, k), (0, kp - k)).reshape(-1, 32)
    d = y.abs().amax(1, keepdim=True) / 127
    idd = torch.where(d == 0, torch.zeros_like(d), 1 / d)
    return (torch.round(y * idd) * d).reshape(-1, kp)[:, :k].reshape(x.shape)


def conv_matrix(w):
    """A conv weight [out, in, kh, kw] as the engine's matrix: rows out, k in
    tap-major order (ky, kx, c)."""
    return w.permute(0, 2, 3, 1).reshape(w.shape[0], -1)


def quantize_vision(m):
    """The converter's Q8_0 on every tower matrix the engine multiplies: the
    pointwise and full convolutions (in the engine's tap-major k order), the
    attention projections, the fusion adapter's two and the embedder's
    projection. The depthwise filters and the norms stay float."""
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "vision_tower" in n and p.ndim == 4 and p.shape[1] != 1:
                k = conv_matrix(p)
                q = q8rows(k).reshape(p.shape[0], p.shape[2], p.shape[3], p.shape[1]).permute(0, 3, 1, 2)
                p.copy_(q)
            elif "embed_vision" in n and n.endswith("embedding_projection.weight"):
                p.copy_(q8rows(p))


class A8:
    """The engine's int8 activations on every matrix's input: a forward
    pre-hook on each full convolution (the input unfolded tap-major per output
    position, as the engine's im2col lays it out) and each linear."""

    def __init__(self, m):
        self.hs = []
        for n, mod in m.named_modules():
            if "vision_tower" in n and isinstance(mod, torch.nn.Conv2d) and mod.groups == 1:
                self.hs.append(mod.register_forward_hook(self.conv))
            elif "embed_vision" in n and isinstance(mod, torch.nn.Linear):
                self.hs.append(mod.register_forward_pre_hook(lambda mod, inp: (qact(inp[0]),)))

    @staticmethod
    def conv(mod, inp, out):
        # Recompute the convolution on int8 activations: unfold the input as
        # the engine does (each output position's taps, tap-major, channels
        # inner), quantize each row in blocks of 32, and multiply.
        x = inp[0]
        kh, kw = mod.kernel_size
        sh, sw = mod.stride
        if hasattr(mod, "pad_input") or type(mod).__name__ == "Conv2dSame":
            from timm.layers.padding import pad_same
            x = pad_same(x, mod.kernel_size, mod.stride, mod.dilation)
            pad = 0
        else:
            pad = mod.padding
        cols = F.unfold(x, (kh, kw), padding=pad, stride=(sh, sw))  # [b, c*kh*kw, L]
        b, ck, L = cols.shape
        c = x.shape[1]
        cols = cols.reshape(b, c, kh * kw, L).permute(0, 3, 2, 1).reshape(b, L, kh * kw * c)
        cols = qact(cols)
        w = conv_matrix(mod.weight)
        y = cols @ w.t()
        if mod.bias is not None:
            y = y + mod.bias
        return y.permute(0, 2, 1).reshape(out.shape)

    def remove(self):
        for h in self.hs:
            h.remove()


def features(m, pv, a8=False, taps=None):
    hook = A8(m) if a8 else None
    hs = []
    if taps is not None:
        tm = m.model.vision_tower.timm_model
        hs.append(tm.conv_stem.register_forward_hook(lambda mod, i, o: taps.__setitem__("stem", o.detach())))
        k = 0
        for si, stage in enumerate(tm.blocks):
            for bi, blk in enumerate(stage):
                hs.append(blk.register_forward_hook(
                    lambda mod, i, o, k=k: taps.__setitem__("blk%d" % k, o.detach())))
                k += 1
        hs.append(tm.msfa.register_forward_hook(lambda mod, i, o: taps.__setitem__("msfa", o.detach())))
    try:
        with torch.no_grad():
            f = m.model.get_image_features(pv)
        f = f.pooler_output if hasattr(f, "pooler_output") else f
        return f.reshape(-1, f.shape[-1])
    finally:
        if hook:
            hook.remove()
        for h in hs:
            h.remove()


def nhwc(t):
    return t[0].permute(1, 2, 0).contiguous().numpy().astype(np.float32)


def main():
    from transformers import AutoTokenizer, Gemma3nForConditionalGeneration, SiglipImageProcessorFast
    cfg = config()
    cfg.text_config._attn_implementation = "eager"
    torch.manual_seed(20261004)
    m = Gemma3nForConditionalGeneration(cfg).float().eval()
    perturb(m)
    d = os.path.join(HFDIR, NAME)
    shutil.rmtree(d, ignore_errors=True)
    m.save_pretrained(d, safe_serialization=True)
    # The audio tower is not this fixture's: llama.cpp's tensor map does not
    # know this transformers' audio names, and no graph here reads audio.
    from safetensors.torch import load_file, save_file
    for f in os.listdir(d):
        if f.endswith(".safetensors"):
            sd = load_file(os.path.join(d, f))
            save_file({k: v for k, v in sd.items() if "audio" not in k}, os.path.join(d, f),
                      metadata={"format": "pt"})
    for f in os.listdir(os.path.join(TOKDIR, "gemma3n")):
        if f not in ("config.json", "full_config.json"):
            shutil.copy(os.path.join(TOKDIR, "gemma3n", f), d)
    improc = SiglipImageProcessorFast(size={"height": 768, "width": 768}, resample=2, do_normalize=False,
                                      do_rescale=True, image_mean=[0.5] * 3, image_std=[0.5] * 3)
    improc.save_pretrained(d)
    pc = json.load(open(os.path.join(d, "preprocessor_config.json")))
    pc["image_seq_length"] = 256
    json.dump(pc, open(os.path.join(d, "preprocessor_config.json"), "w"), indent=1)
    for kind, dst in (("", NAME + ".gguf"), ("--mmproj", "mmproj-" + NAME + ".gguf")):
        args = [PY, os.path.join(LCPP, "convert_hf_to_gguf.py"), d, "--outtype", "f32",
                "--outfile", os.path.join(MODELS, dst)]
        if kind:
            args.append(kind)
        subprocess.run(args, check=True)

    tok = AutoTokenizer.from_pretrained(d)
    img = picture(IMG_H, IMG_W)
    pv = improc(images=[img], return_tensors="pt")["pixel_values"]
    taps = {}
    feats = features(m, pv, taps=taps)
    mq = copy.deepcopy(m)
    quantize_vision(mq)
    feats_q8 = features(mq, pv)
    taps8 = {}
    feats_a8 = features(mq, pv, a8=True, taps=taps8)
    gold = os.path.join(MODELS, NAME + ".gold")
    shutil.rmtree(gold, ignore_errors=True)
    os.makedirs(gold)
    nhwc(pv).tofile(os.path.join(gold, "pixels.f32"))
    shapes = {}
    for k, v in taps8.items():
        nhwc(v).tofile(os.path.join(gold, k + ".a8.f32"))
        shapes[k] = list(v.shape[1:])
    for k, v in taps.items():
        nhwc(v).tofile(os.path.join(gold, k + ".f32"))
    json.dump(shapes, open(os.path.join(gold, "shapes.json"), "w"))

    BOI, EOI, IMG = cfg.boi_token_id, cfg.eoi_token_id, cfg.image_token_id
    pre = [2] + tok.encode("<start_of_turn>user\n", add_special_tokens=False)
    post = tok.encode("What is in the picture?<end_of_turn>\n<start_of_turn>model\n", add_special_tokens=False)
    cont = tok.encode("A disc on a gradient.", add_special_tokens=False)
    nsoft = feats.shape[0]
    ids = pre + [BOI] + [IMG] * nsoft + [EOI] + post + cont
    t_ids = torch.tensor([ids])
    with torch.no_grad():
        out = m(input_ids=t_ids, pixel_values=pv)
    lg = out.logits[0]
    first = len(pre) + nsoft + 2 + len(post) - 1

    def logit_rows(lg):
        return [{"argmax": int(r.argmax()), "max": float(r.max()),
                 "margin": float(r.topk(2).values[0] - r.topk(2).values[1]),
                 "head": [round(float(x), 7) for x in r[:NLOGIT]]} for r in lg[first:]]

    # The same prompt with the tower's rows at the engine's arithmetic
    # (feats_a8) where the image's soft tokens are: the model's own forward,
    # its image features replaced, so the per-layer inputs and the splice are
    # the reference's.
    from types import SimpleNamespace
    real = m.model.get_image_features
    m.model.get_image_features = lambda pv, **kw: SimpleNamespace(pooler_output=feats_a8[None])
    try:
        with torch.no_grad():
            lg8 = m(input_ids=t_ids, pixel_values=pv).logits[0]
    finally:
        m.model.get_image_features = real
    g = {
        "pre": pre, "post": post, "cont": cont, "boi": BOI, "eoi": EOI,
        "image": {"h": IMG_H, "w": IMG_W, "rgb": b64(img), "soft": nsoft,
                  "embd": [round(float(x), 8) for x in feats.reshape(-1)],
                  "embd_q8": [round(float(x), 8) for x in feats_q8.reshape(-1)],
                  "embd_a8": [round(float(x), 8) for x in feats_a8.reshape(-1)]},
        "logits": logit_rows(lg),
        "logits_a8": logit_rows(lg8),
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, NAME + ".json"), "w") as f:
        json.dump(g, f)
    print(f"{NAME}: {len(ids)} ids, {nsoft} soft tokens, {len(g['logits'])} logit rows, {len(taps)} taps")


def b64(img):
    import base64
    return base64.b64encode(img.reshape(-1).tobytes()).decode()


if __name__ == "__main__":
    main()
