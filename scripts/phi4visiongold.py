#!/usr/bin/env python3
"""Phi-4-reasoning-vision's goldens, from the reference itself.

    $JITLLM_HF_PY scripts/phi4visiongold.py
        -> $JITLLM_MODELS/synth-phi4v.gguf, $JITLLM_MODELS/mmproj-synth-phi4v.gguf
           and testdata/golden/phi4v/synth-phi4v.json: the fixture.

    $JITLLM_HF_PY scripts/phi4visiongold.py --model-free
        -> testdata/golden/phi4-naflex.json: the processor's resample size for
           many pictures (transformers' get_image_size_for_max_num_patches, as
           Siglip2ImageProcessorNoUpscale calls it), and SigLIP2's position-
           table resize (F.interpolate, bilinear, antialias) of a 16x16 table
           of a known formula onto several grids.

    $JITLLM_HF_PY scripts/phi4visiongold.py DIR IMAGE...
        -> DIR/gold/: the processor's pixels, the tower's hidden_states[-2]
           and the projector's output, per picture, from the real checkpoint's
           vision.safetensors.

THE FIXTURE'S ORACLE IS transformers' OWN CLASSES, every parameter randomised:
Siglip2VisionModel read at hidden_states[-2] (the checkpoint's
Siglip2VisionTower.feature_select), the mlp2x_gelu projector, and
Phi3ForCausalLM for the text model -- which is what the checkpoint's
Phi4ForCausalLMV is (modeling_phi4_visionr.py subclasses it and splices the
picture's rows where "<image>" stood). BOTH GGUFs ARE WRITTEN BY llama.cpp's
OWN convert_hf_to_gguf.py from that checkpoint, the text model plainly and the
tower with --mmproj, which drops the last block and the post-LayerNorm and
writes the pixel bounds. The picture is small and NOT square, so the processor
resizes it UP to the minimum patch count and the 16x16 table is resized to a
grid of another shape on both axes.
"""
import copy
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

import refenv  # where things are: environment variables only (docs/testing.md)

PATCH, LO, HI = 16, 256, 3600
SIZES = [(640, 488), (896, 448), (384, 384), (100, 100), (3000, 2000), (4000, 300), (300, 4000),
         (1920, 1080), (256, 256), (257, 255), (16, 3000), (961, 959), (1024, 1024), (500, 333)]
GRIDS = [(16, 16), (20, 26), (31, 59), (8, 32), (12, 47), (1, 16), (5, 3)]

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
OUT = os.path.join(HERE, "..", "testdata", "golden", "phi4v")
NAME = "synth-phi4v"
NLOGIT = 512
IMG_H, IMG_W = 136, 200
VIS = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=2,
           patch_size=PATCH, num_patches=256, vision_use_head=False, layer_norm_eps=1e-6)


def table(side, ch):
    """The deterministic table both sides build: t[y][x][c] = sin(0.37*y + 0.11*x*c + c)."""
    t = np.zeros((side, side, ch), dtype=np.float32)
    for y in range(side):
        for x in range(side):
            for c in range(ch):
                t[y, x, c] = math.sin(0.37 * y + 0.11 * x * c + c)
    return t


def naflex_size(w, h):
    from transformers.models.siglip2 import image_processing_siglip2 as ips
    n = max((w // PATCH) * (h // PATCH), 1)
    target = LO if n < LO else HI if n > HI else n
    th, tw = ips.get_image_size_for_max_num_patches(image_height=h, image_width=w, patch_size=PATCH,
                                                   max_num_patches=target)
    return tw, th


def model_free():
    sizes = []
    for w, h in SIZES:
        tw, th = naflex_size(w, h)
        sizes.append({"w": w, "h": h, "tw": tw, "th": th})
    t = torch.from_numpy(table(16, 2)).permute(2, 0, 1).unsqueeze(0)  # [1, C, 16, 16]
    grids = []
    for gh, gw in GRIDS:
        r = F.interpolate(t, size=(gh, gw), mode="bilinear", align_corners=False, antialias=True)
        grids.append({"gh": gh, "gw": gw, "v": r[0].permute(1, 2, 0).reshape(-1).tolist()})
    dst = os.path.join(HERE, "..", "testdata", "golden", "phi4-naflex.json")
    json.dump({"patch": PATCH, "lo": LO, "hi": HI, "sizes": sizes, "grids": grids}, open(dst, "w"))
    print("wrote", dst)


def preprocess(img):
    """Siglip2ImageProcessorNoUpscale, restated because the checkpoint's module
    imports names this transformers no longer has: the size from
    get_image_size_for_max_num_patches (even for an in-range picture), PIL's
    BILINEAR resize, /255, (x - 0.5) / 0.5, and the patches row-major, each
    (ky, kx, c)."""
    from PIL import Image
    w, h = img.size
    tw, th = naflex_size(w, h)
    px = np.asarray(img.resize((tw, th), resample=Image.Resampling.BILINEAR)).astype(np.float32) / 255
    px = ((px - 0.5) / 0.5).astype(np.float32)
    gh, gw = th // PATCH, tw // PATCH
    patches = px.reshape(gh, PATCH, gw, PATCH, 3).transpose(0, 2, 1, 3, 4).reshape(gh * gw, -1)
    return px, torch.from_numpy(np.ascontiguousarray(patches)).unsqueeze(0), gh, gw


def tower_rows(vis, pv, gh, gw):
    mask = torch.ones((1, gh * gw), dtype=torch.int32)
    shapes = torch.tensor([[gh, gw]])
    with torch.no_grad():
        o = vis(pixel_values=pv, pixel_attention_mask=mask, spatial_shapes=shapes, output_hidden_states=True)
    return o.hidden_states[-2][0]


def checkpoint(d, images):
    """The tower and projector the real checkpoint's code builds, from its
    vision.safetensors (model.vision_tower.vision_tower.* and
    model.mm_projector.*)."""
    from PIL import Image
    from safetensors.torch import load_file
    from transformers import Siglip2VisionConfig, Siglip2VisionModel
    cfg = json.load(open(os.path.join(d, "config.json")))
    vc = Siglip2VisionConfig(**{**cfg["vision_config"], "patch_size": PATCH, "num_patches": 256})
    vc._attn_implementation = "eager"
    vis = Siglip2VisionModel(vc)
    sd = {k: v.float() for k, v in load_file(os.path.join(d, "vision.safetensors")).items()}
    pre = "model.vision_tower.vision_tower.vision_model."
    missing, unexpected = vis.load_state_dict({k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)},
                                              strict=False)
    if missing:
        sys.exit(f"missing {missing[:5]}")
    proj = torch.nn.Sequential(torch.nn.Linear(1152, cfg["hidden_size"]), torch.nn.GELU(),
                               torch.nn.Linear(cfg["hidden_size"], cfg["hidden_size"]))
    proj.load_state_dict({k[len("model.mm_projector."):]: v for k, v in sd.items()
                          if k.startswith("model.mm_projector.")}, strict=True)
    vis.float().eval()
    proj.float().eval()
    out = os.path.join(d, "gold")
    os.makedirs(out, exist_ok=True)
    for path in images:
        name = os.path.splitext(os.path.basename(path))[0]
        img = Image.open(path).convert("RGB")
        px, pv, gh, gw = preprocess(img)
        vit = tower_rows(vis, pv, gh, gw)
        with torch.no_grad():
            emb = proj(vit)
        base = os.path.join(out, name)
        np.ascontiguousarray(px).tofile(base + ".px")
        vit.numpy().astype(np.float32).tofile(base + ".vit")
        emb.numpy().astype(np.float32).tofile(base + ".emb")
        json.dump({"w": img.size[0], "h": img.size[1], "gh": gh, "gw": gw,
                   "vit_sum": float(vit.sum()), "emb_sum": float(emb.sum())}, open(base + ".json", "w"))
        print(f"{name}: {img.size} -> {gw}x{gh} patches, emb {tuple(emb.shape)} sum {float(emb.sum()):.4f}")


# The fixture.

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


def chw_patch(p):
    # The patch row as the engine gathers it, channel-planar (c, ky, kx), from
    # the processor's (ky, kx, c) order.
    n = p.shape[-1]
    s = int((n // 3) ** 0.5)
    return p.reshape(*p.shape[:-1], s, s, 3).permute(*range(p.ndim - 1), -1, -3, -2).reshape(p.shape)


def from_chw(p):
    n = p.shape[-1]
    s = int((n // 3) ** 0.5)
    return p.reshape(*p.shape[:-1], 3, s, s).permute(*range(p.ndim - 1), -2, -1, -3).reshape(p.shape)


class Proj(torch.nn.Module):
    """mlp2x_gelu: Linear, GELU, Linear (model.mm_projector.0 and .2). act is
    the reference's erf GELU, or the engine's tanh form for the
    arithmetic-matched golden (llama.cpp's ggml_gelu)."""

    def __init__(self, k, n):
        super().__init__()
        self.mm = torch.nn.Sequential(torch.nn.Linear(k, n), torch.nn.GELU(), torch.nn.Linear(n, n))

    def forward(self, x, tanh=False):
        h = self.mm[0](x)
        h = F.gelu(h, approximate="tanh") if tanh else F.gelu(h)
        return self.mm[2](h)


def features(vis, proj, pv, gh, gw, a8=False):
    hs = []
    if a8:
        for n, mod in list(vis.named_modules()) + [("proj." + n, m) for n, m in proj.named_modules()]:
            if not isinstance(mod, torch.nn.Linear):
                continue
            if "patch_embedding" in n:
                hs.append(mod.register_forward_pre_hook(lambda mod, inp: (from_chw(qact(chw_patch(inp[0]))),)))
            else:
                hs.append(mod.register_forward_pre_hook(lambda mod, inp: (qact(inp[0]),)))
    try:
        vit = tower_rows(vis, pv, gh, gw)
        with torch.no_grad():
            return vit, proj(vit, tanh=a8)
    finally:
        for h in hs:
            h.remove()


def quantize(vis, proj):
    with torch.no_grad():
        for n, p in list(vis.named_parameters()) + list(proj.named_parameters()):
            if not n.endswith("weight") or p.ndim != 2:
                continue
            if "patch_embedding" in n:
                p.copy_(from_chw(q8rows(chw_patch(p))))
            else:
                p.copy_(q8rows(p))


def perturb(mod, tower):
    with torch.no_grad():
        for n, p in mod.named_parameters():
            last = n.split(".")[-2] if "." in n else ""
            if n.endswith("weight") and ("norm" in last or "layer_norm" in last):
                p.copy_(torch.randn_like(p) * 0.3 + 1)
            elif n.endswith("bias"):
                p.copy_(torch.randn_like(p) * 0.3)
            elif tower and ("q_proj" in n or "k_proj" in n):
                p.copy_(torch.randn_like(p) * 0.22)
            elif "position_embedding" in n:
                p.copy_(torch.randn_like(p) * 0.5)
            else:
                p.copy_(torch.randn_like(p) * 0.2)


def b64(img):
    import base64
    return base64.b64encode(img.reshape(-1).tobytes()).decode()


def fixture():
    # Only the fixture writes files, so only it needs to know where they go.
    models = refenv.models()
    tok_dir = refenv.under("PHI4V_TOK", "tok", "phi4v")
    hf_dir = refenv.under("PHI4V_HF", "vlmb", "fixtures")
    lcpp = refenv.need("LCPP_SRC", "a llama.cpp source checkout")
    from PIL import Image
    from safetensors.torch import save_file
    from transformers import AutoTokenizer, Phi3Config, Phi3ForCausalLM, Siglip2VisionConfig, Siglip2VisionModel
    torch.manual_seed(20261004)
    tc = Phi3Config(hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4,
                    num_key_value_heads=2, vocab_size=100352, max_position_embeddings=32768,
                    original_max_position_embeddings=32768, rope_theta=500000.0, rms_norm_eps=1e-5,
                    sliding_window=None, tie_word_embeddings=False, pad_token_id=100349,
                    bos_token_id=100257, eos_token_id=100265, partial_rotary_factor=1.0)
    tc._attn_implementation = "eager"
    text = Phi3ForCausalLM(tc).float().eval()
    perturb(text, False)
    vc = Siglip2VisionConfig(**VIS)
    vc._attn_implementation = "eager"
    vis = Siglip2VisionModel(vc).float().eval()
    perturb(vis, True)
    proj = Proj(VIS["hidden_size"], tc.hidden_size).float().eval()
    perturb(proj, False)
    with torch.no_grad():
        for p in proj.parameters():
            p.mul_(0.15)

    d = os.path.join(hf_dir, NAME)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    sd = {k: v.contiguous() for k, v in text.state_dict().items()}
    for k, v in vis.state_dict().items():
        sd["model.vision_tower.vision_tower.vision_model." + k] = v.contiguous()
    for k, v in proj.mm.state_dict().items():
        sd["model.mm_projector." + k] = v.contiguous()
    save_file(sd, os.path.join(d, "model.safetensors"), metadata={"format": "pt"})
    cfg = json.loads(tc.to_json_string())
    cfg.update({"architectures": ["Phi4ForCausalLMV"], "model_type": "phi4-siglip",
                "mm_hidden_size": VIS["hidden_size"], "mm_projector_type": "mlp2x_gelu",
                "min_num_patches": LO, "max_num_patches": HI,
                "vision_config": {**VIS, "model_type": "siglip2_vision_model"}})
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"), indent=1)
    json.dump({"image_processor_type": "Siglip2ImageProcessorNoUpscale", "patch_size": PATCH,
               "min_num_patches": LO, "max_num_patches": HI, "image_mean": [0.5] * 3, "image_std": [0.5] * 3,
               "do_normalize": True, "do_rescale": True, "do_resize": True, "resample": 2,
               "rescale_factor": 1 / 255}, open(os.path.join(d, "preprocessor_config.json"), "w"))
    for f in os.listdir(tok_dir):
        shutil.copy(os.path.join(tok_dir, f), d)
    for kind, dst in (("", NAME + ".gguf"), ("--mmproj", "mmproj-" + NAME + ".gguf")):
        args = [PY, os.path.join(lcpp, "convert_hf_to_gguf.py"), d, "--outtype", "f32",
                "--outfile", os.path.join(models, dst)]
        if kind:
            args.append(kind)
        subprocess.run(args, check=True)

    tok = AutoTokenizer.from_pretrained(tok_dir)
    img = picture(IMG_H, IMG_W)
    px, pv, gh, gw = preprocess(Image.fromarray(img))
    vit, feats = features(vis, proj, pv, gh, gw)
    vq, pq = copy.deepcopy(vis), copy.deepcopy(proj)
    quantize(vq, pq)
    vit8, feats_a8 = features(vq, pq, pv, gh, gw, a8=True)
    pre = tok.encode("<|im_start|>user<|im_sep|>", add_special_tokens=False)
    post = tok.encode("\nWhat is in the picture?<|im_end|><|im_start|>assistant<|im_sep|>",
                      add_special_tokens=False)
    cont = tok.encode("A disc on a gradient.", add_special_tokens=False)
    emb = text.get_input_embeddings()
    first = len(pre) + gh * gw + len(post) - 1

    def logits(rows):
        with torch.no_grad():
            e = torch.cat([emb(torch.tensor(pre)), rows, emb(torch.tensor(post + cont))], 0)
            lg = text(inputs_embeds=e.unsqueeze(0)).logits[0]
        return [{"argmax": int(r.argmax()), "max": float(r.max()),
                 "margin": float(r.topk(2).values[0] - r.topk(2).values[1]),
                 "head": [round(float(x), 7) for x in r[:NLOGIT]]} for r in lg[first:]]

    g = {
        "pre": pre, "post": post, "cont": cont,
        "image": {"h": IMG_H, "w": IMG_W, "rgb": b64(img), "grid": [gh, gw],
                  "px": [round(float(x), 8) for x in px.reshape(-1)],
                  "vit": [round(float(x), 8) for x in vit.reshape(-1)],
                  "vit_a8": [round(float(x), 8) for x in vit8.reshape(-1)],
                  "embd": [round(float(x), 8) for x in feats.reshape(-1)],
                  "embd_a8": [round(float(x), 8) for x in feats_a8.reshape(-1)]},
        "logits": logits(feats),
        "logits_a8": logits(feats_a8),
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, NAME + ".json"), "w") as f:
        json.dump(g, f)
    print(f"{NAME}: {IMG_W}x{IMG_H} -> {gw}x{gh} patches, {len(pre) + gh * gw + len(post) + len(cont)} positions")


if __name__ == "__main__":
    if sys.argv[1:] == ["--model-free"]:
        model_free()
    elif sys.argv[1:]:
        checkpoint(sys.argv[1], sys.argv[2:])
    else:
        fixture()
