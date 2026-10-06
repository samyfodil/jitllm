#!/usr/bin/env python3
"""Transformers' goldens for a real Gemma 4 tower.

    $JITLLM_HF_PY scripts/gemma4real.py [checkpoint dir] [picture.png]

The checkpoint dir holds the model's config.json, processor_config.json and
vision.safetensors (only the tower's and the embedder's tensors, fetched by
range requests: jpt-models/vlmb/fetchstp.py). The tower and the head are
transformers' own Gemma4VisionModel and Gemma4MultimodalEmbedder, built from
the config and run in float32 on the CPU on Gemma4ImageProcessorPil's pixels.
Writes, beside the picture's name in <dir>/goldens/:

  <pic>.pixels.f32    the processor's pixels as the engine lays them out:
                      float32 [y][x][channel], 2x - 1 as the tower does
  <pic>.json          {"h", "w"}: the resized picture's size in pixels
  <pic>.features.f32  the head's rows, float32 [token][text width]
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

import refenv  # where things are: environment variables only (docs/testing.md)


def main(ckpt, pic):
    from transformers import Gemma4Config
    from transformers.models.gemma4.image_processing_pil_gemma4 import Gemma4ImageProcessorPil
    from transformers.models.gemma4.modeling_gemma4 import Gemma4MultimodalEmbedder, Gemma4VisionModel
    cfg = Gemma4Config.from_dict(json.load(open(os.path.join(ckpt, "config.json"))))
    cfg.vision_config._attn_implementation = "eager"
    tower = Gemma4VisionModel(cfg.vision_config).float().eval()
    head = Gemma4MultimodalEmbedder(cfg.vision_config, cfg.text_config).float().eval()
    sd = load_file(os.path.join(ckpt, "vision.safetensors"))

    def part(prefixes):
        out = {}
        for k, v in sd.items():
            for p in prefixes:
                if k.startswith(p):
                    out[k[len(p):]] = v.float()
        return out
    for mod, prefixes in ((tower, ("model.vision_tower.", "vision_tower.")),
                          (head, ("model.embed_vision.", "embed_vision."))):
        missing, unexpected = mod.load_state_dict(part(prefixes), strict=False)
        missing = [k for k in missing if "inv_freq" not in k]
        assert not missing and not unexpected, (missing, unexpected)
    pp = json.load(open(os.path.join(ckpt, "processor_config.json")))
    ip = pp.get("image_processor", pp)
    improc = Gemma4ImageProcessorPil(max_soft_tokens=ip.get("max_soft_tokens", 280))
    img = Image.open(pic).convert("RGB")
    pv = improc(images=[img], return_tensors="pt")
    pix, pos = pv["pixel_values"], pv["image_position_ids"]
    with torch.no_grad():
        hs = tower(pixel_values=pix, pixel_position_ids=pos).last_hidden_state
        feats = head(inputs_embeds=hs)
    valid = (pos[0] >= 0).all(-1)
    xs, ys = pos[0][valid][:, 0], pos[0][valid][:, 1]
    gw, gh = int(xs.max()) + 1, int(ys.max()) + 1
    nsoft = int(valid.sum()) // (cfg.vision_config.pooling_kernel_size ** 2)
    feats = feats.reshape(-1, feats.shape[-1])[:nsoft]
    ps = cfg.vision_config.patch_size
    rows = pix[0][valid].reshape(-1, ps, ps, 3)
    image = np.zeros((gh * ps, gw * ps, 3), np.float32)
    for r, x, y in zip(rows, xs.tolist(), ys.tolist()):
        image[y * ps:(y + 1) * ps, x * ps:(x + 1) * ps] = 2 * (r.numpy() - 0.5)
    out = os.path.join(ckpt, "goldens")
    os.makedirs(out, exist_ok=True)
    name = os.path.splitext(os.path.basename(pic))[0]
    image.tofile(os.path.join(out, name + ".pixels.f32"))
    feats.contiguous().numpy().astype(np.float32).tofile(os.path.join(out, name + ".features.f32"))
    json.dump({"h": gh * ps, "w": gw * ps}, open(os.path.join(out, name + ".json"), "w"))
    print(f"{name}: {img.size[0]}x{img.size[1]} -> {gw * ps}x{gh * ps}, {feats.shape[0]} rows of {feats.shape[1]}")


if __name__ == "__main__":
    main(refenv.arg(1, "vlmb", "hf", "gemma-4-E2B-it"),
         refenv.arg(2, "vlm", "goldens", "photo.png"))
