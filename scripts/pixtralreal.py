#!/usr/bin/env python3
"""Transformers' goldens for a real Pixtral / Mistral 3 tower.

    $JITLLM_HF_PY scripts/pixtralreal.py [checkpoint dir] [picture.png]

The checkpoint dir holds the model's config.json and vision.safetensors (only
the tower's and the projector's tensors, fetched by range requests:
jpt-models/vlmb/fetchstp.py). Ministral 3's text model is FP8 and is not
needed: the tower and the projector are transformers' own PixtralVisionModel
and Mistral3MultiModalProjector, built from the config and run in float32 on
the CPU on PixtralImageProcessorPil's pixels. Writes, beside the picture's
name in <dir>/goldens/:

  <pic>.pixels.f32    the processor's pixel_values, float32 [y][x][channel]
  <pic>.json          {"h", "w"}: the resized picture's size in pixels
  <pic>.features.f32  the projector's rows, float32 [token][text width]
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
    from transformers import Mistral3Config, PixtralImageProcessorPil
    from transformers.models.mistral3.modeling_mistral3 import Mistral3MultiModalProjector
    from transformers.models.pixtral.modeling_pixtral import PixtralVisionModel
    cfg = Mistral3Config.from_dict(json.load(open(os.path.join(ckpt, "config.json"))))
    cfg.vision_config._attn_implementation = "eager"
    tower = PixtralVisionModel(cfg.vision_config).float().eval()
    proj = Mistral3MultiModalProjector(cfg).float().eval()
    sd = load_file(os.path.join(ckpt, "vision.safetensors"))
    def part(prefixes):
        out = {}
        for k, v in sd.items():
            for p in prefixes:
                if k.startswith(p):
                    out[k[len(p):]] = v.float()
        return out
    for mod, prefixes in ((tower, ("model.vision_tower.", "vision_tower.")),
                          (proj, ("model.multi_modal_projector.", "multi_modal_projector."))):
        missing, unexpected = mod.load_state_dict(part(prefixes), strict=False)
        missing = [k for k in missing if "inv_freq" not in k]
        assert not missing and not unexpected, (missing, unexpected)
    pp = json.load(open(os.path.join(ckpt, "processor_config.json")))
    ip = pp["image_processor"]
    merge = pp.get("spatial_merge_size", 1)
    improc = PixtralImageProcessorPil(patch_size={"height": ip["patch_size"] * merge,
                                                  "width": ip["patch_size"] * merge},
                                      size=ip["size"], image_mean=ip["image_mean"], image_std=ip["image_std"])
    img = Image.open(pic).convert("RGB")
    pv = improc(images=[img], return_tensors="pt")
    h, w = [int(v) for v in pv["image_sizes"][0]]
    px = pv["pixel_values"][:, :, :h, :w]
    with torch.no_grad():
        hs = tower(px, image_sizes=pv["image_sizes"]).last_hidden_state.squeeze(0)
        feats = proj(hs, pv["image_sizes"])
    out = os.path.join(ckpt, "goldens")
    os.makedirs(out, exist_ok=True)
    name = os.path.splitext(os.path.basename(pic))[0]
    px[0].permute(1, 2, 0).contiguous().numpy().astype(np.float32).tofile(os.path.join(out, name + ".pixels.f32"))
    feats.contiguous().numpy().astype(np.float32).tofile(os.path.join(out, name + ".features.f32"))
    json.dump({"h": h, "w": w}, open(os.path.join(out, name + ".json"), "w"))
    print(f"{name}: {img.size[0]}x{img.size[1]} -> {w}x{h}, {feats.shape[0]} rows of {feats.shape[1]}")


if __name__ == "__main__":
    main(refenv.arg(1, "vlmb", "hf", "Ministral-3-3B"),
         refenv.arg(2, "vlm", "goldens", "photo.png"))
