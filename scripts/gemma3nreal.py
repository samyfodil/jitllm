#!/usr/bin/env python3
"""Transformers' goldens for a real Gemma 3n tower.

    $JITLLM_HF_PY scripts/gemma3nreal.py [checkpoint dir] [picture.png]

The checkpoint dir holds the model's config.json, preprocessor_config.json and
vision.safetensors (only the tower's and the embedder's tensors, fetched by
range requests: jpt-models/vlmb/fetchstp.py). The tower and the head are
transformers' own -- the vision config's TimmWrapperModel (MobileNet-V5) and
Gemma3nMultimodalEmbedder -- run as Gemma3nModel.get_image_features runs
them, in float32 on the CPU, on SiglipImageProcessorFast's pixels. Writes,
beside the picture's name in <dir>/goldens/:

  <pic>.pixels.f32    the processor's pixels as the engine lays them out:
                      float32 [y][x][channel]
  <pic>.features.f32  the embedder's rows, float32 [token][text width]
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
    from transformers import AutoImageProcessor, AutoModel, Gemma3nConfig
    from transformers.models.gemma3n.modeling_gemma3n import Gemma3nMultimodalEmbedder
    cfg = Gemma3nConfig.from_dict(json.load(open(os.path.join(ckpt, "config.json"))))
    tower = AutoModel.from_config(cfg.vision_config).float().eval()
    head = Gemma3nMultimodalEmbedder(cfg.vision_config, cfg.text_config).float().eval()
    sd = load_file(os.path.join(ckpt, "vision.safetensors"))

    def part(prefix):
        return {k[len(prefix):]: v.float() for k, v in sd.items() if k.startswith(prefix)}
    for mod, prefix in ((tower, "model.vision_tower."), (head, "model.embed_vision.")):
        missing, unexpected = mod.load_state_dict(part(prefix), strict=False)
        assert not missing and not unexpected, (missing, unexpected)
    improc = AutoImageProcessor.from_pretrained(ckpt)
    img = Image.open(pic).convert("RGB")
    pv = improc(images=[img], return_tensors="pt")["pixel_values"].float()
    with torch.no_grad():
        hs = tower(pixel_values=pv, do_pooling=False, return_dict=True).last_hidden_state
        hs = hs.reshape(hs.shape[0], cfg.vision_config.hidden_size, cfg.vision_soft_tokens_per_image).permute(0, 2, 1)
        hs = hs * cfg.vision_config.hidden_size ** 0.5
        feats = head(inputs_embeds=hs)
    out = os.path.join(ckpt, "goldens")
    os.makedirs(out, exist_ok=True)
    name = os.path.splitext(os.path.basename(pic))[0]
    pv[0].permute(1, 2, 0).contiguous().numpy().astype(np.float32).tofile(os.path.join(out, name + ".pixels.f32"))
    feats[0].contiguous().numpy().astype(np.float32).tofile(os.path.join(out, name + ".features.f32"))
    print(f"{name}: {img.size[0]}x{img.size[1]} -> {pv.shape[3]}x{pv.shape[2]}, {feats.shape[1]} rows of {feats.shape[2]}")


if __name__ == "__main__":
    main(refenv.arg(1, "vlmb", "hf", "gemma-3n-E2B-it"),
         refenv.arg(2, "vlm", "goldens", "photo.png"))
