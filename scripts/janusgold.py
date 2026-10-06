#!/usr/bin/env python3
"""Janus-Pro's own processor, vision tower and aligner, run on pictures. -> DIR/gold/

    $JITLLM_HF_PY scripts/janusgold.py DIR IMAGE...

DIR holds deepseek-community/Janus-Pro-1B's config.json and
preprocessor_config.json and vision.safetensors, the model.vision_model.* and
model.aligner.* tensors cut out of its checkpoint (minicpmv's fetchst.py does
the cutting). transformers' own JanusVisionModel and JanusVisionAlignerMLP run
them in float32 on the CPU, and the processor is the PIL one
(JanusImageProcessorPil), which is Pillow's resize as the GGUF's preprocessing
is.

For each picture: NAME.json, and raw little-endian float32
    NAME.px    the normalised 384x384 picture, [y][x][channel]
    NAME.vit   the tower's last_hidden_state (after its post-LayerNorm), [576][1024]
    NAME.emb   the aligner's output, [576][2048]
"""
import json, os, sys

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import JanusConfig
from transformers.models.janus.modeling_janus import JanusVisionModel, JanusVisionAlignerMLP
from transformers.models.janus.image_processing_pil_janus import JanusImageProcessorPil


def main():
    d, images = sys.argv[1], sys.argv[2:]
    out = os.path.join(d, "gold")
    os.makedirs(out, exist_ok=True)
    cfg = JanusConfig.from_pretrained(d)
    vis = JanusVisionModel(cfg.vision_config)
    al = JanusVisionAlignerMLP(cfg.vision_config)
    sd = {k: v.float() for k, v in load_file(os.path.join(d, "vision.safetensors")).items()}
    vis.load_state_dict({k[len("model.vision_model."):]: v for k, v in sd.items()
                         if k.startswith("model.vision_model.")}, strict=True)
    al.load_state_dict({k[len("model.aligner."):]: v for k, v in sd.items()
                        if k.startswith("model.aligner.")}, strict=True)
    vis.float().eval()
    al.float().eval()
    pc = json.load(open(os.path.join(d, "preprocessor_config.json")))
    proc = JanusImageProcessorPil(**{k: v for k, v in pc.items() if k not in ("image_processor_type", "processor_class")})
    for path in images:
        name = os.path.splitext(os.path.basename(path))[0]
        img = Image.open(path).convert("RGB")
        pv = proc(img, return_tensors="pt")["pixel_values"]  # [1, 3, 384, 384]
        with torch.no_grad():
            vit = vis(pv).last_hidden_state
            emb = al(vit)
        base = os.path.join(out, name)
        np.ascontiguousarray(pv[0].permute(1, 2, 0).numpy().astype(np.float32)).tofile(base + ".px")
        vit[0].numpy().astype(np.float32).tofile(base + ".vit")
        emb[0].numpy().astype(np.float32).tofile(base + ".emb")
        json.dump({"w": img.size[0], "h": img.size[1], "vit_sum": float(vit.sum()), "emb_sum": float(emb.sum())},
                  open(base + ".json", "w"))
        print(f"{name}: {img.size} -> vit {tuple(vit.shape)} emb {tuple(emb.shape)} sum {float(emb.sum()):.4f}")


if __name__ == "__main__":
    main()
