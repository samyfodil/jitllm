#!/usr/bin/env python3
"""Record Moonshot's tower output for the real Kimi-VL on a real picture.

    $JITLLM_HF_PY scripts/kimivlreal.py <checkpoint dir> [image]

JITLLM_MODELS and LCPP_SRC as scripts/refenv.py reads them.

-> testdata/golden/qwenvl/real-Kimi-VL-A3B-Instruct-tower.json, and
   $JITLLM_MODELS/mmproj-Kimi-VL-A3B-Instruct-f16.gguf (llama.cpp's converter)

The whole Kimi-VL is 16B and its float32 reference does not fit a laptop, so
this pins the half that is new: MoonViT and the projector, built by Moonshot's
own modeling_kimi_vl.py (kimivlgold.py's shims) from the checkpoint's own
weights -- the vision shard alone is needed -- on the picture its own
processor prepares. The golden carries the tower's rows in float32 and with
the converter's Q8_0 matrices and the engine's int8 activations
(qwenvlgold.py's simulation). The text model is DeepSeek-V3's, which the engine
already holds to llama.cpp on Moonlight-16B-A3B, the same text model at
pretraining; the test pairs the tower with that file only so a container
exists, and reads nothing past the tower.
"""
import copy
import json
import os
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SRC = sys.argv[1]
os.environ["KVL_SRC"] = SRC
import kimivlgold as kg  # noqa: E402
import qwenvlgold as q  # noqa: E402

NAME = "Kimi-VL-A3B-Instruct"
MODELS = kg.MODELS
IMG = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "engine", "model", "testdata", "quad-and-disc.png")


class Tower(torch.nn.Module):
    # The two halves _extract_image_features reads, without the 16B text model.
    def __init__(self, mod, cfg):
        super().__init__()
        self.vision_tower = mod.MoonVitPretrainedModel(cfg.vision_config)
        self.multi_modal_projector = mod.KimiVLMultiModalProjector(cfg)
        self._extract = mod.KimiVLForConditionalGeneration._extract_image_features

    def _extract_image_features(self, pv, hw):
        return self._extract(self, pv, hw)


def vision_dir():
    # The tower's and projector's tensors, which the index names and which
    # sit in the first shard -- only that shard need be downloaded.
    idx = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
    keep = {k: v for k, v in idx["weight_map"].items()
            if k.startswith("vision_tower.") or k.startswith("multi_modal_projector.")}
    return keep


def write_vision(raw):
    # The checkpoint as the converter's --mmproj reads it: its config and one
    # file of the tower's and projector's tensors, as published (the
    # converter refuses an index whose files carry tensors it does not name).
    from safetensors.torch import save_file
    d = os.path.join(os.path.dirname(SRC.rstrip("/")), NAME + "-vision")
    os.makedirs(d, exist_ok=True)
    save_file(raw, os.path.join(d, "model-vision.safetensors"), metadata={"format": "pt"})
    json.dump({"metadata": {}, "weight_map": {k: "model-vision.safetensors" for k in raw}},
              open(os.path.join(d, "model.safetensors.index.json"), "w"))
    for f in os.listdir(SRC):
        if f.endswith(".json") and f != "model.safetensors.index.json" or f.endswith(".py") or f == "tiktoken.model":
            dst = os.path.join(d, f)
            if not os.path.exists(dst):
                os.symlink(os.path.join(SRC, f), dst)
    return d


def main():
    mod, cmod = kg.remote()
    cfg = cmod.KimiVLConfig.from_pretrained(SRC)
    cfg.vision_config._attn_implementation = "eager"
    m = Tower(mod, cfg).float().eval()
    keep = vision_dir()
    from safetensors import safe_open
    raw = {}
    for f in sorted(set(keep.values())):
        with safe_open(os.path.join(SRC, f), "pt") as st:
            for k in st.keys():
                if k in keep:
                    raw[k] = st.get_tensor(k)
    sd = {k: v.float() for k, v in raw.items()}
    d = write_vision(raw)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        sys.exit(f"the vision weights do not fit Moonshot's classes: missing {missing}, unexpected {unexpected}")
    subprocess.run([sys.executable, os.path.join(kg.LCPP, "convert_hf_to_gguf.py"), d, "--mmproj",
                    "--outtype", "f16", "--outfile", os.path.join(MODELS, "mmproj-" + NAME + "-f16.gguf")],
                   check=True)
    from PIL import Image
    pic = Image.open(IMG).convert("RGB")
    img = np.asarray(pic)
    pv, hw = kg.pixels(img)
    gh, gw = [int(v) for v in hw[0]]
    with torch.no_grad():
        feats = m._extract_image_features(pv, hw)
    mq = copy.deepcopy(m)
    kg.quantize(mq)
    feats_a8 = kg.features_a8(mq, pv, hw)
    g = {"image": {"h": img.shape[0], "w": img.shape[1], "rgb": q.b64(img), "grid": [1, gh // 2, gw // 2],
                   "embd": [round(float(x), 6) for x in feats.reshape(-1)],
                   "embd_a8": [round(float(x), 6) for x in feats_a8.reshape(-1)]}}
    out = os.path.join(kg.OUT, "real-" + NAME + "-tower.json")
    with open(out, "w") as f:
        json.dump(g, f)
    a = feats_a8.reshape(-1).double()
    b = feats.reshape(-1).double()
    print(f"{NAME}: {gh}x{gw} patches, {feats.shape[0]} rows of {feats.shape[1]}; "
          f"the int8 alone moves Moonshot's tower by NMSE {float(((a - b) ** 2).sum() / (b ** 2).sum()):.3e}")


if __name__ == "__main__":
    main()
