#!/usr/bin/env python3
"""MiniCPM-V's own processor, tower and resampler, run on pictures. -> DIR/gold/

    $JITLLM_HF_PY scripts/minicpmvgold.py DIR IMAGE...

DIR holds the checkpoint's remote code and configs (config.json,
preprocessor_config.json, image_processing_minicpmv.py, modeling_navit_siglip.py,
resampler.py, configuration_minicpm.py) and vision.safetensors, the vpm.* and
resampler.* tensors cut out of the checkpoint. Nothing here is llama.cpp's: this
is the reference the weights were trained against (RULE 7m), run in float32 on
the CPU.

For each picture it writes NAME.json (the layout: every piece's pixel size,
patch grid and position ids) and, per piece i, raw little-endian float32:

    NAME.pN.px     the normalised piece, [y][x][channel]
    NAME.pN.vit    the tower's output after its post-LayerNorm, [patches][1152]
    NAME.pN.emb    the resampler's output, [queries][d]

The test reads them with JITLLM_MINICPMV_GOLD pointing at DIR/gold.
"""
import json, os, sys, types

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file


def load(d):
    sys.path.insert(0, d)
    # The remote code imports itself as a package; give it one.
    pkg = types.ModuleType("mcpv")
    pkg.__path__ = [d]
    sys.modules["mcpv"] = pkg
    import importlib
    conf = importlib.import_module("mcpv.configuration_minicpm")
    siglip = importlib.import_module("mcpv.modeling_navit_siglip")
    rs = importlib.import_module("mcpv.resampler")
    cfg = conf.MiniCPMVConfig.from_pretrained(d)
    cfg.vision_config._attn_implementation = "eager"
    vpm = siglip.SiglipVisionTransformer(cfg.vision_config)
    if cfg.drop_vision_last_layer:
        vpm.encoder.layers = vpm.encoder.layers[:-1]
    res = rs.Resampler(num_queries=cfg.query_num, embed_dim=cfg.hidden_size,
                       num_heads=cfg.hidden_size // 128, kv_dim=cfg.vision_config.hidden_size,
                       adaptive=True)
    sd = load_file(os.path.join(d, "vision.safetensors"))
    sd = {k: v.float() for k, v in sd.items()}
    vpm.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("vpm.")}, strict=True)
    res.load_state_dict({k[10:]: v for k, v in sd.items() if k.startswith("resampler.")}, strict=False)
    vpm.float().eval()
    res.float().eval()
    return cfg, vpm, res, processor(d)


def processor(d):
    sys.path.insert(0, d)
    pkg = types.ModuleType("mcpv")
    pkg.__path__ = [d]
    sys.modules.setdefault("mcpv", pkg)
    import importlib
    ip = importlib.import_module("mcpv.image_processing_minicpmv")
    pc = json.load(open(os.path.join(d, "preprocessor_config.json")))
    skip = ("auto_map", "image_processor_type", "processor_class")
    return ip.MiniCPMVImageProcessor(**{k: v for k, v in pc.items() if k not in skip})


def layouts(d):
    """The processor's cut of many picture sizes, from its own methods, into
    testdata/golden/minicpmv-layout.json: no pixels and no weights, so the
    test that reads it needs no model."""
    proc = processor(d)
    sizes = [(384, 384), (448, 448), (500, 300), (300, 500), (640, 488), (896, 448), (448, 896),
             (1000, 700), (1024, 768), (1920, 1080), (1080, 1920), (3000, 400), (400, 3000),
             (14, 6000), (6000, 14), (1344, 1344), (2000, 2000), (700, 699), (449, 448),
             (100, 100), (28, 2000), (1500, 900), (4000, 3000), (813, 1217)]
    rows = []
    for w, h in sizes:
        grid = proc.get_sliced_grid((w, h), proc.max_slice_nums)
        if grid is None:
            ov = proc.find_best_resize((w, h), proc.scale_resolution, proc.patch_size, allow_upscale=True)
            ref = None
        else:
            ov = proc.find_best_resize((w, h), proc.scale_resolution, proc.patch_size)
            ref = proc.get_refine_size((w, h), grid, proc.scale_resolution, proc.patch_size, allow_upscale=True)
        rows.append({"w": w, "h": h, "overview": list(ov), "grid": grid, "refine": list(ref) if ref else None})
    # The tower's position buckets on one axis of n patches, as
    # modeling_navit_siglip.py computes them in float32 (B = 70).
    B = 70
    bounds = torch.arange(1 / B, 1.0, 1 / B)
    # Stored as the places where float32 disagrees with the rational
    # floor(B*i/n) -- llama.cpp's formula -- which is what makes them worth a
    # golden; every other index is that floor.
    buckets = {}
    for n in range(1, 1500):
        frac = torch.arange(0, 1 - 1e-6, 1 / torch.tensor(n))
        got = torch.bucketize(frac, bounds, right=True).tolist()
        assert len(got) == n
        buckets[n] = [[i, b] for i, b in enumerate(got) if b != (B * i) // n]
    here = os.path.dirname(os.path.abspath(__file__))
    dst = os.path.join(here, "..", "testdata", "golden", "minicpmv-layout.json")
    json.dump({"layouts": rows, "buckets": buckets}, open(dst, "w"))
    print("wrote", dst, len(rows), "sizes")


PIL_SIZES = [(448, 448), (200, 150), (300, 620), (523, 523), (384, 97)]


def pil(src):
    """Pillow's BICUBIC resize of one picture at a few sizes, up and down and
    unequal per axis, as PNGs under testdata/golden/minicpmv-pil/: 8-bit
    samples, which is what Pillow stores and what the Go resampler must match."""
    here = os.path.dirname(os.path.abspath(__file__))
    dst = os.path.join(here, "..", "testdata", "golden", "minicpmv-pil")
    os.makedirs(dst, exist_ok=True)
    img = Image.open(src).convert("RGB")
    for w, h in PIL_SIZES:
        img.resize((w, h), resample=Image.Resampling.BICUBIC).save(os.path.join(dst, f"{w}x{h}.png"))
    print("wrote", len(PIL_SIZES), "resizes of", src, "to", dst)


def main():
    if sys.argv[2:] == ["--layouts"]:
        return layouts(sys.argv[1])
    if sys.argv[1] == "--pil":
        return pil(sys.argv[2])
    d, images = sys.argv[1], sys.argv[2:]
    out = os.path.join(d, "gold")
    os.makedirs(out, exist_ok=True)
    cfg, vpm, res, proc = load(d)
    P = proc.patch_size
    for path in images:
        name = os.path.splitext(os.path.basename(path))[0]
        img = Image.open(path).convert("RGB")
        pieces = proc.get_sliced_images(img, proc.max_slice_nums)
        grid = proc.get_sliced_grid(img.size, proc.max_slice_nums)
        man = {"w": img.size[0], "h": img.size[1], "grid": grid, "pieces": []}
        for i, piece in enumerate(pieces):
            arr = np.array(piece).astype(np.float32) / 255
            arr = proc.normalize(arr, mean=proc.mean, std=proc.std, input_data_format="channels_last")
            arr = np.ascontiguousarray(arr.astype(np.float32))
            h, w = arr.shape[0], arr.shape[1]
            gh, gw = h // P, w // P
            chw = np.transpose(arr, (2, 0, 1))
            pv = torch.from_numpy(proc.reshape_by_patch(chw)).unsqueeze(0)  # [1, 3, P, n*P]
            tgt = torch.tensor([[gh, gw]], dtype=torch.int32)
            mask = torch.ones((1, 1, gh * gw), dtype=torch.bool)
            with torch.no_grad():
                vit = vpm(pv, patch_attention_mask=mask, tgt_sizes=tgt).last_hidden_state
                rso = res(vit, tgt)
            # The tower's position ids, as its embeddings computed them.
            B = vpm.embeddings.num_patches_per_side
            bounds = torch.arange(1 / B, 1.0, 1 / B)
            fh = torch.arange(0, 1 - 1e-6, 1 / tgt[0][0])
            fw = torch.arange(0, 1 - 1e-6, 1 / tgt[0][1])
            bh = torch.bucketize(fh, bounds, right=True)
            bw = torch.bucketize(fw, bounds, right=True)
            pos = (bh[:, None] * B + bw).flatten().tolist()
            base = os.path.join(out, f"{name}.p{i}")
            arr.tofile(base + ".px")
            vit[0].numpy().astype(np.float32).tofile(base + ".vit")
            rso[0].numpy().astype(np.float32).tofile(base + ".emb")
            man["pieces"].append({"w": w, "h": h, "gh": gh, "gw": gw, "pos": pos,
                                  "vit_sum": float(vit.sum()), "emb_sum": float(rso.sum())})
            print(f"{name} piece {i}: {w}x{h} ({gw}x{gh} patches) emb {tuple(rso.shape)} sum {float(rso.sum()):.4f}")
        json.dump(man, open(os.path.join(out, name + ".json"), "w"))


if __name__ == "__main__":
    main()
