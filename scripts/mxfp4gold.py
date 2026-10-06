#!/usr/bin/env python3
"""Write format/quant/testdata/mxfp4.json: MXFP4 blocks and gguf-py's decoding of them.

The oracle for quant's MXFP4 decoder is an implementation that is not ours --
the pip `gguf` package, which transcribes ggml_e8m0_to_fp32_half and
kvalues_mxfp4 independently. Run it with the HF reference venv:

    $JITLLM_HF_PY scripts/mxfp4gold.py [model.gguf]

Blocks: every nibble in both halves at a sweep of exponents including the two
subnormal ones (0, 1) and the top of the range, plus the first blocks of a real
gpt-oss expert bank when a GGUF is given.
"""
import json
import sys

import numpy as np
from gguf import GGUFReader
from gguf.quants import MXFP4

blocks = []
for e in (0, 1, 2, 3, 100, 114, 115, 120, 127, 128, 129, 136, 143, 200, 253, 254, 255):
    for rot in range(16):
        qs = bytes(((i + rot) % 16) | (((i + rot + 7) % 16) << 4) for i in range(16))
        blocks.append(bytes([e]) + qs)

if len(sys.argv) > 1:
    r = GGUFReader(sys.argv[1])
    t = next(t for t in r.tensors if t.name == "blk.0.ffn_down_exps.weight")
    raw = bytes(np.asarray(t.data).reshape(-1).view(np.uint8)[: 17 * 64])
    blocks += [raw[i : i + 17] for i in range(0, len(raw), 17)]

arr = np.frombuffer(b"".join(blocks), dtype=np.uint8).reshape(-1, 17)
with np.errstate(over="ignore"):
    vals = MXFP4.dequantize_blocks(arr).astype(np.float32).reshape(-1)
out = {
    "source": "gguf-py MXFP4.dequantize_blocks",
    "blocks": b"".join(blocks).hex(),
    "bits": [int(v) for v in vals.view(np.uint32)],
}
json.dump(out, open("format/quant/testdata/mxfp4.json", "w"))
print(len(blocks), "blocks")
