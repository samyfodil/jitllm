#!/usr/bin/env python3
"""Write format/quant/testdata/q5_1.json: Q5_1 blocks and gguf-py's decoding of them.

The oracle for quant's Q5_1 decoder is an implementation that is not ours --
the pip `gguf` package's Q5_1.dequantize_blocks. Run it with the HF reference
venv:

    $JITLLM_HF_PY scripts/q51gold.py [model.gguf]

Blocks: every 5-bit quant in every position (a rotation of 0..31 through the
32 slots, so each qh bit is both set and clear somewhere), at a sweep of d and m
that reaches subnormal, the largest finite half and negative minimums, plus the
first blocks of a real Q5_1 tensor when a GGUF is given (falcon's attn_qkv).
"""
import json
import struct
import sys

import numpy as np
from gguf import GGUFReader
from gguf.quants import Q5_1

halves = [0x3C00, 0x3800, 0x0001, 0x03FF, 0x0400, 0x1000, 0x2E00, 0x7BFF, 0x5000, 0xBC00, 0xB800, 0x8001, 0xC400, 0x0000]
blocks = []
for i, d in enumerate(halves):
    for j, m in enumerate(halves):
        rot = (i * 7 + j * 3) % 32
        q = [(l + rot) % 32 for l in range(32)]
        qh = 0
        for l in range(32):
            if q[l] & 16:
                qh |= 1 << l
        qs = bytes((q[l] & 15) | ((q[l + 16] & 15) << 4) for l in range(16))
        blocks.append(struct.pack("<HHI", d, m, qh) + qs)

if len(sys.argv) > 1:
    r = GGUFReader(sys.argv[1])
    t = next(t for t in r.tensors if t.tensor_type.name == "Q5_1")
    raw = bytes(np.asarray(t.data).reshape(-1).view(np.uint8)[: 24 * 128])
    blocks += [raw[i : i + 24] for i in range(0, len(raw), 24)]
    src = t.name
else:
    src = "synthetic only"

arr = np.frombuffer(b"".join(blocks), dtype=np.uint8).reshape(-1, 24)
with np.errstate(over="ignore"):
    vals = Q5_1.dequantize_blocks(arr).astype(np.float32).reshape(-1)
out = {
    "source": "gguf-py Q5_1.dequantize_blocks; real blocks from " + src,
    "blocks": b"".join(blocks).hex(),
    "bits": [int(v) for v in vals.view(np.uint32)],
}
json.dump(out, open("format/quant/testdata/q5_1.json", "w"))
print(len(blocks), "blocks")
