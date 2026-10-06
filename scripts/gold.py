#!/usr/bin/env python3
"""Dump dequantization goldens from llama.cpp's libggml, once.

Run this by hand when a new quant type is added. The test suite reads only the
generated files, so it is hermetic: no .so at test time, and a llama.cpp upgrade
cannot silently change what "correct" means underneath a green build.

  JITLLM_LCPP=... JITLLM_MODELS=... python3 scripts/gold.py   # writes testdata/golden/dequant/*
"""
import ctypes, functools, glob, json, os, struct, sys

import refenv  # where things are: environment variables only (docs/testing.md)

LIB = (glob.glob(os.path.join(refenv.lcpp_bin(), "libggml-base.so*")) or
       sys.exit("gold.py: no libggml-base.so in $JITLLM_LCPP"))[0]
OUT = os.path.join(os.path.dirname(__file__), "..", "testdata", "golden", "dequant")
TINY = os.path.join(refenv.models(), "tinyllama-1.1b-q3_K_M.gguf")
GEMMA = os.path.join(refenv.models(), "gemma-2b.gguf")
MINILM = refenv.under("MINILM_GGUF", "MiniLM-L6-v2.Q8_0.gguf")
STORIES = os.path.join(os.path.dirname(__file__), "..", "models", "stories15M-q4_0.gguf")

BLK = {0:(1,4),1:(1,2),2:(32,18),3:(32,20),6:(32,22),7:(32,24),8:(32,34),9:(32,36),
       10:(256,84),11:(256,110),12:(256,144),13:(256,176),14:(256,210),20:(32,18),
       23:(256,136),24:(1,1),25:(1,2),26:(1,4),27:(1,8),28:(1,8),30:(1,2)}

class R:
    def __init__(s, b): s.b, s.o = b, 0
    def u32(s): v = struct.unpack_from('<I', s.b, s.o)[0]; s.o += 4; return v
    def u64(s): v = struct.unpack_from('<Q', s.b, s.o)[0]; s.o += 8; return v
    def st(s):
        n = s.u64(); v = s.b[s.o:s.o+n]; s.o += n; return v
    def skip(s, t):
        w = {0:1,1:1,7:1,2:2,3:2,4:4,5:4,6:4,10:8,11:8,12:8}.get(t)
        if w: s.o += w; return
        if t == 8: s.st(); return
        if t == 9:
            et = s.u32(); n = s.u64()
            w = {0:1,1:1,7:1,2:2,3:2,4:4,5:4,6:4,10:8,11:8,12:8}.get(et)
            if w: s.o += n*w
            elif et == 8:
                for _ in range(n): s.st()
            else: raise SystemExit("array of type %d" % et)
            return
        raise SystemExit("kv type %d" % t)

def tensors(path):
    b = open(path, 'rb').read()
    r = R(b); assert b[:4] == b'GGUF'; r.o = 4
    r.u32(); nt = r.u64(); nkv = r.u64()
    for _ in range(nkv):
        r.st(); r.skip(r.u32())
    out = {}
    for _ in range(nt):
        n = r.st().decode(); nd = r.u32(); d = [r.u64() for _ in range(nd)]
        t = r.u32(); off = r.u64()
        els = functools.reduce(lambda a, x: a*x, d, 1)
        be, bb = BLK[t]
        out[n] = (t, off, els//be*bb)
    ds = (r.o + 31)//32*32
    return b, ds, out

def main():
    if not os.path.exists(LIB): sys.exit("no libggml at %s" % LIB)
    lib = ctypes.CDLL(LIB)
    os.makedirs(OUT, exist_ok=True)

    # (label, ggml type, dequant symbol, model, tensor, block windows)
    jobs = [
        ("q4_0_gemma",  2, "dequantize_row_q4_0", GEMMA,  "blk.0.attn_q.weight",   [0, 977, 5001, 20011]),
        ("q8_0_gemma",  8, "dequantize_row_q8_0", GEMMA,  "token_embd.weight",     [0, 977, 5001, 20011]),
        ("q8_0_minilm", 8, "dequantize_row_q8_0", MINILM, "blk.0.attn_q.weight",   [0, 13, 101, 577]),
        ("q4_0_stories", 2, "dequantize_row_q4_0", STORIES, "blk.0.ffn_down.weight", [0, 37, 411, 1999]),
        ("q3_k_tiny",  11, "dequantize_row_q3_K", TINY, "token_embd.weight",     [0, 97, 1001, 40011]),
        ("q4_k_tiny",  12, "dequantize_row_q4_K", TINY, "*",                     [0, 97, 1001, 4011]),
        ("q5_k_tiny",  13, "dequantize_row_q5_K", TINY, "*",                     [0, 97, 1001, 4011]),
        ("q6_k_tiny",  14, "dequantize_row_q6_K", TINY, "output.weight",         [0, 97, 1001, 40011]),
        ("q8_0_stories", 8, "dequantize_row_q8_0", STORIES, "output.weight",         [0, 37, 411, 99999]),
    ]
    NBLK = 8  # blocks per window
    for label, ty, sym, path, tname, windows in jobs:
        if not os.path.exists(path):
            print("%-14s SKIP (no %s)" % (label, os.path.basename(path))); continue
        raw, ds, tt = tensors(path)
        if tname == "*":  # first tensor of this type that is big enough
            cands = [(n, v) for n, v in tt.items() if v[0] == ty and v[2] >= 64*BLK[ty][1]]
            if not cands: sys.exit("%s: no tensor of type %d" % (path, ty))
            tname = sorted(cands)[0][0]
        if tname not in tt: sys.exit("%s: no tensor %s" % (path, tname))
        t, off, nbytes = tt[tname]
        if t != ty: sys.exit("%s/%s is type %d, wanted %d" % (path, tname, t, ty))
        be, bb = BLK[ty]
        fn = getattr(lib, sym)
        fn.restype = None
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]

        ins, outs = bytearray(), bytearray()
        nblocks = nbytes // bb
        for w in windows:
            w %= max(nblocks - NBLK, 1)
            src = raw[ds + off + w*bb : ds + off + (w+NBLK)*bb]
            assert len(src) == NBLK*bb, (label, w, len(src))
            dst = (ctypes.c_float * (NBLK*be))()
            fn(ctypes.c_char_p(bytes(src)), dst, NBLK*be)
            ins += src
            outs += bytes(memoryview(dst).cast('B'))
        open(os.path.join(OUT, label + ".in"), 'wb').write(bytes(ins))
        open(os.path.join(OUT, label + ".f32"), 'wb').write(bytes(outs))
        print("%-12s %-22s %s  %d windows x %d blocks = %d elems" %
              (label, tname, os.path.basename(path)[:20], len(windows), NBLK, len(windows)*NBLK*be))


# ---------------------------------------------------------------------------
# Whole-tensor CRCs. The windowed goldens above are bit-exact but only cover a
# few hundred blocks; these cover every element of the largest tensor of each
# type — tens of millions — for the cost of one uint32 per tensor. CRC32-IEEE
# because zlib and Go's hash/crc32 agree on it byte for byte, and it is computed
# incrementally on both sides so neither has to hold the tensor in memory.

def crcs():
    import zlib, ctypes as C
    lib = C.CDLL(LIB)
    jobs = [
        (TINY,  "q3_k", 11, "dequantize_row_q3_K"),
        (TINY,  "q4_k", 12, "dequantize_row_q4_K"),
        (TINY,  "q5_k", 13, "dequantize_row_q5_K"),
        (TINY,  "q6_k", 14, "dequantize_row_q6_K"),
        (GEMMA, "q4_0",  2, "dequantize_row_q4_0"),
        (GEMMA, "q8_0",  8, "dequantize_row_q8_0"),
    ]
    out = []
    for path, label, ty, sym in jobs:
        if not os.path.exists(path):
            print("%-6s SKIP" % label); continue
        raw, ds, tt = tensors(path)
        cands = [(v[2], n) for n, v in tt.items() if v[0] == ty]
        if not cands: print("%-6s no tensor" % label); continue
        nbytes, tname = max(cands)
        t, off, _ = tt[tname]
        be, bb = BLK[ty]
        fn = getattr(lib, sym); fn.restype = None
        fn.argtypes = [C.c_void_p, C.POINTER(C.c_float), C.c_int64]
        nblocks = nbytes // bb
        # one chunk = 1024 blocks, so neither side ever holds the whole tensor
        CH = 1024
        buf = (C.c_float * (CH * be))()
        crc = 0
        done = 0
        while done < nblocks:
            n = min(CH, nblocks - done)
            src = raw[ds + off + done*bb : ds + off + (done+n)*bb]
            fn(C.c_char_p(src), buf, n * be)
            crc = zlib.crc32(memoryview(buf).cast('B')[:n*be*4], crc)
            done += n
        out.append({"model": os.path.basename(path), "type": label,
                    "tensor": tname, "elems": nblocks*be, "crc32": crc})
        print("%-6s %-26s %12d elems  crc %08x" % (label, tname, nblocks*be, crc))
    p = os.path.join(OUT, "..", "tensor_crc.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=1)

if __name__ == "__main__":
    main()
    crcs()
