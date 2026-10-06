#!/usr/bin/env python3
"""transformers' own HunYuanMoEV1 decoder layers, run on a GGUF's weights.

    python scripts/hunyuanstream.py model.gguf hf-config-dir outdir id id ...

The reference for a Hunyuan mixture too large to load whole: each decoder layer
is built from transformers' HunYuanMoEV1DecoderLayer, loaded with gguf-py's
dequantization of the SAME file the engines run (so quantization is not a
variable), run over every row in float32, and freed -- ~15 GB at a time for
Hunyuan-A13B where the whole model is 160 GB in bf16. It prints the top logits
of every row and saves each layer's attn_resid/ffn_resid and router selection
(torch.save) for a layer-by-layer diff against TestTeacherForcedTrace.

hf-config-dir holds the model's config.json (tencent/Hunyuan-A13B-Instruct's).
The layers run in eval mode: a fresh module is in training mode, and the
config's attention_dropout of 0.1 then drops a tenth of every attention row.

  GGUF_PY=<llama.cpp>/gguf-py   where gguf-py lives
  THREADS=n                     torch's thread count
  ROUND=llama|jit               round K and V through f16 as a cache would:
                                llama the weighted key, jit the unweighted one
                                with the weight applied after (the engine's fold)
  ACTQ=1                        Q8_0 activations into every linear (absmax/127
                                per 32), as the engines' quantized matvecs take
"""
import sys, os, json
import numpy as np
import torch
sys.path.insert(0, os.environ.get('GGUF_PY', 'gguf-py'))
from gguf import GGUFReader
from gguf.quants import dequantize
from transformers import AutoConfig
from transformers.models.hunyuan_v1_moe import modeling_hunyuan_v1_moe as M

torch.set_num_threads(int(os.environ.get('THREADS', '16')))
path, cfgdir, out = sys.argv[1], sys.argv[2], sys.argv[3]
ids = [int(x) for x in sys.argv[4:]]
os.makedirs(out, exist_ok=True)
cfg = AutoConfig.from_pretrained(cfgdir)
cfg._attn_implementation = 'eager'
cfg.torch_dtype = torch.float32
print(type(cfg).__name__, cfg.rope_parameters, cfg.num_local_experts if hasattr(cfg, 'num_local_experts') else None, flush=True)

r = GGUFReader(path)
T = {t.name: t for t in r.tensors}

def W(name):
    t = T[name]
    a = dequantize(t.data, t.tensor_type)
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))

n = len(ids)

if os.environ.get('ACTQ') == '1':
    # Q8_0 activations, as the engines' quantized matvecs take them: blocks of
    # 32 along the input, scale absmax/127, codes rounded to nearest.
    _lin = torch.nn.functional.linear

    def qlinear(x, w, b=None):
        k = x.shape[-1]
        if k % 32 == 0:
            xb = x.reshape(*x.shape[:-1], k // 32, 32)
            d = xb.abs().amax(-1, keepdim=True) / 127
            q = torch.where(d > 0, torch.round(xb / torch.where(d > 0, d, torch.ones_like(d))), torch.zeros_like(xb))
            x = (q * d).reshape(x.shape)
        return _lin(x, w, b)
    torch.nn.functional.linear = qlinear


class KRound(torch.nn.Module):
    # the key as an f16 cache holds it: 'llama' rounds the weighted key,
    # 'jit' the unweighted one with the weight applied after (the fold)
    def __init__(self, kn, mode):
        super().__init__()
        self.kn, self.mode = kn, mode

    def forward(self, x):
        if self.mode == 'llama':
            return self.kn(x).half().float()
        xf = x.float()
        u = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.kn.variance_epsilon)
        return self.kn.weight * u.half().float()


class VRound(torch.nn.Module):
    def __init__(self, vp):
        super().__init__()
        self.vp = vp

    def forward(self, x):
        return self.vp(x).half().float()

emb = W('token_embd.weight')  # [vocab, hidden]
h = emb[torch.tensor(ids)].unsqueeze(0).clone()
rot = M.HunYuanMoEV1RotaryEmbedding(cfg)
pos = torch.arange(n).unsqueeze(0)
cos, sin = rot(h, pos)
mask = torch.full((n, n), float('-inf')).triu(1)[None, None]
torch.save(h[0], f'{out}/embd.pt')
with torch.no_grad():
    for il in range(cfg.num_hidden_layers):
        L = M.HunYuanMoEV1DecoderLayer(cfg, il).float().eval()
        p = f'blk.{il}.'
        sd = {
            'input_layernorm.weight': W(p + 'attn_norm.weight'),
            'post_attention_layernorm.weight': W(p + 'ffn_norm.weight'),
            'self_attn.q_proj.weight': W(p + 'attn_q.weight'),
            'self_attn.k_proj.weight': W(p + 'attn_k.weight'),
            'self_attn.v_proj.weight': W(p + 'attn_v.weight'),
            'self_attn.o_proj.weight': W(p + 'attn_output.weight'),
            'self_attn.query_layernorm.weight': W(p + 'attn_q_norm.weight'),
            'self_attn.key_layernorm.weight': W(p + 'attn_k_norm.weight'),
            'mlp.gate.wg.weight': W(p + 'ffn_gate_inp.weight'),
            'mlp.shared_mlp.gate_proj.weight': W(p + 'ffn_gate_shexp.weight'),
            'mlp.shared_mlp.up_proj.weight': W(p + 'ffn_up_shexp.weight'),
            'mlp.shared_mlp.down_proj.weight': W(p + 'ffn_down_shexp.weight'),
            'mlp.experts.gate_up_proj': torch.cat([W(p + 'ffn_gate_exps.weight'), W(p + 'ffn_up_exps.weight')], dim=1),
            'mlp.experts.down_proj': W(p + 'ffn_down_exps.weight'),
        }
        L.load_state_dict(sd, strict=True)
        rnd = os.environ.get('ROUND', 'none')
        if rnd != 'none':
            L.self_attn.key_layernorm = KRound(L.self_attn.key_layernorm, rnd)
            L.self_attn.v_proj = VRound(L.self_attn.v_proj)
        resid = h
        x = L.input_layernorm(h)
        a, _ = L.self_attn(hidden_states=x, position_embeddings=(cos, sin), attention_mask=mask)
        h = resid + a
        torch.save(h[0], f'{out}/attn_resid.{il}.pt')
        resid = h
        x = L.post_attention_layernorm(h)
        # the router's selection, for every row
        logits_r, w, sel = L.mlp.gate(x.view(-1, x.shape[-1]))
        torch.save({'router': logits_r, 'w': w, 'sel': sel}, f'{out}/route.{il}.pt')
        h = resid + L.mlp(x)
        torch.save(h[0], f'{out}/ffn_resid.{il}.pt')
        print('layer', il, 'done', float(h[0, -1].norm()), flush=True)
        del L, sd
    norm_w = W('output_norm.weight')
    var = h.pow(2).mean(-1, keepdim=True)
    hn = norm_w * (h * torch.rsqrt(var + cfg.rms_norm_eps))
    logits = hn[0] @ emb.T
    torch.save(logits, f'{out}/logits.pt')
    for i in range(n):
        v, ix = logits[i].topk(8)
        print('row', i, ' | '.join(f'{int(j)} {float(x):.4f}' for x, j in zip(v, ix)), flush=True)
