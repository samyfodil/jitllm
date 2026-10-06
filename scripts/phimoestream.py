#!/usr/bin/env python3
"""transformers' own PhimoeDecoderLayer, run on a GGUF's weights.

    python scripts/phimoestream.py model.gguf hf-config-dir outdir id id ...

The reference for Phi-3.5-MoE, whose llama.cpp graph is not the model's
(an RMSNorm with a bias for its LayerNorm, a renormalised softmax top-2 for its
sparsemixer router): each decoder layer is built from transformers'
PhimoeDecoderLayer, loaded with gguf-py's dequantization of the SAME file the
engine runs (so quantization is not a variable), run over every row in float32
in eval mode, and freed -- one layer at a time, as scripts/hunyuanstream.py
does for Hunyuan-A13B. It prints the top logits of every row and saves them
(logits.pt) for a teacher-forced comparison against TestTeacherForcedTrace.

hf-config-dir holds microsoft/Phi-3.5-MoE-instruct's config.json. Its
short_mscale (1.243) is the class's cos/sin scale; the GGUF carries
llama.cpp's own attn_factor instead (1.190), so MSCALE=file replaces it with
the file's to compare the graph alone.

  GGUF_PY=<llama.cpp>/gguf-py   where gguf-py lives
  THREADS=n                     torch's thread count
  MSCALE=file                   the GGUF's rope.scaling.attn_factor for both mscales
"""
import sys, os
import numpy as np
import torch
sys.path.insert(0, os.environ.get('GGUF_PY', 'gguf-py'))
from gguf import GGUFReader
from gguf.quants import dequantize
from transformers import AutoConfig
from transformers.models.phimoe import modeling_phimoe as M

torch.set_num_threads(int(os.environ.get('THREADS', '16')))
path, cfgdir, out = sys.argv[1], sys.argv[2], sys.argv[3]
ids = [int(x) for x in sys.argv[4:]]
os.makedirs(out, exist_ok=True)
cfg = AutoConfig.from_pretrained(cfgdir)
cfg._attn_implementation = 'eager'
cfg.torch_dtype = torch.float32

r = GGUFReader(path)
T = {t.name: t for t in r.tensors}
if os.environ.get('MSCALE') == 'file':
    f = r.fields['phimoe.rope.scaling.attn_factor']
    af = float(f.parts[f.data[0]][0])
    cfg.rope_parameters['short_mscale'] = cfg.rope_parameters['long_mscale'] = af
print(type(cfg).__name__, cfg.rope_parameters.get('rope_type'), cfg.rope_parameters.get('short_mscale'),
      cfg.router_jitter_noise, flush=True)


def W(name):
    t = T[name]
    a = dequantize(t.data, t.tensor_type)
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


n = len(ids)
emb = W('token_embd.weight')
h = emb[torch.tensor(ids)].unsqueeze(0).clone()
rot = M.PhimoeRotaryEmbedding(cfg)
pos = torch.arange(n).unsqueeze(0)
cos, sin = rot(h, pos)
mask = torch.full((n, n), float('-inf')).triu(1)[None, None]
with torch.no_grad():
    for il in range(cfg.num_hidden_layers):
        L = M.PhimoeDecoderLayer(cfg, il).float().eval()
        p = f'blk.{il}.'
        sd = {
            'input_layernorm.weight': W(p + 'attn_norm.weight'),
            'input_layernorm.bias': W(p + 'attn_norm.bias'),
            'post_attention_layernorm.weight': W(p + 'ffn_norm.weight'),
            'post_attention_layernorm.bias': W(p + 'ffn_norm.bias'),
            'self_attn.q_proj.weight': W(p + 'attn_q.weight'),
            'self_attn.q_proj.bias': W(p + 'attn_q.bias'),
            'self_attn.k_proj.weight': W(p + 'attn_k.weight'),
            'self_attn.k_proj.bias': W(p + 'attn_k.bias'),
            'self_attn.v_proj.weight': W(p + 'attn_v.weight'),
            'self_attn.v_proj.bias': W(p + 'attn_v.bias'),
            'self_attn.o_proj.weight': W(p + 'attn_output.weight'),
            'self_attn.o_proj.bias': W(p + 'attn_output.bias'),
            'mlp.router.weight': W(p + 'ffn_gate_inp.weight'),
            'mlp.experts.gate_up_proj': torch.cat([W(p + 'ffn_gate_exps.weight'), W(p + 'ffn_up_exps.weight')], dim=1),
            'mlp.experts.down_proj': W(p + 'ffn_down_exps.weight'),
        }
        L.load_state_dict(sd, strict=True)
        h = L(h, position_embeddings=(cos, sin), attention_mask=mask)
        if isinstance(h, tuple):
            h = h[0]
        print('layer', il, 'done', float(h[0, -1].norm()), flush=True)
        del L, sd
    hn = torch.nn.functional.layer_norm(h, (cfg.hidden_size,), W('output_norm.weight'),
                                        W('output_norm.bias'), cfg.rms_norm_eps)
    logits = hn[0] @ W('output.weight').T + W('output.bias')
    torch.save(logits, f'{out}/logits.pt')
    for i in range(n):
        v, ix = logits[i].topk(8)
        print('row', i, ' | '.join(f'{int(j)} {float(x):.4f}' for x, j in zip(v, ix)), flush=True)
