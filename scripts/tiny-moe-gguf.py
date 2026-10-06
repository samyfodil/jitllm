#!/usr/bin/env python3
"""Build a tiny qwen3moe GGUF, so the MoE path has an oracle loop that runs in
seconds instead of minutes.

Qwen3-30B-A3B is 18.5 GB. Every correctness iteration against it costs a model
load, and on this box the file does not comfortably fit in page cache -- which
makes the loop slow exactly when it needs to be fast, because a MoE routing bug
does not crash, it answers fluently and wrong (AGENTS.md RULE 7a).

trl-internal-testing/tiny-Qwen3MoeForCausalLM is a real qwen3_moe: 2 layers,
n_embd 8, 4 experts, 2 active, moe_intermediate_size 768, norm_topk_prob true,
q/k per-head norms, no shared experts. Random weights, so its TEXT is nonsense
and it is useless as a quality check -- but it exercises every line of the graph
and, run under both engines, its LOGITS are a perfectly good oracle.

  scripts/tiny-moe-gguf.py <hf-dir> <out.gguf> [--vocab-from <donor.gguf>]

The tokenizer is copied wholesale from a real Qwen3 GGUF rather than rebuilt
from tokenizer.json: the two engines must agree on tokenization before they can
be compared on anything else, and copying is the only way to be sure they do.

llama.cpp's own convert_hf_to_gguf.py would do all of this, but it imports torch
at module scope and torch is 2.5 GB on a box whose home partition has 29 GB
free. safetensors is a length-prefixed JSON header over raw arrays, so reading
it here is thirty lines and no dependency.
"""
import json, struct, sys, os
import numpy as np

# gguf-py's package __init__ pulls in vocab.py, which imports sentencepiece for
# a code path this script never touches. Stubbing it is cheaper and safer than
# --break-system-packages on the system interpreter.
sys.modules.setdefault('sentencepiece', type(sys)('sentencepiece'))
sys.modules['sentencepiece'].SentencePieceProcessor = object
GGUF_PY = os.environ.get('GGUF_PY') or os.path.join(os.environ.get('LCPP_SRC') or sys.exit(
    'tiny-moe-gguf.py: set GGUF_PY to llama.cpp\'s gguf-py, or LCPP_SRC to a llama.cpp source checkout'), 'gguf-py')
sys.path.insert(0, GGUF_PY)
from gguf.gguf_reader import GGUFReader          # noqa: E402
from gguf.gguf_writer import GGUFWriter          # noqa: E402
from gguf.constants import GGMLQuantizationType, GGUFValueType  # noqa: E402


def field_value(f):
    """Decode a ReaderField. This gguf-py has no contents() helper, and the
    parts/data indirection is the whole format: parts holds every raw chunk the
    parser saw and data indexes the ones that are the value."""
    def one(i):
        part = f.parts[i]
        if f.types[-1] == GGUFValueType.STRING:
            return bytes(part).decode('utf-8', 'replace')
        v = part.tolist()
        return v[0] if isinstance(v, list) else v
    if f.types[0] == GGUFValueType.ARRAY:
        return [one(i) for i in f.data]
    return one(f.data[0])


def read_safetensors(path):
    """Yield (name, float32 array) in torch [out, in] order."""
    with open(path, 'rb') as f:
        hdr = json.loads(f.read(struct.unpack('<Q', f.read(8))[0]))
        base = f.tell()
        hdr.pop('__metadata__', None)
        for name, meta in hdr.items():
            lo, hi = meta['data_offsets']
            f.seek(base + lo)
            raw = f.read(hi - lo)
            dt = meta['dtype']
            if dt == 'BF16':
                # bfloat16 is the top 16 bits of a float32, so widening is a
                # shift rather than a conversion table.
                u = np.frombuffer(raw, dtype='<u2').astype(np.uint32) << 16
                arr = u.view(np.float32)
            elif dt in ('F16', 'FP16'):
                arr = np.frombuffer(raw, dtype='<f2').astype(np.float32)
            elif dt in ('F32', 'FP32'):
                arr = np.frombuffer(raw, dtype='<f4')
            else:
                raise SystemExit(f'{name}: dtype {dt} not handled')
            yield name, arr.reshape(meta['shape']).copy()


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    hf, out = sys.argv[1], sys.argv[2]
    donor = None
    if '--vocab-from' in sys.argv:
        donor = sys.argv[sys.argv.index('--vocab-from') + 1]

    cfg = json.load(open(os.path.join(hf, 'config.json')))
    if cfg['model_type'] != 'qwen3_moe':
        raise SystemExit(f"model_type is {cfg['model_type']}, want qwen3_moe")
    n_layer, n_expert = cfg['num_hidden_layers'], cfg['num_experts']
    ten = dict(read_safetensors(os.path.join(hf, 'model.safetensors')))

    w = GGUFWriter(out, 'qwen3moe')
    w.add_name('tiny-qwen3moe')
    w.add_block_count(n_layer)
    w.add_context_length(cfg['max_position_embeddings'])
    w.add_embedding_length(cfg['hidden_size'])
    w.add_feed_forward_length(cfg['intermediate_size'])
    w.add_head_count(cfg['num_attention_heads'])
    w.add_head_count_kv(cfg['num_key_value_heads'])
    w.add_key_length(cfg['head_dim'])
    w.add_value_length(cfg['head_dim'])
    w.add_rope_freq_base(cfg['rope_theta'])
    w.add_layer_norm_rms_eps(cfg['rms_norm_eps'])
    w.add_expert_count(n_expert)
    w.add_expert_used_count(cfg['num_experts_per_tok'])
    w.add_expert_feed_forward_length(cfg['moe_intermediate_size'])
    w.add_file_type(GGMLQuantizationType.F32)

    if donor:
        # Copy the tokenizer verbatim. Rebuilding it from tokenizer.json would
        # be a second implementation of something that only has to MATCH.
        r = GGUFReader(donor)
        copied = 0
        for k, f in r.fields.items():
            if not k.startswith('tokenizer.'):
                continue
            try:
                val = field_value(f)
                if f.types[0] == GGUFValueType.ARRAY:
                    w.add_array(k, val)
                else:
                    w.add_key_value(k, val, f.types[0])
                copied += 1
            except Exception as e:
                print(f'  skip {k}: {e}', file=sys.stderr)
        print(f'copied {copied} tokenizer keys from {os.path.basename(donor)}')

    def put(name, arr):
        w.add_tensor(name, np.ascontiguousarray(arr))

    put('token_embd.weight', ten['model.embed_tokens.weight'])
    put('output_norm.weight', ten['model.norm.weight'])
    if 'lm_head.weight' in ten:
        put('output.weight', ten['lm_head.weight'])
    for i in range(n_layer):
        p, b = f'model.layers.{i}.', f'blk.{i}.'
        put(b + 'attn_norm.weight', ten[p + 'input_layernorm.weight'])
        put(b + 'ffn_norm.weight', ten[p + 'post_attention_layernorm.weight'])
        for hf_n, gg in (('q_proj', 'attn_q'), ('k_proj', 'attn_k'),
                         ('v_proj', 'attn_v'), ('o_proj', 'attn_output'),
                         ('q_norm', 'attn_q_norm'), ('k_norm', 'attn_k_norm')):
            put(f'{b}{gg}.weight', ten[f'{p}self_attn.{hf_n}.weight'])
        put(b + 'ffn_gate_inp.weight', ten[p + 'mlp.gate.weight'])
        # The experts stack into one 3D tensor per role. numpy (n_expert, out,
        # in) becomes ne (in, out, n_expert), which is what makes expert e a
        # contiguous byte range -- the property jitllm's row slicing depends on.
        for hf_n, gg in (('gate_proj', 'ffn_gate_exps'), ('up_proj', 'ffn_up_exps'),
                         ('down_proj', 'ffn_down_exps')):
            put(f'{b}{gg}.weight',
                np.stack([ten[f'{p}mlp.experts.{e}.{hf_n}.weight'] for e in range(n_expert)]))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f'wrote {out}  {os.path.getsize(out)/1e6:.1f} MB  '
          f'{n_layer} layers, {n_expert} experts, {cfg["num_experts_per_tok"]} active')


main()
