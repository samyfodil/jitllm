#!/usr/bin/env python3
"""Build the state-space hybrid fixtures and record transformers' logits.

    $JITLLM_HF_PY scripts/ssmgold.py [name ...]
    JITLLM_MODELS=/path/to/models ... (where the GGUFs go)

-> $JITLLM_MODELS/<name>.gguf and testdata/golden/ssm/<name>.json

moegold.py's method for the Mamba-2 families: the oracle is transformers' OWN
class with every parameter randomised, the GGUF is written by llama.cpp's OWN
converter, and every tensor is F32. The golden is the class's full-sequence
forward, which runs Mamba-2's chunked SSD scan -- a different algorithm from
the recurrence every engine steps, so agreement is the recurrence checked
against the scan, not against itself.

★ dt is kept clear of time_step_limit. transformers clamps dt to it on a prompt
and not on a decode step, llama.cpp never clamps and the GGUF does not carry
it (AGENTS.md RULE 7m, model-coverage.md 4c): the dt bias is drawn so that
softplus(dt) never reaches a clamp, which makes all four agree on the fixture.

★ A Mamba-2 class whose gated norm is NOT grouped (Mamba2, GraniteMoeHybrid)
is built with one group, where grouped and ungrouped are the same arithmetic;
the grouping itself is held by the Nemotron-H fixture, whose class groups.
"""
import json
import os
import shutil
import subprocess
import sys

import torch
from tokenizers import Tokenizer

import refenv  # where things are: environment variables only (docs/testing.md)

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = refenv.models()
TOKDIR = refenv.under("CV_TOK", "tok")
HFDIR = refenv.under("CV_HF", "hf")
LCPP = refenv.need("LCPP_SRC", "a llama.cpp source checkout")
PY = sys.executable
OUT = os.path.join(HERE, "..", "testdata", "golden", "ssm")
NPOS, NLOGIT = 12, 512
TEXTS = [
    "The capital of France is",
    "def getUserName(self):\n    return 12345",
    " leading space and  double  spaces\nline two\ttab",
]


def mamba2():
    from transformers import Mamba2Config, Mamba2ForCausalLM
    # Mamba-2's own block at toy size: 8 heads of 16 over a 128-wide inner
    # (expand 2), a 16-wide state, one group, a 4-tap convolution with a bias,
    # no FFN, an untied head.
    cfg = Mamba2Config(vocab_size=49152, hidden_size=64, state_size=16, num_heads=8, head_dim=16,
                       expand=2, n_groups=1, conv_kernel=4, num_hidden_layers=3, use_conv_bias=True,
                       use_bias=False, layer_norm_epsilon=1e-5, tie_word_embeddings=False,
                       residual_in_fp32=True, chunk_size=256)
    return Mamba2ForCausalLM, cfg, "starcoder", None


def granitehybrid():
    from transformers import GraniteMoeHybridConfig, GraniteMoeHybridForCausalLM
    # Granite 4.0-H's shape: Mamba-2 blocks with one attention block among
    # them (NoPE, GQA 4:2), each followed by the shared SwiGLU MLP, and
    # Granite's four scales at values far from one. A dense one states its MLP
    # width as intermediate_size too, which llama.cpp loads the FFN at.
    cfg = GraniteMoeHybridConfig(
        vocab_size=49152, hidden_size=64, intermediate_size=96, shared_intermediate_size=96,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, num_local_experts=0,
        layer_types=["mamba", "attention", "mamba", "mamba"], position_embedding_type="nope",
        mamba_n_heads=8, mamba_d_head=16, mamba_d_state=16, mamba_n_groups=1, mamba_expand=2,
        mamba_d_conv=4, mamba_conv_bias=True, mamba_proj_bias=False, rms_norm_eps=1e-5,
        embedding_multiplier=6.0, residual_multiplier=0.35, attention_multiplier=0.09,
        logits_scaling=3.5, tie_word_embeddings=False, max_position_embeddings=4096)
    return GraniteMoeHybridForCausalLM, cfg, "starcoder", granite_layer_types


def granitehybridmoe():
    from transformers import GraniteMoeHybridConfig, GraniteMoeHybridForCausalLM
    # Granite 4.0-H-Tiny's shape: the same blocks with a softmax top-2 of 8
    # beside the shared MLP, tied embeddings as the Tiny ships.
    cfg = GraniteMoeHybridConfig(
        vocab_size=49152, hidden_size=64, intermediate_size=32, shared_intermediate_size=96,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, num_local_experts=8,
        num_experts_per_tok=2, layer_types=["mamba", "mamba", "attention", "mamba"],
        position_embedding_type="nope", mamba_n_heads=8, mamba_d_head=16, mamba_d_state=16,
        mamba_n_groups=1, mamba_expand=2, mamba_d_conv=4, mamba_conv_bias=True,
        mamba_proj_bias=False, rms_norm_eps=1e-5, embedding_multiplier=6.0,
        residual_multiplier=0.35, attention_multiplier=0.09, logits_scaling=3.5,
        tie_word_embeddings=True, max_position_embeddings=4096)
    return GraniteMoeHybridForCausalLM, cfg, "starcoder", granite_layer_types


def granite_layer_types(m, cfg, d):
    """Write the layer types in the published files' own names ("mamba",
    "attention"), which llama.cpp's converter reads; transformers saves its
    remapped ones ("linear_attention", "full_attention")."""
    c = json.load(open(os.path.join(d, "config.json")))
    c["layer_types"] = ["mamba" if t == "linear_attention" else "attention" for t in cfg.layer_types]
    json.dump(c, open(os.path.join(d, "config.json"), "w"), indent=2)


NEMOTRON_PATTERN = "M-M*-MM-"


def nemotronh():
    from transformers import NemotronHConfig, NemotronHForCausalLM
    # Nemotron Nano 2's shape at toy size: one mixer a layer -- Mamba-2 (M),
    # attention with no rotary (*) and the squared-ReLU MLP (-) -- in a pattern
    # that has a mixer with its MLP, a mixer alone before another mixer, the
    # attention with its MLP and two Mamba-2 layers in a row. Two groups, so
    # the gated norm is grouped (the class's Zamba2RMSNormGated groups it) and
    # a head reads its group's B and C.
    types = {"M": "linear_attention", "*": "full_attention", "-": "mlp"}
    cfg = NemotronHConfig(vocab_size=49152, hidden_size=64, layers_block_type=[types[c] for c in NEMOTRON_PATTERN],
                          num_attention_heads=4, num_key_value_heads=2, head_dim=16, intermediate_size=96,
                          mlp_hidden_act="relu2", ssm_state_size=16, mamba_num_heads=8, mamba_head_dim=16,
                          n_groups=2, conv_kernel=4, expand=2, use_conv_bias=True, mamba_proj_bias=False,
                          layer_norm_epsilon=1e-5, tie_word_embeddings=False, max_position_embeddings=4096,
                          time_step_limit=(0.0, float("inf")), num_nextn_predict_layers=0,
                          # The class clamps a prompt's dt at time_step_min (its
                          # own (time_step_min, inf) limit) where no engine
                          # clamps (see the docstring): kept far below any dt.
                          time_step_min=1e-30)
    return NemotronHForCausalLM, cfg, "starcoder", nemotron_pattern


def nemotron_pattern(m, cfg, d, pattern=None):
    """State the layers as the published files do, a hybrid_override_pattern
    string, which llama.cpp's converter reads."""
    c = json.load(open(os.path.join(d, "config.json")))
    for k in ("layers_block_type", "layer_types"):
        c.pop(k, None)
    c["hybrid_override_pattern"] = pattern or NEMOTRON_PATTERN
    json.dump(c, open(os.path.join(d, "config.json"), "w"), indent=2)


NEMOTRON_MOE_PATTERN = "ME*EMME"


def nemotronhmoe():
    from transformers import NemotronHConfig, NemotronHForCausalLM
    # Nemotron 3 Nano's shape at toy size: the mixers of nemotronh() with the
    # squared-ReLU MLP replaced by a mixture (E) -- eight ungated experts (up
    # and down, no gate), two routed under DeepSeek-V3's sigmoid router with a
    # selection bias, renormalised and scaled by 2.5, beside one ungated
    # shared expert. One routing group, as every published config has and as
    # llama.cpp (which does not group) computes.
    types = {"M": "linear_attention", "*": "full_attention", "E": "moe"}
    cfg = NemotronHConfig(vocab_size=49152, hidden_size=64,
                          layers_block_type=[types[c] for c in NEMOTRON_MOE_PATTERN],
                          num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                          mlp_hidden_act="relu2", ssm_state_size=16, mamba_num_heads=8, mamba_head_dim=16,
                          n_groups=2, conv_kernel=4, expand=2, use_conv_bias=True, mamba_proj_bias=False,
                          layer_norm_epsilon=1e-5, tie_word_embeddings=False, max_position_embeddings=4096,
                          time_step_limit=(0.0, float("inf")), num_nextn_predict_layers=0,
                          time_step_min=1e-30, n_routed_experts=8, num_experts_per_tok=2,
                          moe_intermediate_size=64, moe_shared_expert_intermediate_size=96,
                          n_shared_experts=1, routed_scaling_factor=2.5, norm_topk_prob=True,
                          n_group=1, topk_group=1)
    return NemotronHForCausalLM, cfg, "starcoder", lambda m, c, d: nemotron_pattern(m, c, d, NEMOTRON_MOE_PATTERN)


# llama.cpp's converter reads a config through AutoConfig, and this
# transformers' NemotronHConfig gets in its way twice: it fills in its mixture
# defaults (num_experts_per_tok among them) for a dense model too, which the
# converter takes as the mark of nemotron_h_moe, and it turns the published
# hybrid_override_pattern string into a list of names in which the converter
# finds no dense MLP layer ("mlp" is not in its _MLP_LAYER_TYPES). A dense
# Nemotron-H's published config carries no mixture keys and states the
# pattern as a string; the shim hands the converter exactly that.
DENSE_NEMOTRON = """
import json, os, runpy, sys
sys.path.insert(0, sys.argv[1])
from conversion import base
orig = base.ModelBase.load_hparams
def load(d, is_mistral_format=False):
    c = orig(d, is_mistral_format)
    for k in [k for k in c if k.startswith("moe_") or k in ("num_experts_per_tok", "n_routed_experts",
              "n_shared_experts", "routed_scaling_factor", "n_group", "topk_group", "norm_topk_prob",
              "layers_block_type", "layer_types")]:
        c.pop(k)
    c["hybrid_override_pattern"] = json.load(open(os.path.join(d, "config.json")))["hybrid_override_pattern"]
    c["num_hidden_layers"] = len(c["hybrid_override_pattern"])
    return c
base.ModelBase.load_hparams = staticmethod(load)
script = sys.argv[1] + "/convert_hf_to_gguf.py"
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
"""


def falconh1(rms_norm=True, groups=2):
    from transformers import FalconH1Config, FalconH1ForCausalLM
    # Falcon-H1's block at toy size: attention (NEOX rotary, GQA 4:2) and a
    # Mamba-2 mixer side by side on one normed input, then a SwiGLU MLP, with
    # every muP multiplier far from one -- llama.cpp's converter folds them
    # into the weights, so a multiplier it drops is a wrong logit here. The
    # gated norm's order (gate first) is every published config's.
    cfg = FalconH1Config(
        vocab_size=49152, hidden_size=64, intermediate_size=96, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, mamba_d_ssm=128, mamba_n_heads=8,
        mamba_d_head=16, mamba_n_groups=groups, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
        mamba_conv_bias=True, mamba_proj_bias=False, mamba_norm_before_gate=False,
        mamba_rms_norm=rms_norm, rms_norm_eps=1e-5, tie_word_embeddings=False,
        max_position_embeddings=4096, rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        embedding_multiplier=2.5, lm_head_multiplier=0.7, mlp_multipliers=[1.6, 0.6],
        key_multiplier=1.4, attention_in_multiplier=0.8, attention_out_multiplier=1.3,
        ssm_in_multiplier=1.2, ssm_out_multiplier=0.9, ssm_multipliers=[0.7, 1.3, 0.8, 1.5, 0.6])
    return FalconH1ForCausalLM, cfg, "starcoder", None


def falconh1nonorm():
    # Falcon-H1-0.5B's shape: no gated norm at all, y*silu(z) alone, one group.
    return falconh1(rms_norm=False, groups=1)


def lfm2():
    from transformers import Lfm2Config, Lfm2ForCausalLM
    # LFM2's block at toy size: gated short convolutions (in_proj to B, C
    # and x, B*x through a 3-tap depthwise convolution, C times that, out)
    # with one attention layer among them (NEOX rotary, a q/k RMSNorm per
    # head, GQA 4:2), a SwiGLU FFN after each, the final norm, tied
    # embeddings as every LFM2 ships.
    cfg = Lfm2Config(vocab_size=49152, hidden_size=64, intermediate_size=96, num_hidden_layers=4,
                     num_attention_heads=4, num_key_value_heads=2, norm_eps=1e-5, conv_bias=False,
                     conv_L_cache=3, block_auto_adjust_ff_dim=False,
                     layer_types=["conv", "full_attention", "conv", "conv"], tie_word_embeddings=True,
                     max_position_embeddings=4096, rope_parameters={"rope_type": "default", "rope_theta": 10000.0})
    return Lfm2ForCausalLM, cfg, "starcoder", None


def lfm2moe():
    from transformers import Lfm2MoeConfig, Lfm2MoeForCausalLM
    # LFM2-8B-A1B's shape: LFM2's blocks with a dense lead and a sigmoid
    # top-2 of 8 after it, its selection-only expert bias randomised (a
    # buffer, which perturb reaches by name). Its routed scale stays 1, as
    # every published one is: llama.cpp's converter writes no key for it.
    cfg = Lfm2MoeConfig(vocab_size=49152, hidden_size=64, intermediate_size=96, moe_intermediate_size=32,
                        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, norm_eps=1e-5,
                        conv_bias=False, conv_L_cache=3, num_dense_layers=1, num_experts=8,
                        num_experts_per_tok=2, use_expert_bias=True, routed_scaling_factor=1.0,
                        norm_topk_prob=True, layer_types=["conv", "full_attention", "conv", "conv"],
                        tie_word_embeddings=True, max_position_embeddings=4096,
                        rope_parameters={"rope_type": "default", "rope_theta": 10000.0})
    return Lfm2MoeForCausalLM, cfg, "starcoder", None


def mamba1():
    from transformers import MambaConfig, MambaForCausalLM
    # Mamba-1's own block at toy size: a 128-wide inner (expand 2), a 16-wide
    # state per channel, a rank-32 dt bottleneck, a 4-tap convolution with a
    # bias, no FFN, an untied head.
    cfg = MambaConfig(vocab_size=49152, hidden_size=64, state_size=16, num_hidden_layers=3, expand=2,
                      conv_kernel=4, use_bias=False, use_conv_bias=True, time_step_rank=32,
                      layer_norm_epsilon=1e-5, tie_word_embeddings=False, residual_in_fp32=True)
    return MambaForCausalLM, cfg, "starcoder", None


def falconmamba():
    from transformers import FalconMambaConfig, FalconMambaForCausalLM
    # FalconMamba is Mamba-1 with dt, B and C each RMS-normed before use, by a
    # norm with no weight (llama.cpp's ssm.dt_b_c_rms). Those norms' epsilon
    # is the class's mixer_rms_eps (1e-6 by default and on falcon-mamba-7b),
    # which the GGUF does not carry: llama.cpp uses the layer norm's. The
    # fixture sets the two equal so the two references agree; the divergence
    # is AGENTS.md RULE 7m's, recorded in model-correctness.md.
    cfg = FalconMambaConfig(vocab_size=49152, hidden_size=64, state_size=16, num_hidden_layers=3, expand=2,
                            conv_kernel=4, use_bias=False, use_conv_bias=True, time_step_rank=32,
                            layer_norm_epsilon=1e-5, mixer_rms_eps=1e-5, tie_word_embeddings=False,
                            residual_in_fp32=True)
    return FalconMambaForCausalLM, cfg, "starcoder", None


def jamba():
    from transformers import JambaConfig, JambaForCausalLM
    # Jamba's block at toy size: four layers, the third attention (no rotary,
    # GQA 4:2) and the rest Mamba-1 mixers whose dt, B and C carry weighted
    # RMSNorms, each followed by a SwiGLU MLP on even layers and a softmax
    # top-2 mixture of four experts, not renormalised, on odd ones.
    cfg = JambaConfig(vocab_size=32064, hidden_size=64, intermediate_size=96, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, num_experts=4, num_experts_per_tok=2,
                      expert_layer_period=2, expert_layer_offset=1, attn_layer_period=4, attn_layer_offset=2,
                      use_mamba_kernels=False, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
                      mamba_dt_rank=32, mamba_conv_bias=True, mamba_proj_bias=False, rms_norm_eps=1e-5,
                      tie_word_embeddings=False, max_position_embeddings=4096)
    return JambaForCausalLM, cfg, "phimoe", None


SPECS = {
    "synth-lfm2moe": lfm2moe,
    "synth-lfm2": lfm2,
    "synth-falconh1": falconh1,
    "synth-falconh1-nonorm": falconh1nonorm,
    "synth-nemotronh": nemotronh,
    "synth-nemotronhmoe": nemotronhmoe,
    "synth-mamba1": mamba1,
    "synth-falconmamba": falconmamba,
    "synth-jamba": jamba,
    "synth-mamba2": mamba2,
    "synth-granitehybrid": granitehybrid,
    "synth-granitehybridmoe": granitehybridmoe,
}


def perturb(m):
    # c6gold's classification plus Mamba's own parameters: A_log drawn so that
    # A = -exp(A_log) spans a decade, D around zero, and the dt bias at
    # inverse-softplus(dt) for dt in [0.01, 0.3], far from any clamp.
    with torch.no_grad():
        for n, p in m.named_parameters():
            leaf = n.split(".")[-1]
            if leaf == "A_log":
                p.copy_(torch.rand_like(p) * 2.3 - 1.2)
            elif leaf == "D":
                p.copy_(torch.randn_like(p))
            elif leaf == "dt_bias":
                dt = torch.exp(torch.rand_like(p) * (torch.log(torch.tensor(0.3)) - torch.log(torch.tensor(0.01)))
                               + torch.log(torch.tensor(0.01)))
                p.copy_(dt + torch.log(-torch.expm1(-dt)))
            elif n.endswith("weight") and "norm" in (n.split(".")[-2] if "." in n else ""):
                p.copy_(torch.randn_like(p) * 0.3 + 1)
            elif n.endswith("bias"):
                p.copy_(torch.randn_like(p) * 0.3)
            else:
                p.copy_(torch.randn_like(p) * 0.2)
        # A selection bias is a buffer, which named_parameters never reaches:
        # left at zero it is the identity and no gate could see it.
        for n, b in m.named_buffers():
            if n.endswith("expert_bias") or n.endswith("e_score_correction_bias"):
                b.copy_(torch.randn_like(b) * 0.3)


def build(name):
    cls, cfg, tokfam, post = SPECS[name]()
    cfg._attn_implementation = "eager"
    torch.manual_seed(20261003)
    m = cls(cfg).float().eval()
    perturb(m)
    d = os.path.join(HFDIR, name)
    shutil.rmtree(d, ignore_errors=True)
    m.save_pretrained(d, safe_serialization=True)
    for f in os.listdir(os.path.join(TOKDIR, tokfam)):
        if f != "config.json":
            shutil.copy(os.path.join(TOKDIR, tokfam, f), d)
    if post:
        post(m, cfg, d)
    dst = os.path.join(MODELS, name + ".gguf")
    conv = [PY, os.path.join(LCPP, "convert_hf_to_gguf.py")]
    if name == "synth-nemotronh":
        conv = [PY, "-c", DENSE_NEMOTRON, LCPP]
    subprocess.run(conv + [d, "--outtype", "f32", "--outfile", dst], check=True)
    ids = [(i * 7919 + 13) % min(cfg.vocab_size, 49000) for i in range(NPOS)]
    with torch.no_grad():
        lg = m(torch.tensor([ids])).logits[0]
    rows = [{"argmax": int(r.argmax()), "max": float(r.max()),
             "head": [round(float(x), 7) for x in r[:NLOGIT]]} for r in lg]
    tk = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
    texts = [{"text": t, "ids": tk.encode(t, add_special_tokens=False).ids} for t in TEXTS]
    os.makedirs(OUT, exist_ok=True)
    json.dump({"ids": ids, "vocab": cfg.vocab_size, "pos": rows, "texts": texts},
              open(os.path.join(OUT, name + ".json"), "w"))
    print("wrote", dst, "argmax", [r["argmax"] for r in rows])


# A fixture's own weights written at Q8_0, for a gate that needs a packed bank
# (a streamed one: a float bank has no packed layout to send a sheet of). No
# golden: such a gate compares the engine with itself. Build its base first.
QUANTIZED = {"synth-nemotronhmoe-q8": "synth-nemotronhmoe"}


def quantized(name):
    base = QUANTIZED[name]
    dst = os.path.join(MODELS, name + ".gguf")
    subprocess.run([PY, os.path.join(LCPP, "convert_hf_to_gguf.py"), os.path.join(HFDIR, base),
                    "--outtype", "q8_0", "--outfile", dst], check=True)
    print("wrote", dst)


if __name__ == "__main__":
    for n in sys.argv[1:] or list(SPECS) + list(QUANTIZED):
        quantized(n) if n in QUANTIZED else build(n)
