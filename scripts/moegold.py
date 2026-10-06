#!/usr/bin/env python3
"""Build the mixture-family fixtures and record transformers' logits.

    $JITLLM_HF_PY scripts/moegold.py [name ...]
    JITLLM_MODELS=/path/to/models ... (where the GGUFs go)

-> $JITLLM_MODELS/<name>.gguf and testdata/golden/moe/<name>.json

c6gold.py's method for the mixture families llama.cpp builds as glm4moe,
qwen2moe, hunyuan-moe/hunyuan-dense and ernie4_5/ernie4_5-moe: the oracle is
transformers' OWN class, the GGUF is written by llama.cpp's OWN converter, and
every tensor is F32.

★ A ROUTER'S SELECTION BIAS IS A BUFFER, NOT A PARAMETER, AND transformers
INITIALISES IT TO ZERO -- WHICH IS THE IDENTITY. hfgold.py's DeepSeek fixture
once proved nothing about it that way (AGENTS.md RULE 7). perturb() randomises
every parameter AND every floating buffer whose name says it is a correction
bias, at a size that moves the top-k (a router logit here is ~0.3 wide, and
the bias is drawn at 0.5).
"""
import json
import math
import os
import shutil
import subprocess
import sys

import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer

import refenv  # where things are: environment variables only (docs/testing.md)

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = refenv.models()
TOKDIR = refenv.under("CV_TOK", "tok")
HFDIR = refenv.under("CV_HF", "hf")
LCPP = refenv.need("LCPP_SRC", "a llama.cpp source checkout")
PY = sys.executable
OUT = os.path.join(HERE, "..", "testdata", "golden", "moe")
NPOS, NLOGIT = 12, 512
TEXTS = [
    "The capital of France is",
    "def getUserName(self):\n    return 12345",
    " leading space and  double  spaces\nline two\ttab",
]


def glm4moe():
    from transformers import Glm4MoeConfig, Glm4MoeForCausalLM
    # GLM-4.5's shape at toy size: GQA 4:2, biased q/k/v and an unbiased o,
    # per-head q/k RMSNorm (the 355B's use_qk_norm), NEOX rotary on HALF of
    # each head, one dense lead block, then sigmoid top-2 of 8 with the
    # selection-only bias, renormalised and scaled by 2.5, beside one shared
    # expert. n_group 1 / topk_group 1 is every published GLM-4.5's: llama.cpp
    # writes no group keys for this arch, so a grouped GLM would part from it.
    # One MTP block (num_nextn_predict_layers) is appended to the safetensors
    # after the trunk is saved; transformers never runs it, so the trunk's
    # logits are the reference and the converter must set the block aside.
    cfg = Glm4MoeConfig(vocab_size=151552, hidden_size=64, intermediate_size=128,
                        moe_intermediate_size=32, num_hidden_layers=3, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=16, n_routed_experts=8,
                        num_experts_per_tok=2, n_shared_experts=1, first_k_dense_replace=1,
                        n_group=1, topk_group=1, norm_topk_prob=True, routed_scaling_factor=2.5,
                        attention_bias=True, use_qk_norm=True, rms_norm_eps=1e-5,
                        rope_parameters={"rope_theta": 1000000.0, "rope_type": "default",
                                         "partial_rotary_factor": 0.5},
                        max_position_embeddings=2048, tie_word_embeddings=False,
                        num_mtp_layers=1)
    return Glm4MoeForCausalLM, cfg, "glm4moe", glm4moe_mtp


def glm4moe_mtp(m, cfg, d):
    """Append one prediction block, model.layers.<n>, as GLM-4.5 ships it."""
    from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeDecoderLayer, Glm4MoeRMSNorm
    n = cfg.num_hidden_layers
    torch.manual_seed(7)
    blk = Glm4MoeDecoderLayer(cfg, n).float()
    extra = {"eh_proj": torch.nn.Linear(2 * cfg.hidden_size, cfg.hidden_size, bias=False),
             "enorm": Glm4MoeRMSNorm(cfg.hidden_size), "hnorm": Glm4MoeRMSNorm(cfg.hidden_size),
             "shared_head.norm": Glm4MoeRMSNorm(cfg.hidden_size)}
    perturb(blk)
    sd = {}
    for k, v in blk.state_dict().items():
        # The block's experts are one fused bank in memory; a checkpoint
        # carries them per expert, as save_pretrained wrote the trunk's.
        if k == "mlp.experts.gate_up_proj":
            g, u = v.chunk(2, dim=1)
            for e in range(v.shape[0]):
                sd["model.layers.%d.mlp.experts.%d.gate_proj.weight" % (n, e)] = g[e].contiguous()
                sd["model.layers.%d.mlp.experts.%d.up_proj.weight" % (n, e)] = u[e].contiguous()
            continue
        if k == "mlp.experts.down_proj":
            for e in range(v.shape[0]):
                sd["model.layers.%d.mlp.experts.%d.down_proj.weight" % (n, e)] = v[e].contiguous()
            continue
        sd["model.layers.%d.%s" % (n, k)] = v.contiguous()
    with torch.no_grad():
        for k, mod in extra.items():
            for pk, v in mod.state_dict().items():
                v = torch.randn_like(v) * 0.2 + (1 if "norm" in k else 0)
                sd["model.layers.%d.%s.%s" % (n, k, pk)] = v.contiguous()
    save_file(sd, os.path.join(d, "mtp.safetensors"), metadata={"format": "pt"})
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(idx):
        j = json.load(open(idx))
    else:
        main = load_file(os.path.join(d, "model.safetensors"))
        j = {"metadata": {}, "weight_map": {k: "model.safetensors" for k in main}}
    for k in sd:
        j["weight_map"][k] = "mtp.safetensors"
    json.dump(j, open(idx, "w"), indent=2)
    c = json.load(open(os.path.join(d, "config.json")))
    c["num_nextn_predict_layers"] = 1
    json.dump(c, open(os.path.join(d, "config.json"), "w"), indent=2)


def qwen2moe():
    from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM
    # Qwen1.5-MoE's shape: biased q/k/v and an unbiased o, NEOX rotary over
    # the whole head, a softmax top-2 of 8 NOT renormalised (norm_topk_prob
    # false, as every published Qwen1.5-MoE and Qwen2-57B states and as
    # llama.cpp hardcodes), and a shared expert twice a routed one's width
    # scaled by its own sigmoid gate. GQA 4:2 where Qwen1.5-MoE is MHA, so the
    # grouped KV is exercised too.
    cfg = Qwen2MoeConfig(vocab_size=151936, hidden_size=64, intermediate_size=128,
                         moe_intermediate_size=32, shared_expert_intermediate_size=64,
                         num_experts=8, num_experts_per_tok=2, norm_topk_prob=False,
                         decoder_sparse_step=1, mlp_only_layers=[], num_hidden_layers=3,
                         num_attention_heads=4, num_key_value_heads=2, qkv_bias=True,
                         rms_norm_eps=1e-6, rope_parameters={"rope_theta": 1000000.0, "rope_type": "default"},
                         max_position_embeddings=2048, tie_word_embeddings=False,
                         use_sliding_window=False)
    return Qwen2MoeForCausalLM, cfg, "qwen2moe", None


def ernie45moe():
    from transformers import Ernie4_5_MoeConfig, Ernie4_5_MoeForCausalLM
    # ERNIE-4.5-21B-A3B's shape: no attention biases, interleaved (NORM)
    # rotary, a tied head, and a softmax top-2 of 8 whose SELECTION adds a
    # per-expert bias to the softmax probability, renormalised, beside two
    # shared experts. Four blocks with moe_layer_start_index 1 and
    # moe_layer_interval 2, so block 0 is the dense lead, block 2 is dense by
    # the step, and blocks 1 and 3 are mixtures. intermediate_size // kv_heads
    # is llama.cpp's converter's reading of the shared width and
    # moe_intermediate_size * shared is transformers'; they agree on every
    # published ERNIE and are made to agree here (128 // 2 == 32 * 2).
    cfg = Ernie4_5_MoeConfig(vocab_size=103424, hidden_size=64, intermediate_size=128,
                             moe_intermediate_size=32, num_hidden_layers=4, num_attention_heads=4,
                             num_key_value_heads=2, head_dim=16, moe_num_experts=8, moe_k=2,
                             moe_num_shared_experts=2, moe_layer_start_index=1,
                             moe_layer_end_index=3, moe_layer_interval=2, moe_norm_min=1e-12,
                             use_bias=False, rms_norm_eps=1e-5,
                             rope_parameters={"rope_theta": 500000.0, "rope_type": "default"},
                             max_position_embeddings=2048, tie_word_embeddings=True)
    return Ernie4_5_MoeForCausalLM, cfg, "ernie4_5", ernie_keys


def ernie45():
    from transformers import Ernie4_5Config, Ernie4_5ForCausalLM
    # ERNIE-4.5-0.3B's shape: llama's block with interleaved rotary and a tied
    # head; head_dim 16 where hidden/heads is 16 too.
    cfg = Ernie4_5Config(vocab_size=103424, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=16, use_bias=False, rms_norm_eps=1e-5,
                         rope_parameters={"rope_theta": 500000.0, "rope_type": "default"},
                         max_position_embeddings=2048, tie_word_embeddings=True)
    return Ernie4_5ForCausalLM, cfg, "ernie4_5", None


def ernie_keys(m, cfg, d):
    """llama.cpp's converter reads ERNIE's own key names; state them."""
    c = json.load(open(os.path.join(d, "config.json")))
    for k in ("moe_num_experts", "moe_k", "moe_layer_interval", "moe_layer_start_index",
              "moe_layer_end_index", "moe_num_shared_experts", "moe_intermediate_size"):
        c.setdefault(k, getattr(cfg, k))
    json.dump(c, open(os.path.join(d, "config.json"), "w"), indent=2)


def hunyuanmoe():
    from transformers import HunYuanMoEV1Config, HunYuanMoEV1ForCausalLM
    # Hunyuan-A13B's shape: no biases, NEOX rotary over the whole head, a
    # per-head q/k RMSNorm applied AFTER the rotary, and in every block a
    # softmax top-2 of 8, renormalised, beside one ungated shared MLP as wide
    # as an expert (the class builds both from intermediate_size). The rotary
    # is A13B's NTK-alpha base (rope_theta 10000, alpha 1000) at head_dim 128,
    # which llama.cpp's converter asserts; GQA 2:1, and a tied head as A13B's.
    cfg = HunYuanMoEV1Config(vocab_size=120818, hidden_size=256, intermediate_size=64,
                             num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1, head_dim=128,
                             num_experts=8, moe_topk=2, rms_norm_eps=1e-5, attention_bias=False,
                             rope_parameters={"rope_type": "dynamic", "alpha": 1000.0, "factor": 1.0,
                                              "rope_theta": 10000.0},
                             max_position_embeddings=32768, tie_word_embeddings=True,
                             bos_token_id=120000, eos_token_id=120020, pad_token_id=120002)
    return HunYuanMoEV1ForCausalLM, cfg, "hunyuan", hunyuan_keys


def hunyuan():
    from transformers import HunYuanDenseV1Config, HunYuanDenseV1ForCausalLM
    # Hunyuan-0.5B's shape at toy size: the same attention, a dense SwiGLU,
    # head_dim 16 (the dense converter reads head_dim, not hidden/heads), the
    # NTK-alpha base, a tied head.
    cfg = HunYuanDenseV1Config(vocab_size=120818, hidden_size=64, intermediate_size=128,
                               num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                               head_dim=16, rms_norm_eps=1e-5, attention_bias=False,
                               rope_parameters={"rope_type": "dynamic", "alpha": 1000.0, "factor": 1.0,
                                                "rope_theta": 10000.0},
                               max_position_embeddings=32768, tie_word_embeddings=True,
                               bos_token_id=120000, eos_token_id=120020, pad_token_id=120002)
    return HunYuanDenseV1ForCausalLM, cfg, "hunyuan", hunyuan_keys


def hunyuan_keys(m, cfg, d):
    """llama.cpp's converter reads Tencent's own key names; state them."""
    c = json.load(open(os.path.join(d, "config.json")))
    n = cfg.num_hidden_layers
    if getattr(cfg, "num_experts", 1) > 1:
        c["moe_intermediate_size"] = [cfg.intermediate_size] * n
        c["moe_topk"] = [cfg.moe_topk] * n
        c["num_shared_expert"] = [1] * n
    c["eod_token_id"] = cfg.eos_token_id
    json.dump(c, open(os.path.join(d, "config.json"), "w"), indent=2)


def minimaxm2():
    from transformers import MiniMaxM2Config, MiniMaxM2ForCausalLM
    # MiniMax-M2's shape at toy size: GQA 4:2 at head_dim 16, q and k
    # RMSNormed over the WHOLE projection (64 and 32 wide), NEOX rotary on
    # half of each head (rotary_dim 64 of 128 on the real model), and in every
    # block a sigmoid top-2 of 8 with the selection-only bias, renormalised,
    # no shared expert, no routed scale, an untied head.
    cfg = MiniMaxM2Config(vocab_size=200064, hidden_size=64, intermediate_size=32,
                          num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                          head_dim=16, num_local_experts=8, num_experts_per_tok=2,
                          rms_norm_eps=1e-6, max_position_embeddings=4096, tie_word_embeddings=False,
                          rope_parameters={"rope_theta": 5000000.0, "rope_type": "default",
                                           "partial_rotary_factor": 0.5},
                          bos_token_id=200034, eos_token_id=200020)
    return MiniMaxM2ForCausalLM, cfg, "minimax-m2", minimax_keys


def minimax_keys(m, cfg, d):
    """llama.cpp's converter reads MiniMax's own key names (rotary_dim, and
    scoring_func for the gate); the published config.json states both."""
    c = json.load(open(os.path.join(d, "config.json")))
    c["rotary_dim"] = int(cfg.head_dim * cfg.rope_parameters["partial_rotary_factor"])
    c["scoring_func"] = "sigmoid"
    json.dump(c, open(os.path.join(d, "config.json"), "w"), indent=2)


# llama.cpp's HunYuanMoEModel.set_vocab rebuilds the vocabulary from Tencent's
# remote-code tiktoken tokenizer, which the published repos no longer ship; the
# dense class reads tokenizer.json. The fixture's vocabulary is Hunyuan-0.5B's
# tokenizer.json, so its mixture converts with the dense class's set_vocab.
TIKTOKEN_FREE = """
import runpy, sys
sys.path.insert(0, sys.argv[1])
from conversion import hunyuan
hunyuan.HunYuanMoEModel.set_vocab = hunyuan.HunYuanModel.set_vocab
script = sys.argv[1] + "/convert_hf_to_gguf.py"
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
"""


# Ling 2.0's reference is inclusionAI's remote code (modeling_bailing_moe_v2.py,
# shipped with Ling-mini-2.0), written against transformers 4.52 -- it imports
# names transformers 5 removed. It is imported from a copy of the repo's own
# files under BAILING_REF, by the interpreter of a venv holding transformers
# 4.52.3 (BAILING_PY); the script re-runs itself there for this fixture.
BAILING_REF = refenv.under("BAILING_REF", "hf", "pyref")
BAILING_PY = os.environ.get("BAILING_PY", "")  # checked where the fixture needs it


def bailingmoe2():
    sys.path.insert(0, BAILING_REF)
    from bailingref.configuration_bailing_moe_v2 import BailingMoeV2Config
    from bailingref.modeling_bailing_moe_v2 import BailingMoeV2ForCausalLM
    # Ling-mini-2.0's shape at toy size: a fused query_key_value with no
    # biases, GQA 4:2, per-head q/k RMSNorm before NEOX rotary on HALF of each
    # head, one dense lead block, then a sigmoid top-4 of 16 in 4 groups of
    # which 2 are kept (Ling-mini keeps 4 of 8), the selection-only
    # expert_bias, renormalised and scaled by 2.5, beside one shared expert as
    # every published Ling 2.0 has. (Two would be one MLP of twice an expert's
    # width in the class, and llama.cpp's loader multiplies the width its own
    # converter already multiplied: it refuses that file.)
    cfg = BailingMoeV2Config(vocab_size=157184, hidden_size=64, intermediate_size=128,
                             moe_intermediate_size=32, num_hidden_layers=3, num_attention_heads=4,
                             num_key_value_heads=2, head_dim=16, num_experts=16,
                             num_experts_per_tok=4, num_shared_experts=1, first_k_dense_replace=1,
                             n_group=4, topk_group=2, norm_topk_prob=True,
                             routed_scaling_factor=2.5, use_qk_norm=True,
                             partial_rotary_factor=0.5, rope_theta=600000.0,
                             max_position_embeddings=2048, tie_word_embeddings=False,
                             use_qkv_bias=False, use_bias=False, rms_norm_eps=1e-6,
                             num_nextn_predict_layers=0, score_function="sigmoid")
    return BailingMoeV2ForCausalLM, cfg, "ling", None


REMOTE = {"synth-bailingmoe2": BAILING_PY}


def phimoe():
    from transformers import PhimoeConfig, PhimoeForCausalLM
    # Phi-3.5-MoE's shape at toy size: LayerNorm with biases, biased q/k/v/o
    # and a biased head, GQA 4:2, NEOX LongRoPE (short factors below the
    # original context, which the fixture's positions all are), and 8 experts
    # routed by sparsemixer at router_jitter_noise 0.01. short_mscale is the
    # value llama.cpp's converter derives from the context ratio,
    # sqrt(1 + ln(4096/512)/ln(512)), so both inputs carry the same cos/sin
    # scale; Phi-3.5-MoE's own (1.243) is a config number no GGUF keeps.
    ms = math.sqrt(1 + math.log(4096 / 512) / math.log(512))
    torch.manual_seed(20261004)
    short = [round(1 + 0.5 * float(x), 4) for x in torch.rand(8)]
    long = [round(2 + 4 * float(x), 4) for x in torch.rand(8)]
    cfg = PhimoeConfig(vocab_size=32064, hidden_size=64, intermediate_size=32, num_hidden_layers=3,
                       num_attention_heads=4, num_key_value_heads=2, num_local_experts=8,
                       num_experts_per_tok=2, router_jitter_noise=0.01, input_jitter_noise=0.0,
                       attention_bias=True, lm_head_bias=True, rms_norm_eps=1e-5,
                       max_position_embeddings=4096, sliding_window=None, tie_word_embeddings=False,
                       rope_parameters={"rope_type": "longrope", "rope_theta": 10000.0,
                                        "short_factor": short, "long_factor": long,
                                        "short_mscale": ms, "long_mscale": ms,
                                        "original_max_position_embeddings": 512})
    return PhimoeForCausalLM, cfg, "phimoe", phimoe_class_name


def phimoe_close_router(m):
    """Random routers put every second expert past sparsemixer's 2% threshold,
    where each weight is exactly 1 and the masks are invisible. Phi's routers
    are not random: give each FFN norm a large common bias and each router row
    a shared component along it, so the logits sit near one value with a few
    percent of spread -- some experts inside the threshold of the top two, some
    outside it."""
    with torch.no_grad():
        for layer in m.model.layers:
            ln = layer.post_attention_layernorm
            ln.bias.copy_(2 + 0.3 * torch.randn_like(ln.bias))
            w = layer.mlp.router.weight
            w.copy_(0.05 + 0.01 * torch.randn_like(w))


def phimoe_class_name(m, cfg, d):
    """The published checkpoint names its class as its remote code did,
    PhiMoEForCausalLM, which is the name llama.cpp's converter registers;
    transformers saves its own spelling."""
    c = json.load(open(os.path.join(d, "config.json")))
    c["architectures"] = ["PhiMoEForCausalLM"]
    json.dump(c, open(os.path.join(d, "config.json"), "w"), indent=2)


# PRE runs after perturb and before the checkpoint is saved.
PRE = {"synth-phimoe": phimoe_close_router}


def dots1():
    from transformers import Dots1Config, Dots1ForCausalLM
    # dots.llm1's shape at toy size: no attention biases, MHA as the 142B is
    # (llama.cpp's loader sizes k and v at every head, so a GQA dots1 is a file
    # it refuses), qwen3's per-head q/k RMSNorm before NEOX rotary over the whole
    # head, one dense lead block, then a sigmoid top-2 of 8 with the
    # selection-only e_score_correction_bias, renormalised and scaled by 2.5,
    # beside two shared experts (one MLP of twice an expert's width, which
    # llama.cpp's loader reads as n_ff_exp * n_expert_shared as well). No
    # sliding window, as every published dots.llm1 states.
    cfg = Dots1Config(vocab_size=152064, hidden_size=64, intermediate_size=128,
                      moe_intermediate_size=32, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=4, head_dim=16, n_routed_experts=8,
                      num_experts_per_tok=2, n_shared_experts=2, first_k_dense_replace=1,
                      n_group=1, topk_group=1, norm_topk_prob=True, routed_scaling_factor=2.5,
                      rms_norm_eps=1e-5, sliding_window=None,
                      rope_parameters={"rope_theta": 10000000.0, "rope_type": "default"},
                      max_position_embeddings=2048, tie_word_embeddings=False,
                      scoring_func="sigmoid")
    return Dots1ForCausalLM, cfg, "dots1", None

SPECS = {
    "synth-hunyuanmoe": hunyuanmoe,
    "synth-hunyuan": hunyuan,
    "synth-glm4moe": glm4moe,
    "synth-qwen2moe": qwen2moe,
    "synth-ernie45moe": ernie45moe,
    "synth-ernie45": ernie45,
    "synth-minimaxm2": minimaxm2,
    "synth-bailingmoe2": bailingmoe2,
    "synth-dots1": dots1,
    "synth-phimoe": phimoe,
}

# The selection bias's size, where the default (0.5) would swamp the score it
# is added to: a softmax over 8 experts puts each near 0.125, so ERNIE's bias
# is drawn at a size that moves the top-k without deciding it alone.
SEL_BIAS = {"synth-ernie45moe": 0.05}


def perturb(m, sel=0.5):
    # c6gold.perturb's classification (a norm is a parameter of a norm
    # module), plus the router's selection bias -- a buffer in GLM's router,
    # a frozen parameter in ERNIE's -- drawn at sel.
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "correction_bias" in n or n.endswith("expert_bias"):
                p.copy_(torch.randn_like(p) * sel)
            elif n.endswith("weight") and "norm" in (n.split(".")[-2] if "." in n else ""):
                p.copy_(torch.randn_like(p) * 0.3 + 1)
            elif n.endswith("bias"):
                p.copy_(torch.randn_like(p) * 0.3)
            else:
                p.copy_(torch.randn_like(p) * 0.2)
        for n, b in m.named_buffers():
            if "correction_bias" in n or n.endswith("expert_bias"):
                b.copy_(torch.randn_like(b) * sel)


def build(name):
    cls, cfg, tokfam, post = SPECS[name]()
    cfg._attn_implementation = "eager"
    torch.manual_seed(20261003)
    m = cls(cfg).float().eval()
    perturb(m, SEL_BIAS.get(name, 0.5))
    if name in PRE:
        PRE[name](m)
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
    if name.startswith("synth-hunyuan"):
        conv = [PY, "-c", TIKTOKEN_FREE, LCPP]
    subprocess.run(conv + [d, "--outtype", "f32", "--outfile", dst], check=True)
    ids = [(i * 7919 + 13) % min(cfg.vocab_size, 50000) for i in range(NPOS)]
    with torch.no_grad():
        lg = m(torch.tensor([ids]), use_cache=False).logits[0]
    rows = [{"argmax": int(r.argmax()), "max": float(r.max()),
             "head": [round(float(x), 7) for x in r[:NLOGIT]]} for r in lg]
    spm = os.path.join(d, "tokenizer.model")
    if os.path.exists(spm):
        import sentencepiece
        sp = sentencepiece.SentencePieceProcessor(model_file=spm)
        texts = [{"text": t, "ids": sp.encode(t)} for t in TEXTS]
    else:
        tk = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
        texts = [{"text": t, "ids": tk.encode(t, add_special_tokens=False).ids} for t in TEXTS]
    os.makedirs(OUT, exist_ok=True)
    json.dump({"ids": ids, "vocab": cfg.vocab_size, "pos": rows, "texts": texts},
              open(os.path.join(OUT, name + ".json"), "w"))
    print("wrote", dst, "argmax", [r["argmax"] for r in rows])


if __name__ == "__main__":
    for n in sys.argv[1:] or list(SPECS):
        py = REMOTE.get(n)
        if n in REMOTE and not py:
            sys.exit(f"moegold.py: {n} needs BAILING_PY, the python of a venv with transformers 4.52.3")
        if py and os.path.realpath(sys.executable) != os.path.realpath(py):
            subprocess.run([py, os.path.abspath(__file__), n], check=True)
            continue
        build(n)
