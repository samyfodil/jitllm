"""Where the reference scripts find things: environment variables only.

Every script under scripts/ that builds a fixture or records a golden reads
its locations through this module, and none of them carries a default that
names a machine. docs/testing.md ("Reference scripts") lists the variables:

    JITLLM_HF_PY    the python of a venv with torch, transformers and
                    tokenizers -- the one every *gold.py / *real.py runs under
    JITLLM_MODELS   the model directory: GGUFs, fixtures and checkpoints
    LCPP_SRC        a llama.cpp source checkout (convert_hf_to_gguf.py, gguf-py)
    JITLLM_LCPP     a llama.cpp build directory (llama-completion and friends)
    JITLLM_LLAMACPP_CPU  a CPU-only llama.cpp build directory (llama-mtmd-cli)

A variable a script needs and does not find stops it, naming the variable,
rather than writing a fixture somewhere nobody asked for.
"""
import os
import sys


def need(name, what):
    """The value of $name, or exit naming it and what it is for."""
    v = os.environ.get(name, "").strip()
    if not v:
        script = os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else "script"
        sys.exit(f"{script}: set {name} to {what} (docs/testing.md, 'Reference scripts')")
    return v


def models():
    """$JITLLM_MODELS, the model directory."""
    return need("JITLLM_MODELS", "the model directory")


def under(name, *rel):
    """$name if set, else $JITLLM_MODELS/<rel...>."""
    v = os.environ.get(name, "").strip()
    return v if v else os.path.join(models(), *rel)


def lcpp_src():
    """$LCPP_SRC, a llama.cpp source checkout."""
    return need("LCPP_SRC", "a llama.cpp source checkout")


def lcpp_bin():
    """$JITLLM_LCPP, a llama.cpp build directory."""
    return need("JITLLM_LCPP", "a llama.cpp build directory (llama-completion, llama-tokenize)")


def arg(i, *rel):
    """sys.argv[i] if given, else $JITLLM_MODELS/<rel...>."""
    return sys.argv[i] if len(sys.argv) > i else os.path.join(models(), *rel)
