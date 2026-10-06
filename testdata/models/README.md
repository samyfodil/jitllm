# stories260K.gguf

1.1 MB, committed on purpose: it makes `go test ./convert/gguf` hermetic on a fresh
checkout with no network and no 2 GB download.

It is the smallest thing that is still a real llama-architecture transformer —
5 blocks, embedding 64, ffn 172, vocab 512, 8 heads / 4 kv heads — and it is
**100% F32**, which is the point. A wrong answer on this model is a graph bug,
never a dequantization bug, so it isolates the two failure modes that otherwise
look identical (fluent nonsense). It is also small enough that a numpy float64
reference forward pass runs instantly, which is what the M3 oracle needs.

Source: https://huggingface.co/ggml-org/models `tinyllamas/stories260K.gguf`,
a GGUF conversion of Karpathy's llama2.c TinyStories models.

It is the only committed model. Every other file a test reads is fetched into
the model directory (`$JITLLM_MODELS`, or `models/` at the repository root) by
`internal/testmodels/fetch.sh`, whose table records where each one comes from;
`fetch.sh --list` prints it. Some tests look for stories260K in the model
directory too, so the script copies this file there. See `docs/testing.md`.
