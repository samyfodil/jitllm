#!/usr/bin/env python3
"""Record llama-mtmd-cli's greedy answer about a picture, for the engine to be
teacher-forced through.

    scripts/qwenvlmtmd.py <name> <text.gguf> <mmproj.gguf> [image]
    JITLLM_LLAMACPP_CPU=<dir of a CPU build of llama.cpp, with llama-mtmd-cli>

-> testdata/golden/qwenvl/mtmd-<name>.json: the prompt and llama.cpp's reply.

The flags are the ones scripts/rungold.py's goldens use -- greedy (top-k 1,
temperature 0), no flash attention, the CPU -- because a sampler or a backend
choice reads as a model difference. The prompt is the model's own chat template
around one picture, which is what model.ChatSpansImages renders.
"""
import json
import os
import subprocess
import sys

import refenv  # where things are: environment variables only (docs/testing.md)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "testdata", "golden", "qwenvl")
BIN = refenv.need("JITLLM_LLAMACPP_CPU", "a CPU build directory of llama.cpp")
PROMPT = "Describe this image in one sentence."


def main(name, text, mmproj, image):
    env = dict(os.environ, LD_LIBRARY_PATH=BIN)
    r = subprocess.run([os.path.join(BIN, "llama-mtmd-cli"), "-m", text, "--mmproj", mmproj,
                        "--image", image, "-p", PROMPT, "--temp", "0", "--top-k", "1",
                        "-n", "48", "-fa", "off", "-dev", "none", "--seed", "1"],
                       env=env, capture_output=True, text=True, check=True)
    reply = r.stdout.strip()
    g = {"prompt": PROMPT, "image": os.path.basename(image), "reply": reply}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "mtmd-" + name + ".json"), "w") as f:
        json.dump(g, f, indent=1)
    print(f"{name}: {reply!r}")


if __name__ == "__main__":
    img = sys.argv[4] if len(sys.argv) > 4 else os.path.join(HERE, "..", "engine", "model", "testdata",
                                                               "quad-and-disc.png")
    main(sys.argv[1], sys.argv[2], sys.argv[3], img)
