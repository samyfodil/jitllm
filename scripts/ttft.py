#!/usr/bin/env python3
"""Time to first token and decode rate against any OpenAI-compatible server.

    scripts/ttft.py URL MODEL [-reps 51] [-n 256] [-k 4] [-label jitllm]

One streaming /v1/completions request at a time, greedy: the time from
sending the request to the first chunk carrying text is the time to first
token, and the decode rate is (completion tokens - 1) over the time from that
chunk to the last. The prompt is a sentence repeated -reps times behind a
per-request number, so no engine can answer it from a prefix cache; llama.cpp
is told cache_prompt false as well. The first request is a warm-up and is not
printed. Each line is JSON, one request.

The same client runs against jitllmd, llama-server and vllm serve, so the
three figures are taken by one stopwatch. It reads nothing from the
environment and names no host: the URL and model are arguments.
"""
import argparse
import http.client
import json
import sys
import time
import urllib.parse

SENTENCE = "The quick brown fox jumps over the lazy dog while the farmer watches from the porch. "


def one(url, model, prompt, n):
    u = urllib.parse.urlparse(url)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=600)
    body = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": n, "temperature": 0,
        "stream": True, "stream_options": {"include_usage": True},
        "ignore_eos": True, "cache_prompt": False,
    })
    t0 = time.perf_counter()
    conn.request("POST", "/v1/completions", body, {"Content-Type": "application/json"})
    r = conn.getresponse()
    if r.status != 200:
        raise SystemExit("HTTP %d: %s" % (r.status, r.read()[:400]))
    first = last = None
    chunks = 0
    usage = None
    for raw in r:
        line = raw.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        ev = json.loads(data)
        if ev.get("usage"):
            usage = ev["usage"]
        for ch in ev.get("choices") or []:
            if ch.get("text"):
                now = time.perf_counter()
                if first is None:
                    first = now
                last = now
                chunks += 1
    conn.close()
    if first is None:
        raise SystemExit("no text came back")
    ntok = (usage or {}).get("completion_tokens") or chunks
    ptok = (usage or {}).get("prompt_tokens")
    dec = (ntok - 1) / (last - first) if last > first else 0.0
    return {"ttft_ms": round((first - t0) * 1000, 1), "prompt_tokens": ptok,
            "completion_tokens": ntok, "chunks": chunks, "decode_tok_s": round(dec, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("model")
    ap.add_argument("-reps", type=int, default=51)
    ap.add_argument("-n", type=int, default=256)
    ap.add_argument("-k", type=int, default=4, help="timed requests after the warm-up")
    ap.add_argument("-label", default="")
    a = ap.parse_args()
    body = SENTENCE * a.reps
    for i in range(a.k + 1):
        res = one(a.url, a.model, "%d. %s" % (i, body), a.n)
        if i == 0:
            continue
        res["label"], res["i"] = a.label, i
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    sys.exit(main())
