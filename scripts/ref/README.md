# scripts/ref — the other engine's kernel, timed on our terms

`q3kbench.c` compiles llama.cpp's own `ggml_vec_dot_q3_K_q8_K` NEON body and
times it with the SAME protocol `dev/tools/fmtbench` uses: one thread, a
280 MB DRAM-resident working set, a 1.5 s soak and the median of fifteen.

**Why this exists.** Reading both kernels said jitllm should be ahead: jitllm issues
about 80 unpack instructions per super-block against llama.cpp's 76, the same
sixteen `SDOT`s, and *no* horizontal reduces where llama.cpp pays sixteen
`vaddvq_s32` (RULE 5d keeps all four lanes with `MLA` by element). A static
instruction count predicted a jitllm win and the measurement says the opposite:

    jitllm Q3_K       12.6 GB/s
    llama.cpp Q3_K 14.3 GB/s     +13.5%

That is the whole reason to build this. Comparing two engines end to end tells
you which is faster; comparing one kernel against the other, on one core, with
everything else held fixed, tells you *where*.

Build the body first — it is extracted from a llama.cpp checkout rather than
vendored, so nothing here is a copy of their source in this repo:

    D=/path/to/llama.cpp
    L=$(grep -n 'void ggml_vec_dot_q3_K_q8_K' $D/ggml-quants.c | head -1 | cut -d: -f1)
    S=$(awk -v s=$L 'NR>s && /^#ifdef __ARM_NEON/{print NR; exit}' $D/ggml-quants.c)
    E=$(awk -v s=$S 'NR>s && (/^#elif/ || /^#else/){print NR; exit}' $D/ggml-quants.c)
    sed -n "$((S+1)),$((E-1))p" $D/ggml-quants.c > lcpp_q3k_body.inc
    clang -O3 -march=armv8.2-a+dotprod -o q3kbench q3kbench.c && ./q3kbench
