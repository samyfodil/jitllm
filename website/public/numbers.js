// The benchmark boards, from docs/perf/current.md (the final board).
// Ratios are jitllm ÷ the other engine as recorded there;
// nothing here is recomputed or rounded again.
;(() => {
  const root = document.querySelector('.bench')
  if (!root) return

  // [model, tag, gpus, prefill jitllm, prefill llama.cpp, ×, decode jitllm, decode llama.cpp, ×]
  const V100 = [
    ['SmolLM2-360M Q8_0', '', 1, 33613, 24361, 1.38, 492.2, 405.5, 1.21],
    ['Qwen3-0.6B Q8_0', '', 1, 23980, 20217, 1.19, 422.2, 381.1, 1.11],
    ['gemma-3-1b', '', 1, 17324, 14238, 1.22, 331.8, 273.2, 1.21],
    ['Llama-3.2-1B', '', 1, 21062, 16081, 1.31, 530.1, 464.6, 1.14],
    ['SmolLM2-1.7B', '', 1, 13129, 10147, 1.29, 376.5, 338.8, 1.11],
    ['Qwen2.5-1.5B', '', 1, 12504, 11548, 1.08, 359.2, 285.6, 1.26],
    ['gemma-2-2b', '', 1, 9869, 8177, 1.21, 246.3, 208.1, 1.18],
    ['Qwen3-1.7B Q8_0', '', 1, 12468, 11961, 1.04, 266.8, 250.9, 1.06],
    ['Llama-3.2-3B', '', 1, 8022, 6279, 1.28, 244.6, 220.7, 1.11],
    ['Phi-3.5-mini', '', 1, 5643, 5661, 1, 211.2, 199.5, 1.06],
    ['Phi-4-mini', '', 1, 6793, 6051, 1.12, 201.1, 198.2, 1.01],
    ['gemma-3-4b', '', 1, 6924, 5735, 1.21, 172.8, 154.4, 1.12],
    ['Qwen3-4B', '', 1, 6670, 5170, 1.29, 190.8, 168.5, 1.13],
    ['OLMoE-1B-7B', 'MoE', 1, 10922, 4152, 2.63, 459.9, 390.5, 1.18],
    ['Mistral-7B v0.3', '', 1, 4112, 3248, 1.27, 144.6, 133.3, 1.08],
    ['Qwen2.5-7B', '', 1, 4205, 3400, 1.24, 144.5, 131.5, 1.1],
    ['Llama-3.1-8B', '', 1, 4102, 3229, 1.27, 136.6, 125.8, 1.09],
    ['Qwen3-8B', '', 1, 3793, 3080, 1.23, 128, 119.8, 1.07],
    ['gemma-2-9b', '', 1, 3058, 2491, 1.23, 98.3, 92.3, 1.06],
    ['gemma-3-12b', '', 1, 2484, 2010, 1.24, 77.9, 75.2, 1.03],
    ['Qwen3-14B', '', 1, 2315, 1774, 1.31, 78.7, 72.4, 1.09],
    ['phi-4 (14B)', '', 1, 2289, 1847, 1.24, 78.8, 74.5, 1.06],
    ['gpt-oss-20b MXFP4', 'MoE', 1, 4820, 2714, 1.78, 191.1, 170.7, 1.12],
    ['DeepSeek-V2-Lite', 'MoE · MLA', 1, 4561, 2089, 2.18, 183.8, 171.4, 1.07],
    ['DeepSeek-Coder-V2-Lite', 'MoE · MLA', 1, 4518, 2026, 2.23, 184.1, 171.3, 1.07],
    ['Moonlight-16B-A3B', 'MoE · MLA', 1, 4632, 2121, 2.18, 176.3, 131, 1.35],
    ['gemma-3-27b', '', 2, 1020, 902, 1.13, 39.2, 38.7, 1.01],
    ['Qwen3-30B-A3B', 'MoE', 2, 3761, 1076, 3.5, 167.4, 153.5, 1.09],
    ['Qwen3-32B', '', 2, 1051, 796, 1.32, 37.3, 34.5, 1.08],
    ['Mixtral-8x7B', 'MoE', 3, 1723, 865, 1.99, 84.7, 81.2, 1.04],
    ['Llama-3.3-70B', '', 4, 457, 371, 1.23, 18.3, 17.5, 1.05],
    ['Qwen2.5-72B', '', 4, 455, 374, 1.22, 16.8, 16.2, 1.04],
    ['Qwen3-Next-80B-A3B', 'hybrid MoE', 4, 2058, 445, 4.62, 105.4, 93.9, 1.12],
    ['Kimi-Linear-48B-A3B', 'hybrid · MLA', 4, 3046, null, null, 111.9, null, null],
    ['gpt-oss-120b MXFP4', 'MoE', 5, 2257, 1235, 1.83, 130.2, 117.5, 1.11],
  ]
  // Llama-3.1-8B Q4_K_M: [batch, jitllm, llama.cpp, vLLM, vs llama.cpp, vs vLLM]
  const BATCH = [
    [16, 955, 723, 394, 1.32, 2.42],
    [32, 1288, 1034, 415, 1.25, 3.11],
    [64, 1860, 660, 436, 2.82, 4.27],
    [128, 2440, 1142, 441, 2.14, 5.53],
  ]
  // Time to first token at ~2048 prompt tokens, warm, median of 7, ms:
  // [model, tag, cards, jitllm, llama.cpp]
  const TTFT = [
    ['Llama-3.2-1B', '', 1, 118.7, 127.8],
    ['Llama-3.1-8B', '', 1, 556.8, 652.8],
    ['Qwen3-8B', '', 1, 585.4, 690.8],
    ['gemma-3-27b', '', 2, 1338, 1969],
    ['Qwen3-32B', '', 2, 1332, 1729],
    ['Qwen3-30B-A3B', 'MoE', 2, 590, 1677],
    ['Mixtral-8x7B', 'MoE', 3, 1314, 2555],
    ['Llama-3.3-70B', '', 4, 2222, 3015],
    ['Qwen2.5-72B', '', 4, 2160, 3080],
    ['Qwen3-Next-80B-A3B', 'hybrid MoE', 4, 1104, 4506],
    ['gpt-oss-120b MXFP4', 'MoE', 5, 1080, 1876],
  ]
  // [model, Metal prefill, Metal decode, CPU prefill, CPU decode]
  const M4 = [
    ['stories15M', 0.86, 1.53, 0.32, 0.95],
    ['SmolLM2-360M Q8_0', 0.92, 1.05, 0.86, 0.92],
    ['TinyLlama-1.1B q3_K_M', 1.02, 1.22, 0.89, 1.09],
    ['Llama-3.2-1B', 0.95, 0.97, 1.53, 0.97],
    ['Qwen2-1.5B', 0.95, 0.99, 1.67, 0.9],
    ['Qwen3-MoE-4x0.6B', 0.91, 0.97, 1.46, 0.93],
    ['gemma-2b', 0.89, 1.01, 0.85, 0.96],
  ]

  const fmt = (x) => `${x.toFixed(2)}×`
  const num = (x) => (x == null ? '—' : typeof x === 'string' ? x : x >= 1000 ? x.toLocaleString('en-US') : String(x))
  // A ratio as a bar on a shared scale, with the 1× line drawn across it.
  const ratio = (x, max, what) => {
    if (x == null) return `<span class="ratio"><span class="track"></span><b>—</b></span>`
    const up = x >= 1
    return (
      `<span class="ratio" data-tip="${what}: ${fmt(x)}"><span class="track">` +
      `<span class="rbar${up ? ' up' : ''}" style="width:${Math.min(100, (x / max) * 100).toFixed(1)}%"></span>` +
      `<span class="one" style="left:${((1 / max) * 100).toFixed(1)}%"></span></span>` +
      `<b class="${up ? 'up' : ''}">${fmt(x)}</b></span>`
    )
  }

  const v100 = root.querySelector('[data-board="v100"]')
  v100.innerHTML =
    `<table class="btable"><caption class="sr-only">jitllm against llama.cpp on NVIDIA V100, tokens per second and ratio</caption>` +
    `<thead><tr><th scope="col">Model</th><th scope="col">GPUs</th><th scope="col">Prefill, 512 tokens <span aria-hidden="true">(scale 0–5×)</span></th><th scope="col" class="n">jitllm</th><th scope="col" class="n">llama.cpp</th><th scope="col">Decode, 128 tokens <span aria-hidden="true">(0–1.5×)</span></th><th scope="col" class="n">jitllm</th><th scope="col" class="n">llama.cpp</th></tr></thead><tbody>` +
    V100.map(([m, tag, g, pj, pl, px, dj, dl, dx]) =>
      `<tr><td class="name">${m}${tag ? `<span class="tag">${tag}</span>` : ''}</td><td class="gpus">${g}× V100</td>` +
      `<td>${ratio(px, 5, `${m} prefill`)}</td><td class="n">${num(pj)}</td><td class="n">${num(pl)}</td>` +
      `<td>${ratio(dx, 1.5, `${m} decode`)}</td><td class="n">${num(dj)}</td><td class="n">${num(dl)}</td></tr>`,
    ).join('') +
    `</tbody></table>` +
    `<p class="bnote">Tesla V100-SXM2-16GB (sm_70, CUDA) on 2× Xeon E5-2680 v4; llama.cpp built on the box for sm_70; tok/s. 36 models, each engine back to back, one pass. Kimi-Linear-48B ran from a Hugging Face conversion with no GGUF, so it has no llama.cpp row.</p>`

  // Grouped bars: jitllm in the brand colour, the other two in neutral greys.
  const batch = root.querySelector('[data-board="batch"]')
  {
    const W = 640
    const H = 260
    const L = 44
    const B = 28
    const T = 16
    const max = 2500
    const y = (v) => T + (H - T - B) * (1 - v / max)
    const gw = (W - L) / BATCH.length
    const bw = 30
    const series = [['jitllm', 'var(--brand)'], ['llama.cpp', 'var(--muted)'], ['vLLM 0.18.1', 'var(--line-strong)']]
    let svg = `<svg class="bchart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Aggregate decode throughput by batch size for jitllm, llama.cpp and vLLM">`
    for (const v of [0, 500, 1000, 1500, 2000, 2500]) {
      svg += `<line x1="${L}" x2="${W}" y1="${y(v)}" y2="${y(v)}" stroke="var(--line)" stroke-width="1"/>`
      svg += `<text x="${L - 8}" y="${y(v) + 4}" text-anchor="end">${v}</text>`
    }
    BATCH.forEach(([n, ...vals], i) => {
      const cx = L + gw * i + gw / 2
      vals.slice(0, 3).forEach((v, s) => {
        const x = cx + (s - 1) * (bw + 2) - bw / 2
        const top = y(v)
        const h = H - B - top
        const tip = `batch ${n} · ${series[s][0]}: ${v} tok/s`
        svg += `<path class="mark" data-tip="${tip}" d="M${x},${H - B}V${top + 4}q0,-4 4,-4h${bw - 8}q4,0 4,4V${H - B}Z" fill="${series[s][1]}"/>`
        if (s === 0) svg += `<text class="val" x="${x + bw / 2}" y="${top - 6}" text-anchor="middle">${v}</text>`
      })
      svg += `<text x="${cx}" y="${H - 8}" text-anchor="middle">batch ${n}</text>`
    })
    svg += `</svg>`
    batch.innerHTML =
      `<div class="legend">${series.map(([n, c]) => `<span><i style="background:${c}"></i>${n}</span>`).join('')}<span style="margin-left:auto;color:var(--muted)">aggregate tok/s</span></div>` +
      svg +
      `<table class="btable"><caption class="sr-only">Batched decode throughput, tokens per second</caption><thead><tr><th scope="col">Batch</th><th scope="col" class="n">jitllm</th><th scope="col" class="n">llama.cpp</th><th scope="col" class="n">vLLM</th><th scope="col">vs llama.cpp</th><th scope="col">vs vLLM</th></tr></thead><tbody>` +
      BATCH.map(([n, j, l, v, xl, xv]) => `<tr><td>${n}</td><td class="n">${j}</td><td class="n">${l}</td><td class="n">${v}</td><td>${ratio(xl, 6, `batch ${n} vs llama.cpp`)}</td><td>${ratio(xv, 6, `batch ${n} vs vLLM`)}</td></tr>`).join('') +
      `</tbody></table>` +
      `<p class="bnote">Many independent sequences decoding together: one V100-SXM2-16GB, Llama-3.1-8B-Instruct Q4_K_M, a 6-token prompt then 128 tokens each, all three engines in the same pass. jitllm leads both at every batch size; vLLM ran the same GGUF.</p>`
  }

  const ttft = root.querySelector('[data-board="ttft"]')
  ttft.innerHTML =
    `<table class="btable"><caption class="sr-only">Time to first token on NVIDIA V100, milliseconds, jitllm against llama.cpp</caption>` +
    `<thead><tr><th scope="col">Model</th><th scope="col">GPUs</th><th scope="col">Sooner than llama.cpp <span aria-hidden="true">(scale 0–5×)</span></th><th scope="col" class="n">jitllm ms</th><th scope="col" class="n">llama.cpp ms</th></tr></thead><tbody>` +
    TTFT.map(([m, tag, g, j, l]) =>
      `<tr><td class="name">${m}${tag ? `<span class="tag">${tag}</span>` : ''}</td><td class="gpus">${g}× V100</td>` +
      `<td>${ratio(Math.round((l / j) * 100) / 100, 5, `${m}: llama.cpp ms ÷ jitllm ms`)}</td><td class="n">${num(j)}</td><td class="n">${num(l)}</td></tr>`,
    ).join('') +
    `</tbody></table>` +
    `<p class="bnote">A ~2048-token prompt in, one token out, tokenizing included; warm, median of 7. jitllm <code>speed -ttft</code> against <code>llama-server --no-cache-prompt</code> timed at the client, same Q4_K_M weights and cards. Over several cards jitllm pipelines the prompt, so the cards work at once. On a ~48-token prompt jitllm is 3–14% behind on one card.</p>`

  const m4 = root.querySelector('[data-board="m4"]')
  const cell = (x) => `<td>${ratio(x, 1.8, 'ratio')}</td>`
  m4.innerHTML =
    `<table class="btable"><caption class="sr-only">jitllm against llama.cpp b10985 on an Apple M4, ratios</caption><thead><tr><th scope="col">Model</th><th scope="col">Metal prefill</th><th scope="col">Metal decode</th><th scope="col">CPU prefill</th><th scope="col">CPU decode</th></tr></thead><tbody>` +
    M4.map(([m, ...r]) => `<tr><td class="name">${m}</td>${r.map(cell).join('')}</tr>`).join('') +
    `</tbody></table>` +
    `<p class="bnote">Mac mini M4 against llama.cpp b10985, one machine, one pass. Metal decode is ahead on four of seven models and CPU prefill on the three k-quant ones. Metal prefill and small-model CPU prefill, where llama.cpp runs Apple's Accelerate, are still behind.</p>`

  // Tabs.
  const tabs = [...root.querySelectorAll('[role="tab"]')]
  const show = (tab) => {
    for (const t of tabs) {
      const on = t === tab
      t.setAttribute('aria-selected', String(on))
      t.tabIndex = on ? 0 : -1
      document.getElementById(t.getAttribute('aria-controls')).hidden = !on
    }
  }
  tabs.forEach((t, i) => {
    t.addEventListener('click', () => show(t))
    t.addEventListener('keydown', (e) => {
      const d = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0
      if (!d) return
      const next = tabs[(i + d + tabs.length) % tabs.length]
      show(next)
      next.focus()
    })
  })
  show(tabs[0])

  // One tooltip for every mark that carries data-tip.
  const tip = document.createElement('div')
  tip.className = 'tip'
  tip.hidden = true
  document.body.appendChild(tip)
  root.addEventListener('pointermove', (e) => {
    const t = e.target.closest('[data-tip]')
    if (!t) return (tip.hidden = true)
    tip.textContent = t.dataset.tip
    tip.hidden = false
    tip.style.left = `${Math.min(e.clientX + 14, innerWidth - tip.offsetWidth - 8)}px`
    tip.style.top = `${e.clientY + 16}px`
  })
  root.addEventListener('pointerleave', () => (tip.hidden = true))
})()
