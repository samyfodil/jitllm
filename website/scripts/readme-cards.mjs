// Renders the cards at the top of the repository README from the built site:
// the site's own number styling, its "Watch it schedule" machine and two of
// the docs' figures, each framed as a dark card. It drives the pages' own
// scripts under a fake clock (Playwright's page.clock), so every frame of an
// animation is a fixed step of the page's time, not of how fast the capture
// runs, and seeds Math.random so a rerun draws the same frames.
//
// It is run by readme-cards.sh, which builds the site and finds Playwright;
// run alone it takes:
//   DIST        the built site (default: ../dist)
//   OUT         where the cards go (default: ../../docs/assets/readme)
//   PLAYWRIGHT  a directory whose node_modules holds playwright (default: here)
//   ONLY        a comma-separated subset of card names
//
// The perf card's numbers are the README's Performance section, word for
// word in what they claim; change them there first.
import { createServer } from 'node:http'
import { createRequire } from 'node:module'
import { readFile, mkdir, mkdtemp, rm, stat } from 'node:fs/promises'
import { spawnSync } from 'node:child_process'
import { tmpdir } from 'node:os'
import { join, extname, dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const DIST = resolve(process.env.DIST || join(here, '..', 'dist'))
const OUT = resolve(process.env.OUT || join(here, '..', '..', 'docs', 'assets', 'readme'))
const req = createRequire(join(resolve(process.env.PLAYWRIGHT || here), 'noop.js'))
const { chromium } = req('playwright')
const ONLY = process.env.ONLY ? new Set(process.env.ONLY.split(',')) : null

// The built site, served from memory-free reads of dist/.
const TYPES = { '.html': 'text/html', '.css': 'text/css', '.js': 'text/javascript', '.mjs': 'text/javascript', '.svg': 'image/svg+xml', '.png': 'image/png', '.json': 'application/json', '.woff2': 'font/woff2', '.webp': 'image/webp', '.txt': 'text/plain' }
const server = createServer(async (rq, rs) => {
  let p = join(DIST, decodeURIComponent(new URL(rq.url, 'http://x').pathname))
  if (!p.startsWith(DIST)) return rs.writeHead(403).end()
  try {
    if ((await stat(p)).isDirectory()) p = join(p, 'index.html')
    rs.writeHead(200, { 'content-type': TYPES[extname(p)] || 'application/octet-stream' }).end(await readFile(p))
  } catch {
    rs.writeHead(404).end()
  }
})
await new Promise((r) => server.listen(0, '127.0.0.1', r))
const SITE = `http://127.0.0.1:${server.address().port}/`

// A seeded Math.random (mulberry32), installed before any page script runs.
const SEED = `(() => { let a = 20260; Math.random = () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296 } })()`

// The card frame, in the site's own tokens, so every card carries the
// palette of the page it came from. The page body is replaced by it.
const CARD_CSS = `
  html, body { margin: 0 !important; padding: 0 !important; background: var(--bg) !important; }
  body > :not(#rc) { display: none !important; }
  #rc { position: fixed; top: 0; left: 0; z-index: 9999; overflow: hidden; box-sizing: border-box; display: flex; flex-direction: column; gap: 14px; padding: 22px 24px 20px;
        background: var(--bg); border: 1px solid var(--line-strong); color: var(--text);
        font-family: Geist, var(--sans, var(--sl-font, system-ui)); }
  #rc .rc-eye { margin: 0; font: 500 13px/1 'JetBrains Mono', var(--mono, var(--sl-font-mono, monospace)); letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); display: flex; align-items: center; gap: 8px; }
  #rc .rc-eye i { width: 8px; height: 8px; background: var(--brand); display: inline-block; }
  #rc .rc-title { margin: 0; font-size: 25px; line-height: 1.2; font-weight: 600; letter-spacing: -0.01em; color: var(--text); }
  #rc .rc-body { flex: 1; min-height: 0; display: flex; flex-direction: column; justify-content: center; }
  #rc .rc-foot { margin: 0; font-size: 15px; line-height: 1.45; color: var(--text-2); }
  #rc .rc-foot b { color: var(--text); font-weight: 500; }
  #rc .rc-mono { font: 13px/1.4 'JetBrains Mono', var(--mono, var(--sl-font-mono, monospace)); color: var(--muted); }
`

async function open(path, { w = 640, h = 420 } = {}) {
  const b = await browser.newContext({ viewport: { width: w + 200, height: h + 400 }, deviceScaleFactor: 2, colorScheme: 'dark', reducedMotion: 'no-preference' })
  const page = await b.newPage()
  await page.addInitScript(SEED)
  await page.clock.install()
  await page.goto(SITE + path, { waitUntil: 'networkidle' })
  await page.evaluate(() => document.fonts.ready)
  return page
}

// Builds the card around nodes moved out of the page (moved, not cloned, so
// the page's scripts keep driving them).
async function frame(page, { w, h, eye, title, foot, body }) {
  await page.evaluate(({ css, w, h, eye, title, foot, body }) => {
    const s = document.createElement('style')
    s.textContent = css
    document.head.append(s)
    const rc = document.createElement('div')
    rc.id = 'rc'
    rc.style.width = w + 'px'
    rc.style.height = h + 'px'
    rc.innerHTML = `<p class="rc-eye"><i></i>${eye}</p>${title ? `<p class="rc-title">${title}</p>` : ''}<div class="rc-body"></div>${foot ? `<p class="rc-foot">${foot}</p>` : ''}`
    window.__keep = [...document.body.children]
    document.body.append(rc)
    const into = rc.querySelector('.rc-body')
    if (typeof body === 'string') into.innerHTML = body
    else for (const sel of body) into.append(document.querySelector(sel))
  }, { css: CARD_CSS, w, h, eye, title, foot, body })
  return page.locator('#rc')
}

async function png(card, name) {
  const p = join(OUT, name + '.png')
  await card.screenshot({ path: p })
  return p
}

// Steps the page's clock and screenshots every frame, running `at[t]` when
// the clock reaches t ms; then encodes the frames with a palette made from
// all of them.
async function gif(page, card, name, { fps, ms, at = {}, width }) {
  const dir = await mkdtemp(join(tmpdir(), 'readme-cards-'))
  const step = 1000 / fps
  const cues = Object.entries(at).map(([t, f]) => [+t, f]).sort((a, b) => a[0] - b[0])
  let n = 0
  for (let t = 0; t < ms; t += step) {
    while (cues.length && cues[0][0] <= t) await cues.shift()[1]()
    await card.screenshot({ path: join(dir, `f${String(n++).padStart(4, '0')}.png`) })
    await page.clock.runFor(step)
  }
  const p = join(OUT, name + '.gif')
  const vf = `scale=${width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=full[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle`
  const r = spawnSync('ffmpeg', ['-y', '-loglevel', 'error', '-framerate', String(fps), '-i', join(dir, 'f%04d.png'), '-vf', vf, '-loop', '0', p], { stdio: 'inherit' })
  await rm(dir, { recursive: true, force: true })
  if (r.status !== 0) throw new Error(`ffmpeg failed on ${name}`)
  return p
}

const pause = async (page) => page.clock.pauseAt((await page.evaluate(() => Date.now())) + 20)

const cards = {
  // The README's Performance numbers, in the landing page's stat style.
  async perf() {
    const page = await open('')
    const stat = (n, l, c, upTo) => `<div class="stat"><p class="stat-n">${upTo ? '<small>up to</small>' : ''}${n}</p><p class="stat-l">${l}</p><p class="stat-c">${c}</p></div>`
    const body = `<div class="stats" style="grid-template-columns:repeat(3,1fr);border-top:1px solid var(--line-strong);column-gap:18px">${[
      stat('4.62×', "llama.cpp's prefill on Qwen3-Next-80B", '4× V100 (CUDA), 512-token prompt'),
      stat('35<span>/35</span>', 'models decode faster than on llama.cpp', 'V100, 0.4B to 120B, up to 1.35×'),
      stat('34<span>/35</span>', 'models prefill faster than on llama.cpp', 'V100, the 35th at parity'),
      stat('4.08×', 'sooner to the first token than llama.cpp', 'V100, first on 29 of 33', true),
      stat('5.53×', "vLLM 0.18.1's batched decode", 'V100, Llama-3.1-8B, 16-128 sequences', true),
      stat('1.53×', "llama.cpp's decode on Apple M4", 'Metal, 7 models', true),
    ].join('')}</div>`
    const card = await frame(page, {
      w: 640, h: 440, eye: 'performance', title: 'Measured against the engines you know', body,
      foot: 'jitllm ÷ the other engine, same machine, same pass, same GGUF; above 1.00× is faster. Every row: <b>docs/perf/current.md</b>.',
    })
    await page.addStyleTag({ content: '#rc .stat { padding: 12px 0 4px; } #rc .stat-n { font-size: 38px; } #rc .stat-n small { font-size: 14px; font-weight: 500; letter-spacing: 0; color: var(--muted); margin-right: 6px; } #rc .stat-l { font-size: 15px; line-height: 1.3; margin-top: 6px; } #rc .stat-c { font-size: 12.5px; line-height: 1.35; }' })
    return png(card, 'perf')
  },

  // "Watch it schedule": a second card arrives, the CPU-only mode empties
  // both, auto brings them back, and the card leaves again.
  async schedule() {
    const page = await open('#demo')
    await page.locator('[data-machine]').scrollIntoViewIfNeeded()
    await pause(page)
    await page.clock.runFor(5000) // the two models' first page-ins land
    const card = await frame(page, {
      w: 640, h: 440, eye: 'placement · paging · relocation', title: 'Blocks move between devices, KV with them',
      body: ['.machine-wrap'],
      foot: '<span class="rc-mono" data-sum></span><br><span data-say></span>',
    })
    await page.addStyleTag({ content: '#rc .machine-wrap { margin: -6px 0 -10px; } #rc [data-say] { color: var(--text); }' })
    // Mirror the panel's counters and log into the card's foot.
    await page.evaluate(() => {
      const [sum, log] = [document.querySelector('[data-summary]'), document.querySelector('[data-log]')]
      const copy = () => {
        document.querySelector('#rc [data-sum]').textContent = sum.textContent
        document.querySelector('#rc [data-say]').textContent = log.textContent
      }
      new MutationObserver(copy).observe(sum, { childList: true, characterData: true, subtree: true })
      new MutationObserver(copy).observe(log, { childList: true, characterData: true, subtree: true })
      copy()
    })
    const click = (sel) => () => page.evaluate((s) => document.querySelector(s).click(), sel)
    return gif(page, card, 'schedule', {
      fps: 12, ms: 17000, width: 800,
      at: {
        800: click('[data-add-gpu]'),
        4800: click('[data-mode="cpu"]'),
        8800: click('[data-mode="auto"]'),
        12800: click('[data-devices] .dev-chip:last-child button'),
      },
    })
  },

  // The JIT figure: the loop emitted for an AVX2+VNNI CPU, then for CUDA.
  async jit() {
    const page = await open('docs/concepts/how-it-works/')
    await pause(page)
    const card = await frame(page, {
      w: 640, h: 560, eye: 'jit', title: 'Kernels are written at load, for this machine',
      body: ['[data-fig="jit"]'],
      foot: 'Shape, weight format and instruction set go into the emitted loop. A sketch of its structure; <b>jitllm asm</b> prints the real kernel.',
    })
    await page.addStyleTag({ content: `#rc [data-fig="jit"] { margin: 0; } #rc [data-fig="jit"] figcaption, #rc [data-fig="jit"] .ctl:nth-of-type(2), #rc [data-fig="jit"] [data-replay], #rc [data-fig="jit"] .legend, #rc .jit-grid > :last-child { display: none; } #rc .jit-grid { grid-template-columns: 1fr; } #rc .jit-phases { margin-bottom: 0.6rem; } #rc .jit-code { min-height: 0; height: 15.6rem; font-size: 11.5px; overflow: hidden; }` })
    const host = (h) => () => page.evaluate((h) => document.querySelector(`[data-host="${h}"]`).click(), h)
    // The CPU's loop is written before the first frame, and again at the
    // end, so the GIF opens on a whole listing and loops without a jump.
    await page.clock.runFor(100)
    await host('vnni')()
    await page.clock.runFor(8000)
    await page.waitForTimeout(600) // the lines' CSS fade-in runs on real time
    return gif(page, card, 'jit', { fps: 8, ms: 16000, width: 800, at: { 1500: host('ptx'), 9500: host('vnni') } })
  },

  // Expert pages: each expert of a mixture is its own page.
  async experts() {
    const page = await open('docs/guides/memory/')
    await pause(page)
    const card = await frame(page, {
      w: 640, h: 560, eye: 'experts', title: 'A mixture reads only the experts it routes to',
      body: ['[data-fig="experts"]'],
      foot: 'Every expert is its own page in the <b>.jlm</b>, evicted least recently used, so a token reads only the routed experts that are not resident. A simulation of the policy.',
    })
    await page.addStyleTag({ content: '#rc [data-fig="experts"] { margin: 0; } #rc [data-fig="experts"] figcaption, #rc [data-fig="experts"] .ctl { display: none; } #rc [data-fig="experts"] .cells { --n: 8 !important; gap: 5px; } #rc [data-fig="experts"] .cell { height: 46px; font-size: 13px; }' })
    await page.clock.runFor(100)
    return gif(page, card, 'experts', { fps: 6, ms: 9000, width: 800 })
  },
}

await mkdir(OUT, { recursive: true })
const browser = await chromium.launch()
try {
  for (const [name, make] of Object.entries(cards)) {
    if (ONLY && !ONLY.has(name)) continue
    const p = await make()
    console.log(`${p}  ${((await stat(p)).size / 1024).toFixed(0)} KB`)
  }
} finally {
  await browser.close()
  server.close()
}
