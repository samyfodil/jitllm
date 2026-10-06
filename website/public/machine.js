// "Watch it schedule": an isometric machine, drawn in SVG, that the runtime's
// pager and scheduler play out on. It illustrates the policy; it is not
// telemetry. Placement decides which device runs each block, the fastest
// filling first. A block pages in from the .jlm on disk straight onto that
// device, and eviction drops the copy: the disk still has it. Blocks move
// between devices only when placement changes, which adding or removing a
// device or switching mode does, and a token walks the layers in order across
// every CPU/GPU seam. Block pages are evicted most recently used first. Each
// layer's KV cache sits on the device that runs the layer, grows with every
// token, and moves only with the layer.
;(() => {
  const svg = document.querySelector('[data-machine]')
  if (!svg) return
  const root = document.documentElement
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches
  const NS = 'http://www.w3.org/2000/svg'

  /* ------------------------------------------------------------ geometry */

  const U = 34
  const CX = Math.cos(Math.PI / 6)
  const SY = Math.sin(Math.PI / 6)
  const proj = (x, y, z) => [(x - z) * CX * U, ((x + z) * SY - y) * U]
  const pts = (...vs) => vs.map((v) => proj(...v).map((n) => n.toFixed(1)).join(',')).join(' ')
  // A transform that lays text flat on a horizontal plane at height y.
  const flat = (x, y, z) => {
    const [px, py] = proj(x, y, z)
    return `matrix(${CX.toFixed(4)} ${SY} ${(-CX).toFixed(4)} ${SY} ${px.toFixed(1)} ${py.toFixed(1)})`
  }

  const ROOM = { x: 14.6, z: 8.5 }
  const MAX_CPUS = 4
  const GPU_NAMES = ['gpu0', 'gpu1', 'gpu2']
  const GPU_SLOTS = 6
  const RAM_SLOTS = 12

  // Zones hold blocks in slots that fill from the back row outward. GPU zones
  // come and go with their cards, so they are looked up, not fixed.
  const ZONES = {
    disk: { x0: 0.5, z0: 0.5, w: 4, d: 5, y: 0, label: 'disk · .jlm' },
    ram: { x0: 5.5, z0: 0.5, w: 4, d: 3, y: 0, label: 'RAM' },
  }
  const gpuZone = (k) => ({ x0: 10.3, z0: 0.5 + k * 2.7, w: 3, d: 2, y: 0.22, label: `${GPU_NAMES[k]} · VRAM` })
  const cpuBox = (k) => {
    const x0 = k % 2 ? 7.9 : 5.7
    const z0 = k < 2 ? 4.3 : 6.4
    return { x0, z0, x1: x0 + 1.6, z1: z0 + 1.6 }
  }
  // Each device keeps its KV cache in a strip beside its frames, one stack per
  // model: RAM's along its front edge, a card's down its right side.
  const kvStrip = (dev) => {
    if (dev === 'ram') return { x0: 5.5, z0: 3.6, x1: 9.5, z1: 4.05, y: 0, along: 'x' }
    const Z = gpuZone(+dev.slice(3))
    return { x0: Z.x0 + Z.w + 0.2, z0: Z.z0, x1: Z.x0 + Z.w + 0.65, z1: Z.z0 + Z.d, y: Z.y, along: 'z' }
  }
  const kvSeg = (dev, c) => {
    const s = kvStrip(dev)
    const pad = 0.03
    if (s.along === 'x') {
      const w = (s.x1 - s.x0) / 3
      return { x0: s.x0 + c * w + pad, x1: s.x0 + (c + 1) * w - pad, z0: s.z0 + pad, z1: s.z1 - pad, y: s.y }
    }
    const d = (s.z1 - s.z0) / 3
    return { x0: s.x0 + pad, x1: s.x1 - pad, z0: s.z0 + c * d + pad, z1: s.z0 + (c + 1) * d - pad, y: s.y }
  }
  const segCentre = (g) => ({ x: (g.x0 + g.x1) / 2, y: g.y, z: (g.z0 + g.z1) / 2 })
  const slotAt = (zone, k) => {
    const Z = ZONES[zone]
    return { x: Z.x0 + (k % Z.w) + 0.5, y: Z.y, z: Z.z0 + Math.floor(k / Z.w) + 0.5 }
  }
  const HEIGHT_DISK = 0.14
  const HEIGHT_UP = 0.52
  const heightIn = (zone) => (zone === 'disk' ? HEIGHT_DISK : HEIGHT_UP)

  // Crop the view to the room once; it never changes size.
  {
    const cs = [[0, 0], [ROOM.x, 0], [0, ROOM.z], [ROOM.x, ROOM.z]].flatMap(([x, z]) => [proj(x, 0, z), proj(x, 2.2, z)])
    const xs = cs.map((p) => p[0])
    const ys = cs.map((p) => p[1])
    const m = 12
    const l = Math.min(...xs) - m
    const t = Math.min(...ys) - m
    svg.setAttribute('viewBox', `${l.toFixed(0)} ${t.toFixed(0)} ${(Math.max(...xs) + m - l).toFixed(0)} ${(Math.max(...ys) + m - t).toFixed(0)}`)
  }

  /* ------------------------------------------------------------ colour */

  let ink = {}
  const readInk = () => {
    const cs = getComputedStyle(root)
    const v = (n) => cs.getPropertyValue(n).trim()
    ink = {
      m: [v('--m1'), v('--m2'), v('--m3')],
      bg: v('--bg-deep'), line: v('--line'), strong: v('--line-strong'), muted: v('--muted'),
      text: v('--text'), surface: v('--surface'), surface2: v('--surface-2'), brand: v('--brand'),
      crest: v('--f-crest'), light: cs.colorScheme === 'light',
    }
  }
  const shade = (hex, k) => {
    const n = parseInt(hex.slice(1), 16)
    const f = (c) => Math.min(255, Math.round(c * k)).toString(16).padStart(2, '0')
    return `#${f(n >> 16)}${f((n >> 8) & 255)}${f(n & 255)}`
  }

  /* ------------------------------------------------------------ state */

  const CATALOG = [
    { name: 'llama-3.2-1b', n: 7 },
    { name: 'qwen3-moe', n: 6 },
    { name: 'gemma-2b', n: 6 },
  ]
  // A block always lives on disk, in the .jlm. A device holds a resident
  // copy in one of its frames; evicting it just drops that copy.
  const frames = { ram: Array(RAM_SLOTS).fill(null) }
  const diskUsed = Array(20).fill(false)
  let gpus = [] // [{ k, id }]
  let cpus = 1
  let mode = 'auto' // 'auto' | 'gpu' | 'cpu'
  let models = []
  let queueEnd = 0 // when the last queued transfer lands
  let run = null
  let requests = []
  let nextAuto = 0
  let active = null
  let hover = null
  let clock = 0
  const counts = { in: 0, out: 0, mig: 0, tok: 0 }
  let hops = [] // KV slices in flight between devices
  const MOVE_MS = reduced ? 0 : 420

  const isGpu = (zone) => zone?.startsWith('gpu')
  const freeFrame = (dev) => frames[dev].indexOf(null)
  const homeOf = (b) => slotAt('disk', b.home)
  const posOf = (b) => slotAt(b.res, b.slot)
  const devName = (dev) => (dev === 'ram' ? 'cpu' : GPU_NAMES[+dev.slice(3)])

  function hopKv(m, from, to, now) {
    if (!(from in frames) || !(to in frames) || reduced) return
    hops.push({ c: m.color, from: segCentre(kvSeg(from, m.color)), to: segCentre(kvSeg(to, m.color)), t0: Math.max(now, queueEnd), dur: MOVE_MS })
  }
  // How many KV pages a model holds on a device: one per few tokens of
  // context for every layer that runs there.
  const KV_CUBE = 0.26
  const KV_GAP = 0.05
  const kvPages = (m, dev) => {
    if (m.leaving || !m.kv) return 0
    const layers = m.blocks.filter((b) => b.dev === dev).length
    return Math.ceil((layers * m.kv) / 8)
  }

  function animate(b, kind, from, fromH, now, share = 0.6) {
    const t0 = Math.max(now, queueEnd)
    b.anim = { kind, from, fromH, t0, dur: MOVE_MS }
    queueEnd = t0 + MOVE_MS * share
  }
  // Disk to the block's own device, never through another one.
  function pageIn(b, now) {
    const slot = freeFrame(b.dev)
    frames[b.dev][slot] = b
    b.res = b.dev
    b.slot = slot
    animate(b, 'in', homeOf(b), HEIGHT_DISK, now)
    counts.in++
  }
  function evict(b, now) {
    const from = posOf(b)
    frames[b.res][b.slot] = null
    b.res = null
    animate(b, 'drop', from, HEIGHT_UP, now, 0.15)
    counts.out++
  }
  // Only a change of placement moves a resident block between devices.
  function relocate(b, dev, now) {
    const from = posOf(b)
    frames[b.res][b.slot] = null
    const slot = freeFrame(dev)
    frames[dev][slot] = b
    b.res = dev
    b.slot = slot
    animate(b, 'mig', from, HEIGHT_UP, now)
    counts.mig++
  }
  // Block pages are evicted MOST recently used first: a pass walks the layers
  // in order, so the block just used is the one needed last. `skip` says
  // which blocks may not go.
  function victim(dev, skip) {
    let v = null
    for (const b of frames[dev]) if (b && !skip(b) && (!v || b.used > v.used)) v = b
    return v
  }

  // Placement: which device runs each block. The fastest device fills first;
  // "all on GPU" over-commits the cards and lets them page from disk.
  function placement() {
    const devs = mode === 'cpu' ? [] : gpus.map((g) => g.id)
    const taken = Object.fromEntries(devs.map((d) => [d, 0]))
    let spill = 0
    const want = new Map()
    for (const m of models.filter((m) => !m.leaving))
      for (const b of m.blocks) {
        const d = devs.find((d) => taken[d] < GPU_SLOTS)
        if (d) taken[d]++, want.set(b, d)
        else want.set(b, mode === 'gpu' && devs.length ? devs[spill++ % devs.length] : 'ram')
      }
    return want
  }

  function replace(now) {
    const before = { ...counts }
    const want = placement()
    const movers = []
    for (const [b, dev] of want) {
      if (b.dev === dev) continue
      // The block's KV cache lives where it runs, so it moves with the placement.
      if (b.model.kv) hopKv(b.model, b.dev, dev, now)
      if (b.res) movers.push([b, dev])
      else b.dev = dev
    }
    for (const [b, dev] of movers) {
      if (freeFrame(dev) < 0) {
        const v = victim(dev, (x) => x.model === active || x.dev !== dev)
        if (v) evict(v, now)
      }
      if (freeFrame(dev) >= 0) relocate(b, dev, now)
      else evict(b, now)
      b.dev = dev
    }
    return { in: counts.in - before.in, out: counts.out - before.out, mig: counts.mig - before.mig }
  }

  const summary = (log) => {
    const parts = []
    if (log.in) parts.push(`paged in ${log.in} from disk`)
    if (log.mig) parts.push(`relocated ${log.mig}`)
    if (log.out) parts.push(`evicted ${log.out}`)
    return parts.length ? parts.join(', ') : 'already in place'
  }

  function load(now) {
    const spec = CATALOG.find((c) => !models.some((m) => m.name === c.name))
    if (!spec) return
    const free = diskUsed.map((u, i) => (u ? -1 : i)).filter((i) => i >= 0)
    if (free.length < spec.n) return
    const m = { name: spec.name, color: CATALOG.indexOf(spec), kv: 0, blocks: [] }
    free.slice(0, spec.n).forEach((home, i) => {
      diskUsed[home] = true
      m.blocks.push({ model: m, i, home, dev: 'ram', res: null, slot: -1, used: 0, flash: 0, arrive: Math.max(now, queueEnd) + i * 60 })
    })
    queueEnd = Math.max(now, queueEnd) + spec.n * 60
    models.push(m)
    replace(now)
    say(`${m.name}: ${spec.n} blocks mapped from .jlm`)
    request(m)
  }

  function unload(m, now) {
    if (run?.model === m) run = null
    requests = requests.filter((r) => r !== m)
    for (const b of m.blocks) if (b.res) evict(b, now)
    m.leaving = true
    m.gone = Math.max(now, queueEnd) + MOVE_MS + 200
    const log = replace(now)
    say(`${m.name}: unloaded${log.mig ? `, ${log.mig} blocks of the others relocated into its room` : ''}`)
  }

  function addGpu(now) {
    const k = [0, 1, 2].find((i) => !gpus.some((g) => g.k === i))
    if (k === undefined) return
    const id = `gpu${k}`
    ZONES[id] = gpuZone(k)
    frames[id] = Array(GPU_SLOTS).fill(null)
    gpus = [...gpus, { k, id }].sort((a, b) => a.k - b.k)
    drawStatic()
    say(`${GPU_NAMES[k]} added: ${summary(replace(now))}`)
    renderPanel()
  }

  function removeGpu(g, now) {
    gpus = gpus.filter((x) => x !== g)
    const log = replace(now)
    for (const b of frames[g.id]) if (b) evict(b, now)
    delete ZONES[g.id]
    delete frames[g.id]
    drawStatic()
    say(`${GPU_NAMES[g.k]} removed: ${summary(log)}, no state lost`)
    renderPanel()
  }

  function setCpus(n) {
    cpus = Math.max(1, Math.min(MAX_CPUS, n))
    drawStatic()
    say(`${cpus} CPU${cpus > 1 ? 's' : ''}: host blocks spread over ${cpus > 1 ? 'more cores' : 'one package'}`)
    renderPanel()
  }

  function setMode(next, now) {
    mode = next
    const label = { auto: 'auto', gpu: 'all on GPU', cpu: 'CPU only' }[mode]
    say(`placement ${label}: ${summary(replace(now))}`)
    renderPanel()
  }

  const request = (m) => {
    if (!requests.includes(m)) requests.push(m)
  }

  // How long one layer takes, for the picture: a card is quicker than a
  // host package, and more packages share the host blocks.
  const stepMs = (b) => (isGpu(b.res) ? 70 : 170 / Math.sqrt(cpus))

  // Bring a model's blocks in from disk before its pass, as far as that is
  // free: nothing it will need itself is evicted to make room.
  function prefetch(m, now) {
    const before = { ...counts }
    for (const b of m.blocks) {
      if (b.res) continue
      if (freeFrame(b.dev) < 0) {
        const v = victim(b.dev, (x) => x.model === m)
        if (!v) continue
        evict(v, now)
      }
      pageIn(b, now)
    }
    return { in: counts.in - before.in, out: counts.out - before.out, mig: 0 }
  }

  function startNext(now) {
    let m = requests.shift()
    if (!m) {
      if (reduced || now < nextAuto || !models.some((x) => !x.leaving)) return
      const pool = models.filter((x) => x !== active && !x.leaving)
      m = pool[Math.floor(Math.random() * pool.length)] || models.find((x) => !x.leaving)
    }
    active = m
    say(`request → ${m.name}: ${summary(prefetch(m, now))}`)
    run = { model: m, tokens: 4 + Math.floor(Math.random() * 4), k: -1, from: now + 150, t0: 0, t1: 0, cur: null }
    renderPanel()
  }

  // Advance the pass one layer at a time; a layer whose block is not resident
  // pages in from disk first, onto its own device.
  function step(now) {
    if (!run || now < run.t1) return
    const n = run.model.blocks.length
    run.k++
    // A finished pass is one more token of context in every layer's KV cache.
    if (run.k > 0 && run.k % n === 0) run.model.kv++, counts.tok++, renderPanel()
    if (run.k >= n * run.tokens) {
      run = null
      nextAuto = now + 1400 + Math.random() * 1800
      renderPanel()
      return
    }
    const b = run.model.blocks[run.k % n]
    let start = Math.max(now, run.from, queueEnd)
    if (!b.res) {
      if (freeFrame(b.dev) < 0) {
        const v = victim(b.dev, (x) => x === b)
        if (v) evict(v, now)
      }
      if (freeFrame(b.dev) >= 0) pageIn(b, now), (start = Math.max(start, b.anim.t0 + b.anim.dur))
      renderPanel()
    }
    run.cur = b
    run.t0 = start
    run.t1 = start + stepMs(b)
  }

  /* ------------------------------------------------------------ drawing */

  const el = (tag, attrs, parent) => {
    const e = document.createElementNS(NS, tag)
    for (const k in attrs) e.setAttribute(k, attrs[k])
    parent?.appendChild(e)
    return e
  }
  const staticG = el('g', {}, svg)
  const dynG = el('g', {}, svg)

  function slab(g, x0, z0, x1, z1, h, fill, attrs = {}) {
    const side = shade(ink.light ? '#c8c8c8' : ink.surface2, 0.8)
    el('polygon', { points: pts([x0, 0, z1], [x1, 0, z1], [x1, h, z1], [x0, h, z1]), fill: ink.surface2 }, g)
    el('polygon', { points: pts([x1, 0, z0], [x1, 0, z1], [x1, h, z1], [x1, h, z0]), fill: side }, g)
    return el('polygon', { points: pts([x0, h, z0], [x1, h, z0], [x1, h, z1], [x0, h, z1]), fill, stroke: ink.strong, 'stroke-width': 1, ...attrs }, g)
  }
  const label = (g, text, at, size = 10.5) => {
    const t = el('text', { transform: flat(...at), 'font-size': size, fill: ink.muted }, g)
    t.textContent = text
  }
  const bus = (g, a, b) =>
    el('polyline', { points: pts(a, b), stroke: ink.strong, 'stroke-width': 3, fill: 'none', 'stroke-dasharray': '1 5', 'stroke-linecap': 'round' }, g)

  function drawStatic() {
    staticG.replaceChildren()
    const { line, strong, muted, surface, surface2, bg } = ink
    el('polygon', { points: pts([0, 0, 0], [ROOM.x, 0, 0], [ROOM.x, 0, ROOM.z], [0, 0, ROOM.z]), fill: bg, stroke: strong, 'stroke-width': 1 }, staticG)
    for (let x = 1; x < ROOM.x; x++) el('polyline', { points: pts([x, 0, 0], [x, 0, ROOM.z]), stroke: line, 'stroke-width': 0.7, fill: 'none' }, staticG)
    for (let z = 1; z < ROOM.z; z++) el('polyline', { points: pts([0, 0, z], [ROOM.x, 0, z]), stroke: line, 'stroke-width': 0.7, fill: 'none' }, staticG)

    // Buses: NVMe from disk to RAM, DDR down a trunk to the CPUs, PCIe down a trunk to the cards.
    bus(staticG, [4.5, 0, 2], [5.5, 0, 2])
    label(staticG, 'NVMe', [4.6, 0, 1.75], 8.5)
    bus(staticG, [7.6, 0, 4.1], [7.6, 0, (cpus > 2 ? 7.2 : 5.1)])
    for (let k = 0; k < cpus; k++) {
      const c = cpuBox(k)
      const zm = (c.z0 + c.z1) / 2
      bus(staticG, [7.6, 0, zm], k % 2 ? [c.x0, 0, zm] : [c.x1, 0, zm])
    }
    label(staticG, 'DDR', [7.7, 0, 3.95], 8.5)
    if (gpus.length) {
      const last = gpuZone(gpus[gpus.length - 1].k)
      bus(staticG, [9.5, 0, 1.5], [9.9, 0, 1.5])
      bus(staticG, [9.9, 0, 1.5], [9.9, 0, last.z0 + 1])
      for (const g of gpus) {
        const Z = gpuZone(g.k)
        bus(staticG, [9.9, 0, Z.z0 + 1], [10.1, 0, Z.z0 + 1])
      }
      label(staticG, 'PCIe', [9.55, 0, 1.25], 8.5)
    }

    for (const g of gpus) {
      const Z = gpuZone(g.k)
      slab(staticG, Z.x0 - 0.2, Z.z0 - 0.2, Z.x0 + Z.w + 0.85, Z.z0 + Z.d + 0.2, Z.y, surface, { 'data-card': g.id })
    }
    for (const [key, Z] of Object.entries(ZONES)) {
      const x1 = Z.x0 + Z.w
      const z1 = Z.z0 + Z.d
      el('polygon', { points: pts([Z.x0, Z.y, Z.z0], [x1, Z.y, Z.z0], [x1, Z.y, z1], [Z.x0, Z.y, z1]), fill: 'none', stroke: muted, 'stroke-width': 1, 'stroke-dasharray': '4 4' }, staticG)
      label(staticG, Z.label, key === 'ram' ? [4.66, 0, 3.4] : [Z.x0, isGpu(key) ? 0 : Z.y, z1 + (isGpu(key) ? 0.62 : 0.42)])
    }

    // KV strips: where each device keeps the attention history of what it runs.
    for (const dev of ['ram', ...gpus.map((g) => g.id)]) {
      const k = kvStrip(dev)
      el('polygon', { points: pts([k.x0, k.y, k.z0], [k.x1, k.y, k.z0], [k.x1, k.y, k.z1], [k.x0, k.y, k.z1]), fill: 'none', stroke: muted, 'stroke-width': 0.8, 'stroke-dasharray': '2 3' }, staticG)
      if (dev === 'ram') label(staticG, 'KV', [4.84, 0, 4.0], 9)
      else label(staticG, 'KV', [k.x0 + 0.02, k.y, k.z1 + 0.36], 9)
    }

    // The CPUs: packages with pins.
    for (let k = 0; k < cpus; k++) {
      const { x0, z0, x1, z1 } = cpuBox(k)
      const h = 0.16
      for (let i = 0; i < 5; i++) {
        const x = x0 + 0.22 + i * ((x1 - x0 - 0.44) / 4)
        el('polyline', { points: pts([x, 0.02, z1], [x, 0.02, z1 + 0.16]), stroke: strong, 'stroke-width': 1.5 }, staticG)
      }
      slab(staticG, x0, z0, x1, z1, h, surface, { 'data-cpu': k })
      el('polygon', { points: pts([x0 + 0.4, h, z0 + 0.4], [x1 - 0.4, h, z0 + 0.4], [x1 - 0.4, h, z1 - 0.4], [x0 + 0.4, h, z1 - 0.4]), fill: surface2 }, staticG)
      label(staticG, `cpu${k}`, [x0 + 0.48, h, z0 + 0.95], 9)
    }
  }

  const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2)
  // Where a resident copy is drawn, and how tall: a page-in rises off its
  // disk slab and grows on the way; a relocation hops device to device.
  function where(b, now) {
    const to = posOf(b)
    const a = b.anim
    if (!a || a.kind === 'drop' || now >= a.t0 + a.dur) return { ...to, h: HEIGHT_UP }
    if (now < a.t0) return { ...a.from, h: a.fromH, waiting: a.kind === 'in' }
    const k = ease((now - a.t0) / a.dur)
    return {
      x: a.from.x + (to.x - a.from.x) * k,
      y: a.from.y + (to.y - a.from.y) * k + Math.sin(k * Math.PI) * 1.1,
      z: a.from.z + (to.z - a.from.z) * k,
      h: a.fromH + (HEIGHT_UP - a.fromH) * k,
      moving: true,
    }
  }

  function box(g, { x, y, z }, h, col, flash, opacity, text) {
    const r = 0.4
    const [x0, x1, z0, z1] = [x - r, x + r, z - r, z + r]
    const y1 = y + h
    const grp = el('g', { opacity }, g)
    el('polygon', { points: pts([x1, y, z0], [x1, y1, z0], [x1, y1, z1], [x1, y, z1]), fill: shade(col, 0.72) }, grp)
    el('polygon', { points: pts([x0, y, z1], [x1, y, z1], [x1, y1, z1], [x0, y1, z1]), fill: shade(col, 0.56) }, grp)
    el('polygon', { points: pts([x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]), fill: flash ? ink.crest : col, stroke: shade(col, 0.45), 'stroke-width': 0.6 }, grp)
    if (text) {
      const t = el('text', { transform: flat(x0 + 0.14, y1, z0 + 0.55), 'font-size': 9, fill: 'rgba(0,0,0,0.62)' }, grp)
      t.textContent = text
    }
  }

  function slabBox(g, { x0, x1, z0, z1, y }, h, col, opacity) {
    const y1 = y + h
    const grp = el('g', { opacity }, g)
    el('polygon', { points: pts([x1, y, z0], [x1, y1, z0], [x1, y1, z1], [x1, y, z1]), fill: shade(col, 0.72) }, grp)
    el('polygon', { points: pts([x0, y, z1], [x1, y, z1], [x1, y1, z1], [x0, y1, z1]), fill: shade(col, 0.56) }, grp)
    el('polygon', { points: pts([x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]), fill: col, stroke: shade(col, 0.45), 'stroke-width': 0.5 }, grp)
  }

  function draw(now) {
    dynG.replaceChildren()
    for (const m of models.filter((m) => m.gone && now > m.gone)) {
      for (const b of m.blocks) diskUsed[b.home] = false
      if (active === m) active = null
      models = models.filter((x) => x !== m)
      renderPanel()
    }

    step(now)
    const cur = run && now >= run.t0 ? run.cur : null
    if (cur) (cur.used = ++clock), (cur.flash = now)
    const prev = cur && cur.i ? run.model.blocks[cur.i - 1] : null

    const items = []
    for (const m of models) {
      const col = ink.m[m.color]
      const dim = hover && hover !== m ? 0.28 : 1
      for (const b of m.blocks) {
        // The .jlm copy: always on disk, whatever else holds the block.
        if (now >= b.arrive) {
          const home = homeOf(b)
          const k = Math.min(1, (now - b.arrive) / (MOVE_MS * 0.6 || 1))
          const y = (1 - ease(k)) * 2
          const fade = m.gone ? Math.max(0, (m.gone - now) / (MOVE_MS + 200 || 1)) : 1
          items.push({ d: home.x + home.z, draw: () => box(dynG, { ...home, y }, HEIGHT_DISK, col, false, dim * fade * (b.res ? 0.55 : 1), '') })
        }
        const a = b.anim
        // An evicted copy fades where it stood; the disk slab never moved.
        if (!b.res && a?.kind === 'drop' && now < a.t0 + a.dur) {
          const k = now < a.t0 ? 0 : (now - a.t0) / a.dur
          items.push({ d: a.from.x + a.from.z, draw: () => box(dynG, a.from, HEIGHT_UP * (1 - 0.6 * k), col, false, dim * (1 - k), `L${b.i}`) })
        }
        if (!b.res) continue
        const p = where(b, now)
        if (p.waiting) continue
        const flash = now - b.flash < 120
        items.push({ d: p.x + p.z + (p.moving ? 0.6 : 0.02), draw: () => box(dynG, p, p.h, col, flash, dim, p.h > 0.3 ? `L${b.i}` : '') })
      }
    }
    // KV pages as small cubes in each device's strip, filling a layer at a
    // time, and cubes hopping between devices when a layer relocates.
    for (const m of models) {
      const col = ink.m[m.color]
      const op = hover && hover !== m ? 0.28 : 1
      for (const dev of Object.keys(frames)) {
        const n = kvPages(m, dev)
        if (!n) continue
        const g = kvSeg(dev, m.color)
        const step = KV_CUBE + KV_GAP
        const cols = Math.max(1, Math.floor((g.x1 - g.x0 + KV_GAP) / step))
        const rows = Math.max(1, Math.floor((g.z1 - g.z0 + KV_GAP) / step))
        const shown = Math.min(n, cols * rows * 2)
        for (let i = 0; i < shown; i++) {
          const lvl = Math.floor(i / (cols * rows))
          const x0 = g.x0 + (i % cols) * step
          const z0 = g.z0 + (Math.floor(i / cols) % rows) * step
          const cube = { x0, x1: x0 + KV_CUBE, z0, z1: z0 + KV_CUBE, y: g.y + lvl * KV_CUBE }
          items.push({ d: x0 + z0 + KV_CUBE + lvl * 0.01, draw: () => slabBox(dynG, cube, KV_CUBE, col, op) })
        }
      }
    }
    hops = hops.filter((p) => now < p.t0 + p.dur)
    for (const p of hops) {
      if (now < p.t0) continue
      const k = ease((now - p.t0) / p.dur)
      const c = { x: p.from.x + (p.to.x - p.from.x) * k, y: p.from.y + (p.to.y - p.from.y) * k + Math.sin(k * Math.PI) * 1.3, z: p.from.z + (p.to.z - p.from.z) * k }
      const r = KV_CUBE / 2
      items.push({ d: c.x + c.z + 0.7, draw: () => slabBox(dynG, { x0: c.x - r, x1: c.x + r, z0: c.z - r, z1: c.z + r, y: c.y }, KV_CUBE, ink.m[p.c], 1) })
    }
    items.sort((a, b) => a.d - b.d)
    for (const it of items) it.draw()

    // The token's hop from one layer to the next, bright where it crosses a device boundary.
    if (cur?.res && prev?.res) {
      const a = where(prev, now)
      const b = where(cur, now)
      const seam = prev.res !== cur.res
      el('polyline', { points: pts([a.x, a.y + a.h + 0.05, a.z], [b.x, b.y + b.h + 0.05, b.z]), stroke: seam ? ink.crest : ink.text, 'stroke-width': seam ? 2.2 : 1.4, fill: 'none', opacity: 0.9, 'stroke-linecap': 'round' }, dynG)
    }
    // Light the device doing the work: the card holding the block, or the
    // host packages.
    const hot = shade(ink.brand, ink.light ? 1 : 0.55)
    // A host block runs across every core at once, so all packages light.
    for (const p of staticG.querySelectorAll('[data-cpu]')) p.setAttribute('fill', cur?.res === 'ram' ? hot : ink.surface)
    for (const p of staticG.querySelectorAll('[data-card]')) p.setAttribute('fill', cur?.res === p.dataset.card ? hot : ink.surface)

    if (!run && queueEnd <= now) startNext(now)
  }

  /* ------------------------------------------------------------ panel */

  const $ = (s) => document.querySelector(s)
  const list = $('[data-models]')
  const loadBtn = $('[data-load]')
  const devList = $('[data-devices]')
  const addCpuBtn = $('[data-add-cpu]')
  const addGpuBtn = $('[data-add-gpu]')
  const modeBtns = [...document.querySelectorAll('[data-mode]')]
  const logEl = $('[data-log]')
  const sumEl = $('[data-summary]')
  const say = (text) => (logEl.textContent = text)

  function renderPanel() {
    hover = null
    list.replaceChildren()
    for (const m of models) {
      const c = { gpu: 0, ram: 0, disk: 0 }
      for (const b of m.blocks) c[!b.res ? 'disk' : isGpu(b.res) ? 'gpu' : 'ram']++
      const on = [...new Set(m.blocks.map((b) => devName(b.dev)))].join(' + ')
      const n = m.blocks.length
      const col = ink.m[m.color]
      const li = document.createElement('li')
      const serving = active === m && run
      const state = m.leaving ? 'unloading' : serving ? 'serving' : c.disk === n ? 'paged out' : 'resident'
      li.className = 'mp-item'
      li.innerHTML =
        `<button class="mp-model" type="button"${serving ? ' data-active' : ''}>` +
        `<span class="mp-row"><span class="sw" style="background:${col}"></span>${m.name}` +
        `<span class="st"${serving ? ' data-on' : ''}>${state}</span></span>` +
        `<span class="mp-bar"><i style="width:${(c.gpu / n) * 100}%;background:${col}"></i><i style="width:${(c.ram / n) * 100}%;background:${shade(col, 0.6)}"></i></span>` +
        `<span class="mp-where"><span>${c.gpu} gpu</span><span>${c.ram} ram</span><span>${c.disk} on disk only</span></span><span class="mp-where"><span>runs on ${on}</span><span>kv ${m.kv * 32} tok</span></span></button>` +
        `<button class="mp-x" type="button" aria-label="Unload ${m.name}">×</button>`
      const [btn, x] = li.children
      btn.addEventListener('click', () => {
        if (m.leaving) return
        request(m)
        say(`queued a request for ${m.name}`)
      })
      x.addEventListener('click', () => !m.leaving && (unload(m, performance.now()), renderPanel()))
      btn.addEventListener('mouseenter', () => (hover = m))
      btn.addEventListener('mouseleave', () => (hover = null))
      list.appendChild(li)
    }
    loadBtn.disabled = models.length >= CATALOG.length
    loadBtn.textContent = loadBtn.disabled ? 'All loaded' : '+ Load model'

    devList.replaceChildren()
    const chip = (name, onRemove, canRemove) => {
      const s = document.createElement('span')
      s.className = 'dev-chip'
      s.innerHTML = `${name}<button type="button" aria-label="Remove ${name}"${canRemove ? '' : ' disabled'}>×</button>`
      s.lastChild.addEventListener('click', onRemove)
      devList.appendChild(s)
    }
    for (let k = 0; k < cpus; k++) chip(`cpu${k}`, () => setCpus(cpus - 1), cpus > 1 && k === cpus - 1)
    for (const g of gpus) chip(GPU_NAMES[g.k], () => removeGpu(g, performance.now()), true)
    addCpuBtn.disabled = cpus >= MAX_CPUS
    addGpuBtn.disabled = gpus.length >= GPU_NAMES.length
    for (const b of modeBtns) {
      b.setAttribute('aria-checked', String(b.dataset.mode === mode))
      b.disabled = b.dataset.mode === 'gpu' && !gpus.length
    }

    const n = (k, one) => `${k} ${one}${k === 1 ? '' : 's'}`
    sumEl.textContent = `${n(counts.in, 'page-in')} · ${n(counts.out, 'eviction')} · ${n(counts.mig, 'relocation')} · ${n(counts.tok * 32, 'token')}`
  }

  loadBtn.addEventListener('click', () => (load(performance.now()), renderPanel()))
  addCpuBtn.addEventListener('click', () => setCpus(cpus + 1))
  addGpuBtn.addEventListener('click', () => addGpu(performance.now()))
  for (const b of modeBtns) b.addEventListener('click', () => b.dataset.mode !== mode && setMode(b.dataset.mode, performance.now()))

  /* ------------------------------------------------------------ loop */

  readInk()
  addEventListener('jitllm:theme', () => (readInk(), drawStatic(), renderPanel(), draw(performance.now())))

  let frame = 0
  const loop = (now) => {
    frame = requestAnimationFrame(loop)
    draw(now)
  }
  new IntersectionObserver(([e]) => {
    if (e.isIntersecting && !frame) frame = requestAnimationFrame(loop)
    else if (!e.isIntersecting && frame) cancelAnimationFrame(frame), (frame = 0)
  }).observe(svg)

  // One CPU and one card to start, two models loaded; the rest is up to the visitor.
  const t0 = performance.now()
  gpus = [{ k: 0, id: 'gpu0' }]
  ZONES.gpu0 = gpuZone(0)
  frames.gpu0 = Array(GPU_SLOTS).fill(null)
  drawStatic()
  load(t0)
  load(t0)
  renderPanel()
  draw(t0)
})()
