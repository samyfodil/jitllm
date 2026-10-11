// /llms.txt (llmstxt.org): a short index of the docs, generated from the
// pages themselves and filed by route (src/nav.mjs), each entry pointing at
// the page's clean Markdown. /llms-full.txt is every page's Markdown in one.
import type { APIRoute } from 'astro'
import { docs, markdownURL } from '../markdown'
import { routes } from '../nav.mjs'

const summary =
  'jitllm is an operating system for LLM inference: one self-contained executable that compiles its kernels for the machine at run time, places a model\'s blocks across every CPU core and GPU, and pages weights, experts and KV state through memory. It serves OpenAI- and Anthropic-compatible APIs and embeds as a Go library.'

const notes = [
  'The engine reads only `.jlm` containers: `jitllm convert <gguf|hf-dir|hf://repo|catalog-name> out.jlm` makes one, and a GGUF given to the engine is refused with that command.',
  'Put flags before the model path; a flag after it is refused.',
  '`jitllmd serve` serves `POST /v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `GET /v1/models` (OpenAI), `POST /v1/messages` (Anthropic) and `GET /healthz` on one address.',
  'Every recipe below is run as a test against a real server; start with one when integrating.',
  'Every page is also served as Markdown: replace a page\'s trailing slash with `.md`.',
]

export const GET: APIRoute = async ({ site }) => {
  const base = import.meta.env.BASE_URL.replace(/\/$/, '')
  const origin = site ? site.origin : ''
  const byId = new Map((await docs()).map((d) => [d.id, d]))
  const lines = ['# jitllm', '', `> ${summary}`, '', 'Things to know:', '', ...notes.map((n) => `- ${n}`), '']
  for (const r of routes) {
    lines.push(`## ${r.label}`, '', r.blurb, '')
    for (const i of r.items) {
      if (i.slug === undefined) {
        lines.push(`- [${i.label}](${i.link}): ${i.about}`)
        continue
      }
      const d = byId.get(i.slug)
      if (!d) throw new Error(`src/nav.mjs files ${i.slug}, which is not a docs page`)
      lines.push(`- [${d.data.title}](${origin}${markdownURL(base, d.id)}): ${d.data.description ?? ''}`)
    }
    lines.push('')
  }
  lines.push('## Optional', '', `- [Everything above in one file](${origin}${base}/llms-full.txt)`, '')
  return new Response(lines.join('\n'), { headers: { 'Content-Type': 'text/plain; charset=utf-8' } })
}
