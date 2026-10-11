// /llms-full.txt: every docs page's clean Markdown, in the order of the
// routes (src/nav.mjs), then any page the routes do not file.
import type { APIRoute } from 'astro'
import { clean, docs, pageURL } from '../markdown'
import { routes } from '../nav.mjs'

export const GET: APIRoute = async ({ site }) => {
  const base = import.meta.env.BASE_URL.replace(/\/$/, '')
  const origin = site ? site.origin : ''
  const all = await docs()
  const byId = new Map(all.map((d) => [d.id, d]))
  const order: string[] = []
  for (const r of routes) for (const i of r.items) if (i.slug !== undefined && !order.includes(i.slug)) order.push(i.slug)
  for (const d of all) if (!order.includes(d.id)) order.push(d.id)
  const parts = order.map((id) => {
    const d = byId.get(id)
    if (!d) throw new Error(`src/nav.mjs files ${id}, which is not a docs page`)
    return `<!-- ${origin}${pageURL(base, id)} -->\n\n${clean(d, base)}`
  })
  return new Response(parts.join('\n---\n\n'), { headers: { 'Content-Type': 'text/plain; charset=utf-8' } })
}
