// /docs/<page>.md beside every rendered /docs/<page>/: the page as clean
// Markdown (src/markdown.ts), for agents and anyone reading without a browser.
import type { APIRoute, GetStaticPaths } from 'astro'
import { clean, docs } from '../markdown'

export const getStaticPaths: GetStaticPaths = async () =>
  (await docs()).map((entry) => ({ params: { slug: entry.id }, props: { entry } }))

export const GET: APIRoute = ({ props }) =>
  new Response(clean(props.entry, import.meta.env.BASE_URL.replace(/\/$/, '')), {
    headers: { 'Content-Type': 'text/markdown; charset=utf-8' },
  })
