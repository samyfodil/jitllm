// Clean Markdown for a docs page: what /docs/<page>.md, llms.txt and
// llms-full.txt serve, so an agent can read a page without its HTML, its
// navigation or the site's components.
import { getCollection, type CollectionEntry } from 'astro:content'

type Doc = CollectionEntry<'docs'>

// pageURL is a page's rendered address under the site's base.
export function pageURL(base: string, id: string): string {
  return `${base}/${id}/`.replace(/\/+/g, '/')
}

// markdownURL is the clean Markdown beside it: /docs/foo/ -> /docs/foo.md.
export function markdownURL(base: string, id: string): string {
  return `${base}/${id}.md`.replace(/\/+/g, '/')
}

export async function docs(): Promise<Doc[]> {
  return (await getCollection('docs')).sort((a, b) => a.id.localeCompare(b.id))
}

const attr = (tag: string, name: string) => new RegExp(`${name}="([^"]*)"`).exec(tag)?.[1] ?? ''

// clean turns a page's MDX source into plain Markdown: imports and figure
// components dropped (each figure illustrates text that stays), Starlight's
// cards turned into headings and links, root-relative links given the base.
export function clean(entry: Doc, base: string): string {
  const out: string[] = [`# ${entry.data.title}`, '']
  if (entry.data.description) out.push(`> ${entry.data.description}`, '')
  let fence = false
  let card = false
  for (const line of (entry.body ?? '').split('\n')) {
    const t = line.trim()
    if (t.startsWith('```')) fence = !fence
    if (fence || t.startsWith('```')) {
      out.push(line)
      continue
    }
    if (/^import\s.+\sfrom\s/.test(t) || /^export\s/.test(t)) continue
    if (t.startsWith('<Card ')) {
      out.push(`### ${attr(t, 'title')}`, '')
      card = true
      continue
    }
    if (t === '</Card>') {
      out.push('')
      card = false
      continue
    }
    if (t === '<CardGrid>' || t === '</CardGrid>') continue
    if (t.startsWith('<LinkCard ')) {
      const d = attr(t, 'description')
      out.push(`- [${attr(t, 'title')}](${attr(t, 'href')})${d ? `: ${d}` : ''}`)
      continue
    }
    // A component on a line of its own is a figure.
    if (/^<[A-Z][A-Za-z]*(\s[^>]*)?\/>$/.test(t)) continue
    // A card's text is indented inside its tag, which Markdown would read as code.
    out.push(card ? t : line)
  }
  let md = out.join('\n').replace(/\n{3,}/g, '\n\n').trim() + '\n'
  // Root-relative links and pictures carry the site's base.
  if (base) md = md.replace(/\]\(\/(?!\/)/g, `](${base}/`)
  return md
}
