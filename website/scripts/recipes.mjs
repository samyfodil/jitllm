// Copies the repository's tested recipes (docs/recipes/*.md) into the docs
// collection, so the site serves the same text the recipe test runs: one
// source, never a second copy edited by hand. Runs before every build and dev
// server (package.json); the output directory is ignored by git.
//
// Each file gains Starlight frontmatter from its first heading and paragraph,
// loses the test's HTML comments, and has its links pointed at the site's
// pages or, for anything else in the repository, at GitHub.

import { mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const src = join(here, '..', '..', 'docs', 'recipes')
const out = join(here, '..', 'src', 'content', 'docs', 'docs', 'recipes')
const blob = 'https://github.com/jitllm/jitllm/blob/main'

// Repository documents that have a page of their own on the site.
const onSite = {
  '../models.md': '/docs/reference/models/',
  '../../README.md#install': '/docs/install/',
}

function link(target) {
  if (/^[a-z]+:/.test(target) || target.startsWith('#') || target.startsWith('/')) return target
  if (onSite[target]) return onSite[target]
  const m = /^([a-z0-9-]+)\.md(#.*)?$/.exec(target)
  if (m) return m[1] === 'README' ? `/docs/recipes/${m[2] ?? ''}` : `/docs/recipes/${m[1]}/${m[2] ?? ''}`
  // Anything else is a path in the repository, relative to docs/recipes.
  const parts = ['docs', 'recipes']
  for (const p of target.split('/')) {
    if (p === '..') parts.pop()
    else if (p !== '.') parts.push(p)
  }
  return `${blob}/${parts.join('/')}`
}

function convert(md, name) {
  const lines = md.split('\n')
  const h1 = lines.findIndex((l) => l.startsWith('# '))
  if (h1 < 0) throw new Error(`docs/recipes/${name} has no title heading`)
  const title = lines[h1].slice(2).trim()
  let body = lines.slice(h1 + 1).join('\n')
  const para = body.trim().split(/\n\s*\n/)[0].replace(/\s+/g, ' ').replace(/\[([^\]]*)\]\([^)]*\)/g, '$1').replace(/^(.*?[.:])\s.*$/, '$1')
  // The test's directives are for the test.
  body = body.replace(/^<!-- test: .* -->\n/gm, '')
  body = body.replace(/\]\(([^)\s]+)\)/g, (_, t) => `](${link(t)})`)
  const yaml = (s) => JSON.stringify(s)
  return `---\ntitle: ${yaml(title)}\ndescription: ${yaml(para)}\n---\n${body}`
}

rmSync(out, { recursive: true, force: true })
mkdirSync(out, { recursive: true })
let n = 0
for (const f of readdirSync(src).filter((f) => f.endsWith('.md')).sort()) {
  const name = f === 'README.md' ? 'index.md' : f
  writeFileSync(join(out, name), convert(readFileSync(join(src, f), 'utf8'), f))
  n++
}
if (n === 0) throw new Error(`no recipe in ${src}`)
console.log(`recipes: ${n} page(s) from docs/recipes`)
