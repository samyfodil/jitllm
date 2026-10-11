// @ts-check
import { defineConfig } from 'astro/config'
import starlight from '@astrojs/starlight'
import { satteri } from '@astrojs/markdown-satteri'
import { sidebar } from './src/nav.mjs'

// The landing page is src/pages/index.astro, a plain page with its own CSS and
// scripts (public/). Starlight owns only what is under src/content/docs, which
// lives at /docs. Beside each page, /docs/<page>.md serves it as clean Markdown,
// and /llms.txt and /llms-full.txt index them (src/pages/*.ts). The recipes
// are copied in from docs/recipes before each build (scripts/recipes.mjs).

// SITE and BASE_PATH are set by the Pages workflow: BASE_PATH is /jitllm on a
// GitHub project page and empty on a custom domain or a local build.
const base = (process.env.BASE_PATH ?? '').replace(/\/$/, '')

// The docs link to the site's own pages and pictures from its root ("/docs/...",
// "/shots/..."); this prefixes them with the base in Markdown links, raw HTML
// and MDX components alike. Starlight appends its own plugins to this processor.
const prefix = (v) => (typeof v === 'string' && v.startsWith('/') && !v.startsWith('//') ? base + v : v)
const prefixAttrs = (node) => {
  for (const a of node.attributes) if (a.name === 'href' || a.name === 'src') a.value = prefix(a.value)
  return node
}
const basePlugin = {
  name: 'jitllm-base',
  element: {
    filter: ['a', 'img'],
    visit(node, ctx) {
      for (const k of ['href', 'src']) if (node.properties?.[k]) ctx.setProperty(node, k, prefix(node.properties[k]))
    },
  },
  mdxJsxFlowElement: { filter: ['LinkCard', 'img', 'a'], visit: (node) => prefixAttrs({ ...node, attributes: node.attributes.map((a) => ({ ...a })) }) },
  mdxJsxTextElement: { filter: ['a', 'img'], visit: (node) => prefixAttrs({ ...node, attributes: node.attributes.map((a) => ({ ...a })) }) },
  raw: (node) => ({ ...node, value: node.value.replace(/\b(href|src)="\/(?!\/)/g, `$1="${base}/`) }),
}

export default defineConfig({
  site: process.env.SITE,
  base: base || undefined,
  markdown: { processor: satteri({ hastPlugins: base ? [basePlugin] : [] }) },
  integrations: [
    starlight({
      title: 'jitllm',
      description: 'An operating system for LLM inference: compiled for your machine, scheduled across every core and GPU, paged through memory.',
      editLink: { baseUrl: 'https://github.com/jitllm/jitllm/edit/main/website/' },
      lastUpdated: false,
      favicon: '/favicon.svg',
      social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/jitllm/jitllm' }],
      customCss: ['./src/styles/docs.css', './src/styles/figures.css'],
      components: {
        SiteTitle: './src/components/SiteTitle.astro',
        ThemeProvider: './src/components/ThemeProvider.astro',
        ThemeSelect: './src/components/ThemeSelect.astro',
      },
      head: [
        { tag: 'link', attrs: { rel: 'stylesheet', href: `${base}/themes.css` } },
        { tag: 'link', attrs: { rel: 'preconnect', href: 'https://fonts.googleapis.com' } },
        { tag: 'link', attrs: { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: true } },
        {
          tag: 'link',
          attrs: {
            rel: 'stylesheet',
            href: 'https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap',
          },
        },
      ],
      // The use / operate / contribute routes, shared with llms.txt.
      sidebar,
    }),
  ],
})
