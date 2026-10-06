// @ts-check
import { defineConfig } from 'astro/config'
import starlight from '@astrojs/starlight'

// The landing page is src/pages/index.astro, a plain page with its own CSS and
// scripts (public/). Starlight owns only what is under src/content/docs, which
// lives at /docs.
export default defineConfig({
  integrations: [
    starlight({
      title: 'jitllm',
      description: 'An operating system for LLM inference: compiled for your machine, scheduled across every core and GPU, paged through memory.',
      editLink: { baseUrl: 'https://github.com/samyfodil/jitllm/edit/main/website/' },
      lastUpdated: false,
      favicon: '/favicon.svg',
      social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/samyfodil/jitllm' }],
      customCss: ['./src/styles/docs.css', './src/styles/figures.css'],
      components: {
        SiteTitle: './src/components/SiteTitle.astro',
        ThemeProvider: './src/components/ThemeProvider.astro',
        ThemeSelect: './src/components/ThemeSelect.astro',
      },
      head: [
        { tag: 'link', attrs: { rel: 'stylesheet', href: '/themes.css' } },
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
      sidebar: [
        {
          label: 'Start here',
          items: [
            { label: 'Overview', slug: 'docs' },
            { label: 'Install', slug: 'docs/install' },
            { label: 'Quickstart', slug: 'docs/get-started' },
            { label: 'Compared with other engines', slug: 'docs/compare' },
          ],
        },
        {
          label: 'Guides',
          items: [
            { label: 'Convert models', slug: 'docs/guides/convert' },
            { label: 'Run from the command line', slug: 'docs/guides/run' },
            { label: 'Serve an API', slug: 'docs/guides/serve' },
            { label: 'Devices and placement', slug: 'docs/guides/devices' },
            { label: 'Memory and paging', slug: 'docs/guides/memory' },
            { label: 'Embed it in Go', slug: 'docs/guides/go' },
            { label: 'The desktop app', slug: 'docs/guides/desktop' },
            { label: 'Troubleshooting', slug: 'docs/guides/troubleshooting' },
          ],
        },
        {
          label: 'Concepts',
          items: [{ label: 'How it works', slug: 'docs/concepts/how-it-works' }],
        },
        {
          label: 'Reference',
          items: [
            { label: 'Supported models', slug: 'docs/reference/models' },
            { label: 'Hardware support', slug: 'docs/reference/hardware' },
            { label: 'CLI reference', slug: 'docs/reference/cli' },
            { label: 'The .jlm format', slug: 'docs/reference/jlm' },
            { label: 'Benchmarks', slug: 'docs/reference/benchmarks' },
          ],
        },
      ],
    }),
  ],
})
