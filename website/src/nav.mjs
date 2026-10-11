// The docs' three routes -- use, operate, contribute -- in one place: the
// sidebar (astro.config.mjs) and the llms.txt index (src/pages/llms.txt.ts)
// both read this list, so a page is filed once.
//
// An item is a docs page by slug, or an outside link ({ label, link }).

const repo = 'https://github.com/jitllm/jitllm/blob/main'

export const routes = [
  {
    label: 'Use jitllm',
    blurb: 'Install it, get a model, and connect an application.',
    items: [
      { label: 'Overview', slug: 'docs' },
      { label: 'Install', slug: 'docs/install' },
      { label: 'Quickstart', slug: 'docs/get-started' },
      { label: 'Recipe: start a server', slug: 'docs/recipes/start-a-server' },
      { label: 'Recipe: OpenAI-compatible clients', slug: 'docs/recipes/openai-clients' },
      { label: 'Recipe: tools and structured output', slug: 'docs/recipes/tools-and-structured-output' },
      { label: 'Recipe: choose and convert a checkpoint', slug: 'docs/recipes/convert-a-checkpoint' },
      { label: 'Serve an API', slug: 'docs/guides/serve' },
      { label: 'Convert models', slug: 'docs/guides/convert' },
      { label: 'Run from the command line', slug: 'docs/guides/run' },
      { label: 'Embed it in Go', slug: 'docs/guides/go' },
      { label: 'The desktop app', slug: 'docs/guides/desktop' },
      { label: 'The terminal app', slug: 'docs/guides/terminal' },
      { label: 'Supported models', slug: 'docs/reference/models' },
      { label: 'Compared with other engines', slug: 'docs/compare' },
    ],
  },
  {
    label: 'Operate jitllm',
    blurb: 'Memory, devices, diagnosis and the reference for running it.',
    items: [
      { label: 'Recipe: memory budgets and placement', slug: 'docs/recipes/memory-and-placement' },
      { label: 'Recipe: diagnose a slow or failed request', slug: 'docs/recipes/diagnose' },
      { label: 'All recipes, and how they are tested', slug: 'docs/recipes' },
      { label: 'Devices and placement', slug: 'docs/guides/devices' },
      { label: 'Memory and paging', slug: 'docs/guides/memory' },
      { label: 'Troubleshooting', slug: 'docs/guides/troubleshooting' },
      { label: 'Hardware support', slug: 'docs/reference/hardware' },
      { label: 'CLI reference', slug: 'docs/reference/cli' },
      { label: 'Benchmarks', slug: 'docs/reference/benchmarks' },
    ],
  },
  {
    label: 'Contribute to jitllm',
    blurb: 'How the engine works inside, and the rules a change is held to.',
    items: [
      { label: 'How it works', slug: 'docs/concepts/how-it-works' },
      { label: 'The .jlm format', slug: 'docs/reference/jlm' },
      { label: 'Project rules (AGENTS.md)', link: `${repo}/AGENTS.md`, about: 'the binding rules: measurement, correctness gates, architecture scope' },
      { label: 'Contributing', link: `${repo}/CONTRIBUTING.md`, about: 'how to send a change' },
      { label: 'Testing', link: `${repo}/docs/testing.md`, about: 'how tests find models, and every model-selecting variable' },
      { label: 'Agent evaluations', link: `${repo}/dev/agenteval/README.md`, about: 'tasks that measure whether an agent can use and change jitllm' },
    ],
  },
]

// sidebar is routes in Starlight's sidebar shape.
export const sidebar = routes.map((r) => ({
  label: r.label,
  items: r.items.map((i) => (i.slug !== undefined ? { label: i.label, slug: i.slug } : { label: i.label, link: i.link, attrs: { target: '_blank' } })),
}))
