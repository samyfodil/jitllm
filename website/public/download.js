// The download chooser (src/components/Downloads.astro). Without this script
// every platform is listed and each link opens the latest release's page.
//
// It does two things. It puts the visitor's platform first, under "For your
// machine", and the others under "Other platforms". And it asks GitHub's API
// for the latest release once, pointing each link at that release's asset;
// asset names carry the version, so this is how a page with no version in it
// links a file. When the API does not answer, the links stay on the release
// page.
;(async () => {
  const roots = document.querySelectorAll('[data-downloads]')
  if (!roots.length) return

  // Safari reports neither the architecture nor Apple silicon, so a Mac is
  // taken as Apple silicon and the Intel image is the second link.
  let os = '', arch = ''
  try {
    const uad = navigator.userAgentData
    if (uad && uad.getHighEntropyValues) {
      const h = await uad.getHighEntropyValues(['platform', 'architecture'])
      os = (h.platform || '').toLowerCase()
      arch = (h.architecture || '').toLowerCase()
    }
  } catch {}
  const ua = (navigator.userAgent + ' ' + (navigator.platform || '')).toLowerCase()
  if (!os) os = ua
  const which = /mac|iphone|ipad/.test(os) ? 'mac' : /win/.test(os) ? 'windows' : /linux|x11|cros/.test(os) && !/android/.test(ua) ? 'linux' : ''
  if (!arch) arch = /arm|aarch64/.test(ua) ? 'arm' : ''
  const goarch = arch.startsWith('arm') ? 'arm64' : 'amd64'

  for (const root of roots) {
    // Windows shows the setup for the visitor's architecture; with JavaScript
    // off both are shown.
    for (const a of root.querySelectorAll('[data-os="windows"] .dl-main[data-arch]')) a.hidden = a.dataset.arch !== goarch
    if (!which) continue
    const mine = root.querySelector(`[data-os="${which}"]`)
    const label = root.querySelector('[data-dl-mine]')
    const other = root.querySelector('[data-dl-other]')
    if (!mine) continue
    mine.classList.add('dl-yours')
    label.hidden = false
    other.hidden = false
    root.prepend(label)
    label.after(mine)
    mine.after(other)
  }

  try {
    const r = await fetch('https://api.github.com/repos/jitllm/jitllm/releases/latest', { headers: { Accept: 'application/vnd.github+json' } })
    if (!r.ok) return
    const rel = await r.json()
    const v = String(rel.tag_name || '').replace(/^v/, '')
    const urls = new Map((rel.assets || []).map((x) => [x.name, x.browser_download_url]))
    for (const a of document.querySelectorAll('[data-downloads] a[data-asset]')) {
      const name = a.dataset.asset.replace('{v}', v)
      const url = urls.get(name)
      if (url) {
        a.href = url
        a.title = name
      }
    }
  } catch {}
})()
