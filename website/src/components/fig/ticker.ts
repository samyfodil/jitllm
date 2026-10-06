// Drives a figure's animation: step() every ms while the figure is on screen
// and playing. A reader who asked for reduced motion starts paused, and the
// Play button is how they start it.
export const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches

export function ticker(el: Element, step: () => void, ms: number, btn?: HTMLButtonElement | null) {
  let playing = !reduced
  let visible = false
  let id = 0
  const sync = () => {
    clearInterval(id)
    id = playing && visible ? window.setInterval(step, ms) : 0
    if (btn) {
      btn.textContent = playing ? 'Pause' : 'Play'
      btn.setAttribute('aria-pressed', String(playing))
    }
  }
  new IntersectionObserver(([e]) => {
    visible = e.isIntersecting
    sync()
  }).observe(el)
  btn?.addEventListener('click', () => {
    playing = !playing
    sync()
  })
  sync()
  return {
    get playing() {
      return playing
    },
    set(p: boolean) {
      playing = p
      sync()
    },
  }
}
