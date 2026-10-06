package app

// The async -> UI bridge. Neither library provides one (no RunOnMain or
// PostTask), so a worker that needs the UI goroutine to do something (open a
// dialog, switch tab, rebuild after a theme change) posts it here.
//
// Values do not come this way: a value is a signal Set, which is safe from any
// goroutine. This queue is for actions, and it is deliberately small.

const uiQueueDepth = 64

// Post runs fn on the UI goroutine at the next frame. Safe from any goroutine.
//
// The RequestRedraw is mandatory: the loop is event-driven, so an idle window
// has no next frame until something asks for one.
//
// The send is non-blocking: a full queue drops the task rather than parking a
// decode loop on the UI goroutine.
func (s *Shell) Post(fn func()) {
	if fn == nil {
		return
	}
	select {
	case s.q <- fn:
	default:
	}
	if s.GPU != nil {
		s.GPU.RequestRedraw()
	}
}

// DrainQueue runs every queued task. Install it once with
// gogpuApp.OnUpdate(shell.DrainQueue); it runs on the main thread inside the
// frame, which is what makes everything it calls UI-safe.
func (s *Shell) DrainQueue(_ float64) {
	for {
		select {
		case fn := <-s.q:
			fn()
		default:
			return
		}
	}
}
