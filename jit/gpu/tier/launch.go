package tier

import "github.com/samyfodil/jitllm/jit/gpu/backend"

// launcher is a submission's launch path: its session, and the slice every
// launch's buffer list is copied into before the call reaches the session. A
// variadic slice handed to an interface method escapes, so a launch with its
// buffers listed at the call site was a heap object per dispatch -- every
// dispatch of a submission that issues its launches rather than replaying a
// graph (Vulkan and Metal always, CUDA when it captures). Through launch the
// caller's list stays on its stack and the copy is reused.
//
// It is a concrete type passed by pointer, never a func value: a call through
// a func value is opaque to escape analysis and its arguments escape too.
type launcher struct {
	// to is the session and the reused list, on the devTier. Behind a pointer
	// because escape analysis does not tell one field from another: the
	// session escapes into its own Launch, and as a field here it would carry
	// the pointers to the caller's locals below to the heap with it.
	to *launchTo
	// err is the submission's latched error; la launches nothing once it is
	// set.
	err *error
	// la's bookkeeping, shared with layersSession: fsrc is the float buffer
	// the current int8 activation was quantized from, recognised by a launch
	// that writes (a, ax), and memoK/memoSrc the ActF16 memo every launch
	// through la forgets.
	fsrc    *backend.Buf
	memoK   *backend.Kernel
	memoSrc *backend.Buf
	a, ax   backend.Buf
}

// launch is s.Launch through the reused buffer list.
func (l *launcher) launch(k backend.Kernel, groups, width int, bufs ...backend.Buf) error {
	t := l.to
	t.bufs = append(t.bufs[:0], bufs...)
	err := t.s.Launch(k, groups, width, t.bufs...)
	clear(t.bufs)
	return err
}

// launchTo is where a launcher sends its launches: the submission's session
// and the buffer list reused across submissions.
type launchTo struct {
	s    backend.Session
	bufs []backend.Buf
}

// la is the ordinary launch: threads in groups of 128, nothing once the
// submission has failed.
func (l *launcher) la(k backend.Kernel, threads int, bufs ...backend.Buf) {
	*l.memoK, *l.memoSrc = nil, nil
	if *l.err == nil {
		*l.err = l.launch(k, (threads+127)/128, 128, bufs...)
	}
	if len(bufs) == 3 && bufs[1] == l.a && bufs[2] == l.ax {
		*l.fsrc = bufs[0]
	}
}
