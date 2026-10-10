package app

import (
	"github.com/jitllm/jitllm/common/api"
	"github.com/jitllm/jitllm/server"
)

// ServeAPI serves e, the window's own engine, as jitllm's API while the API
// setting is on. The address is read when the setting turns on, not on every
// keystroke in its field; turning the API off and on again applies a new one.
// The API's model folder follows the first of the window's.
//
// Call it before the engine's own OnShutdown, so the API stops taking requests
// before the engine closes.
func (s *Shell) ServeAPI(e *server.Engine) {
	dir := func() {
		if dirs := s.Store.ModelDirs.Get(); len(dirs) > 0 {
			e.SetModelDir(dirs[0])
		}
	}
	dir()
	on(s.Store.ModelDirs, dir)

	var srv *api.Server
	sync := func() {
		var msg string
		srv, msg = api.Toggle(srv, s.Store.API.Get(), s.Store.APIAddr.Get(), e)
		if msg != "" {
			s.SetStatus(msg)
		}
	}
	on(s.Store.API, sync)
	if s.Store.API.Get() {
		sync()
	}
	s.OnShutdown(func() {
		if srv != nil {
			srv.Close()
		}
	})
}
