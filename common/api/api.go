// Package api serves jitllm's API (server.Engine.Handler: ConnectRPC, the
// OpenAI and Anthropic shims) from inside a front end, so the window and the
// terminal can turn it on without running jitllmd beside them.
//
// It serves the front end's own engine (common/engine.Engine.Server): a model
// open in the app is a model of the API, and a request through the API is one
// more session on it, under the same budget and gates.
package api

import (
	"context"
	"errors"
	"net"
	"net/http"
	"time"

	"github.com/jitllm/jitllm/server"
)

// DefaultAddr is loopback only: turning the API on must not open it to the
// network unless the address says so.
const DefaultAddr = "127.0.0.1:8080"

// Server is a running API.
type Server struct {
	srv  *http.Server
	addr string
	// asked is the address as the setting spells it, so Toggle can tell a
	// changed setting from the one already served.
	asked string
}

// Start listens on addr and serves e. The listen happens before it returns,
// so an address in use is its error.
func Start(addr string, e *server.Engine) (*Server, error) {
	if addr == "" {
		addr = DefaultAddr
	}
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, err
	}
	s := &Server{addr: ln.Addr().String(), asked: addr, srv: &http.Server{
		Handler: e.Handler(),
		// No WriteTimeout, as jitllmd: a streaming generate can run for minutes.
		ReadHeaderTimeout: 20 * time.Second,
	}}
	go func() {
		if err := s.srv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
			ln.Close()
		}
	}()
	return s, nil
}

// Addr is the address it listens on.
func (s *Server) Addr() string { return s.addr }

// Close stops listening and waits for the requests in flight. The engine and
// its models stay: they are the front end's.
func (s *Server) Close() {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	s.srv.Shutdown(ctx)
}

// Toggle brings cur in line with a setting: off closes it, on starts one, and
// on with a different address restarts it. It returns what now runs (nil when
// off) and a line for the front end's status.
func Toggle(cur *Server, on bool, addr string, e *server.Engine) (*Server, string) {
	if addr == "" {
		addr = DefaultAddr
	}
	if cur != nil && on && cur.asked == addr {
		return cur, "API on http://" + cur.Addr()
	}
	if cur != nil {
		cur.Close()
	}
	if !on {
		if cur == nil {
			return nil, ""
		}
		return nil, "API off"
	}
	s, err := Start(addr, e)
	if err != nil {
		return nil, "API not started: " + err.Error()
	}
	return s, "API on http://" + s.Addr()
}
