package api

import (
	"io"
	"net/http"
	"testing"

	"github.com/jitllm/jitllm/server"
)

func TestStartServesAndCloses(t *testing.T) {
	e := server.New(server.Config{ModelDir: t.TempDir()})
	defer e.Close()
	s, err := Start("127.0.0.1:0", e)
	if err != nil {
		t.Fatal(err)
	}
	r, err := http.Get("http://" + s.Addr() + "/healthz")
	if err != nil {
		t.Fatal(err)
	}
	b, _ := io.ReadAll(r.Body)
	r.Body.Close()
	if string(b) != "ok\n" {
		t.Fatalf("healthz: %q", b)
	}
	if _, err := Start(s.Addr(), e); err == nil {
		t.Fatal("a second Start on the same address succeeded")
	}
	s.Close()
	if _, err := http.Get("http://" + s.Addr() + "/healthz"); err == nil {
		t.Fatal("still serving after Close")
	}
}

func TestToggle(t *testing.T) {
	e := server.New(server.Config{ModelDir: t.TempDir()})
	defer e.Close()
	s, msg := Toggle(nil, true, "127.0.0.1:0", e)
	if s == nil {
		t.Fatal(msg)
	}
	if again, _ := Toggle(s, true, "127.0.0.1:0", e); again != s {
		t.Fatal("an unchanged setting restarted the server")
	}
	if off, msg := Toggle(s, false, "", e); off != nil || msg != "API off" {
		t.Fatalf("off: %v %q", off, msg)
	}
}
