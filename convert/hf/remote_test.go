package hf

import (
	"bytes"
	"fmt"
	"math/rand/v2"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// TestRemoteReadsTheRangesItIsAskedFor serves a file with a Hub that fails the
// way a long transfer does -- a 500, and a response cut off mid-body -- and
// demands every range come back exact. A server that ignores Range must be a
// failure, not the file's first bytes.
func TestRemoteReadsTheRangesItIsAskedFor(t *testing.T) {
	data := make([]byte, 1<<20)
	rng := rand.New(rand.NewPCG(1, 2))
	for i := range data {
		data[i] = byte(rng.Uint32())
	}
	var n atomic.Int64
	ignoreRange := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/resolve/abc/m.safetensors") {
			http.NotFound(w, r)
			return
		}
		if ignoreRange && r.Method == "GET" {
			w.Write(data)
			return
		}
		switch n.Add(1) % 7 {
		case 6:
			if r.Header.Get("Range") != "" {
				// A body that starts and then stops producing bytes, with the
				// connection left open.
				var a, b int
				fmt.Sscanf(r.Header.Get("Range"), "bytes=%d-%d", &a, &b)
				w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", a, b, len(data)))
				w.Header().Set("Content-Length", fmt.Sprint(b-a+1))
				w.WriteHeader(http.StatusPartialContent)
				w.Write(data[a:min(a+10, b+1)])
				w.(http.Flusher).Flush()
				<-r.Context().Done()
				return
			}
		case 2:
			http.Error(w, "busy", http.StatusServiceUnavailable)
			return
		case 4:
			if r.Header.Get("Range") != "" {
				// A 206 whose body stops early, which is what a reset looks like.
				hj, _ := w.(http.Hijacker)
				c, buf, _ := hj.Hijack()
				buf.WriteString("HTTP/1.1 206 Partial Content\r\nContent-Length: 1000000\r\n\r\n")
				var a, b int
				fmt.Sscanf(r.Header.Get("Range"), "bytes=%d-%d", &a, &b)
				buf.Write(data[a:min(a+100, b+1)])
				buf.Flush()
				c.Close()
				return
			}
		}
		http.ServeContent(w, r, "m.safetensors", time.Time{}, bytes.NewReader(data))
	}))
	defer srv.Close()

	c := &Client{Endpoint: srv.URL}
	oldBackoff, oldStall, oldChunk := backoffBase, stallAfter, rangeChunk
	backoffBase, stallAfter, rangeChunk = time.Millisecond, 50*time.Millisecond, 16<<10
	defer func() { backoffBase, stallAfter, rangeChunk = oldBackoff, oldStall, oldChunk }()
	rm, err := c.OpenRemote(Ref{Repo: "o/r", Rev: "abc", File: "m.safetensors"})
	if err != nil {
		t.Fatal(err)
	}
	if rm.Size() != int64(len(data)) {
		t.Fatalf("size %d, want %d", rm.Size(), len(data))
	}
	for i := 0; i < 40; i++ {
		off := rng.IntN(len(data) - 1)
		l := 1 + rng.IntN(min(len(data)-off, 200000))
		p := make([]byte, l)
		if _, err := rm.ReadAt(p, int64(off)); err != nil {
			t.Fatalf("ReadAt(%d, %d): %v", off, l, err)
		}
		if !bytes.Equal(p, data[off:off+l]) {
			t.Fatalf("ReadAt(%d, %d): wrong bytes", off, l)
		}
	}
	if n.Load() < 50 {
		t.Fatalf("only %d requests, so the injected failures never fired", n.Load())
	}

	ignoreRange = true
	p := make([]byte, 10)
	if _, err := rm.ReadAt(p, 5000); err == nil {
		t.Fatal("a server that ignored Range was read as if it had honoured it")
	}
}
