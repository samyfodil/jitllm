package hf

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

// A streaming converter reads the model where it lives: jlm's writer lays the
// container out from the tensors' shapes and then asks for their bytes one
// tensor at a time, so each safetensors shard is read by byte range from the
// Hub and never stored locally.

// repoInfo is what the Hub's API says about one revision of a repository.
type repoInfo struct {
	SHA      string `json:"sha"`
	Siblings []struct {
		Name string `json:"rfilename"`
	} `json:"siblings"`
}

func (c *Client) info(r Ref) (repoInfo, error) {
	var doc repoInfo
	req, err := c.req("GET", r.APIURL(c.Endpoint))
	if err != nil {
		return doc, err
	}
	resp, err := c.client().Do(req)
	if err != nil {
		return doc, fmt.Errorf("%s: %w", r, err)
	}
	defer resp.Body.Close()
	if err := httpErr(r, resp); err != nil {
		return doc, err
	}
	// The body is untrusted, so it is read under a bound.
	b, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return doc, fmt.Errorf("%s: %w", r, err)
	}
	if err := json.Unmarshal(b, &doc); err != nil {
		return doc, fmt.Errorf("%s: the Hub's file list did not parse: %w", r, err)
	}
	return doc, nil
}

var commitSHA = regexp.MustCompile(`^[0-9a-f]{40}$`)

// Pin resolves r's revision to the commit it names at the time of the call and
// lists that commit's files.
//
// A streamed conversion reads for hours, so "main" is not an address: a push
// mid-conversion would mix two checkpoints. Every range is taken from the
// immutable commit the header was read at.
func (c *Client) Pin(r Ref) (Ref, []string, error) {
	doc, err := c.info(r)
	if err != nil {
		return r, nil, err
	}
	if !commitSHA.MatchString(doc.SHA) {
		return r, nil, fmt.Errorf("%s: the Hub gave no commit for revision %q", r, r.Rev)
	}
	r.Rev = doc.SHA
	files := make([]string, len(doc.Siblings))
	for i, s := range doc.Siblings {
		files[i] = s.Name
	}
	return r, files, nil
}

// Remote is a file on the Hub read by byte range. It is an io.ReaderAt and
// safe for concurrent use.
type Remote struct {
	c    *Client
	r    Ref
	url  string
	size int64
	got  atomic.Int64
}

// OpenRemote stats r's file. r should be pinned (see Pin).
func (c *Client) OpenRemote(r Ref) (*Remote, error) {
	in, err := c.Stat(r)
	if err != nil {
		return nil, err
	}
	if in.Size < 0 {
		return nil, fmt.Errorf("%s: the Hub will not say how big it is, and a file read "+
			"by range has to know", r)
	}
	return &Remote{c: c, r: r, url: r.URL(c.Endpoint), size: in.Size}, nil
}

// Size is the file's length in bytes.
func (m *Remote) Size() int64 { return m.size }

// Transferred is how many bytes have arrived so far, over every ReadAt.
func (m *Remote) Transferred() int64 { return m.got.Load() }

// Name is the file's reference, for messages.
func (m *Remote) Name() string { return m.r.String() }

// ReadAt fills p from bytes [off, off+len(p)) of the file.
//
// It is many small ranges, rangeFanout at a time, each retried and watched on
// its own, so a stall costs one piece and the link is shared across
// connections.
func (m *Remote) ReadAt(p []byte, off int64) (int, error) {
	if off < 0 || off+int64(len(p)) > m.size {
		return 0, fmt.Errorf("%s: bytes [%d,%d) of %d", m.r, off, off+int64(len(p)), m.size)
	}
	var (
		wg    sync.WaitGroup
		sem   = make(chan struct{}, rangeFanout)
		mu    sync.Mutex
		first error
	)
	for at := 0; at < len(p); at += rangeChunk {
		piece := p[at:min(at+rangeChunk, len(p))]
		wg.Add(1)
		sem <- struct{}{}
		go func(piece []byte, off int64) {
			defer func() { <-sem; wg.Done() }()
			if err := m.piece(piece, off); err != nil {
				mu.Lock()
				if first == nil {
					first = err
				}
				mu.Unlock()
			}
		}(piece, off+int64(at))
	}
	wg.Wait()
	if first != nil {
		return 0, first
	}
	return len(p), nil
}

// rangeChunk and rangeFanout size the pieces a read is split into; the chunk
// is a variable so a gate can split a small file.
var rangeChunk = 8 << 20

const rangeFanout = 4

// piece reads one range, retrying.
//
// It keeps what a failed attempt delivered. Only a transport error, a stall, a
// 5xx or a 429 is retried; a deliberate answer (401, 404) is final.
func (m *Remote) piece(p []byte, off int64) error {
	done := 0
	for attempt := 0; done < len(p); attempt++ {
		n, err := m.once(p[done:], off+int64(done))
		done += n
		if err == nil {
			continue
		}
		var perm permanent
		if errors.As(err, &perm) || attempt >= maxAttempts {
			return fmt.Errorf("%s: bytes [%d,%d): %w", m.r, off, off+int64(len(p)), err)
		}
		if n > 0 {
			attempt = 0 // progress resets the budget; only a stuck range gives up
		}
		time.Sleep(backoff(attempt))
	}
	return nil
}

// maxAttempts is ten failures in a row on one range, about eight minutes.
const maxAttempts = 10

// backoffBase is a variable so a gate can inject failures without sleeping.
var backoffBase = time.Second

func backoff(attempt int) time.Duration {
	return min(backoffBase<<attempt, time.Minute)
}

type permanent struct{ error }

// stallAfter is how long a response body may produce nothing before the
// request is abandoned and retried. A variable so a gate need not wait it out.
var stallAfter = time.Minute

func (m *Remote) once(p []byte, off int64) (int, error) {
	req, err := m.c.req("GET", m.url)
	if err != nil {
		return 0, permanent{err}
	}
	// The transport times out a missing header, but a body that stops
	// mid-stream blocks Read forever. The watchdog cancels the request after
	// stallAfter with no bytes; each byte that arrives winds it back.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	dog := time.AfterFunc(stallAfter, cancel)
	defer dog.Stop()
	req = req.WithContext(ctx)
	req.Header.Set("Range", "bytes="+strconv.FormatInt(off, 10)+"-"+strconv.FormatInt(off+int64(len(p))-1, 10))
	resp, err := m.c.client().Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	switch {
	case resp.StatusCode == http.StatusPartialContent:
	case resp.StatusCode == http.StatusTooManyRequests, resp.StatusCode/100 == 5:
		return 0, fmt.Errorf("HTTP %d", resp.StatusCode)
	case resp.StatusCode == http.StatusOK:
		// A server that ignores Range sends the whole file from byte 0, and
		// reading p's worth of it would silently be the wrong bytes.
		return 0, permanent{fmt.Errorf("the server ignored the byte range")}
	default:
		if err := httpErr(m.r, resp); err != nil {
			return 0, permanent{err}
		}
		return 0, permanent{fmt.Errorf("HTTP %d to a byte range", resp.StatusCode)}
	}
	n, err := io.ReadFull(watched{resp.Body, dog}, p)
	m.got.Add(int64(n))
	if ctx.Err() != nil {
		err = fmt.Errorf("no bytes for %v after %d of %d", stallAfter, n, len(p))
	}
	if err == io.ErrUnexpectedEOF {
		err = fmt.Errorf("the range ended after %d of %d bytes", n, len(p))
	}
	return n, err
}

// watched winds a stall watchdog back on every read that delivers bytes.
type watched struct {
	r   io.Reader
	dog *time.Timer
}

func (w watched) Read(p []byte) (int, error) {
	n, err := w.r.Read(p)
	if n > 0 {
		w.dog.Reset(stallAfter)
	}
	return n, err
}
