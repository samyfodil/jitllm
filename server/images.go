package server

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"image"
	_ "image/gif" // a GIF's first frame, which image.Decode returns
	_ "image/jpeg"
	_ "image/png"
	"io"
	"net"
	"net/http"
	"net/netip"
	"strings"
	"syscall"
	"time"

	_ "golang.org/x/image/webp"
)

// Pictures in a chat request.
//
// Both HTTP shims accept pictures the way their APIs send them -- OpenAI's
// image_url parts, Anthropic's image blocks -- and turn each into a decoded
// image.Image, in the order the client sent them, carried on ChatInput.Images.
// The engine then lays them out where the model's own chat template puts its
// image markers (Engine.pictureSpans), so a picture is a span of the prompt and
// the prefix cache names it by its ImageKey as it does on every other path.
//
// Every number that sizes a decode comes from a header checked against
// ImagePolicy first: the encoded bytes before base64 is decoded or a body is
// read, and the pixel count from image.DecodeConfig before image.Decode
// allocates the raster.

// ImagePolicy bounds the pictures a request may carry. The zero value takes
// every default and fetches nothing remote.
type ImagePolicy struct {
	// FetchRemote lets a request name a picture by an http(s) URL, which the
	// server then fetches. It is off by default: a server that fetches what a
	// request names is a proxy into whatever network it sits on. Even on, an
	// address that resolves to a loopback, private, link-local, multicast or
	// unspecified IP is refused at connect time, and no proxy is used.
	FetchRemote bool
	// FetchTimeout bounds one fetch, the connection and the body together
	// (default 10 s).
	FetchTimeout time.Duration
	// MaxBytes bounds one picture's encoded bytes (default 20 MiB).
	MaxBytes int64
	// MaxPixels bounds one picture's width times height (default 4096x4096).
	MaxPixels int64
	// MaxImages bounds the pictures of one request (default 8).
	MaxImages int

	// allowPrivate lets a fetch reach a private address; only tests set it,
	// since an httptest server is on the loopback.
	allowPrivate bool
}

// The ImagePolicy defaults.
const (
	DefaultImageMaxBytes     = 20 << 20
	DefaultImageMaxPixels    = 4096 * 4096
	DefaultImageMaxImages    = 8
	DefaultImageFetchTimeout = 10 * time.Second
)

func (p ImagePolicy) withDefaults() ImagePolicy {
	if p.MaxBytes <= 0 {
		p.MaxBytes = DefaultImageMaxBytes
	}
	if p.MaxPixels <= 0 {
		p.MaxPixels = DefaultImageMaxPixels
	}
	if p.MaxImages <= 0 {
		p.MaxImages = DefaultImageMaxImages
	}
	if p.FetchTimeout <= 0 {
		p.FetchTimeout = DefaultImageFetchTimeout
	}
	return p
}

// imageFormats maps a decoder's format name to the media type it is sent as.
var imageFormats = map[string]string{
	"png":  "image/png",
	"jpeg": "image/jpeg",
	"gif":  "image/gif",
	"webp": "image/webp",
}

// imageMediaType is the media type for a declared one, "image/jpg" taken as
// the alias clients send; ok is false for a type that is not decoded here.
func imageMediaType(mt string) (string, bool) {
	mt = strings.ToLower(strings.TrimSpace(mt))
	if mt == "image/jpg" {
		mt = "image/jpeg"
	}
	for _, v := range imageFormats {
		if v == mt {
			return mt, true
		}
	}
	return "", false
}

// errImageFetchOff is a remote picture on a server that does not fetch.
var errImageFetchOff = errors.New("names a picture by a remote URL, and this server does not fetch " +
	"remote images: send it inline (base64), or start the server with remote image fetching " +
	"enabled (jitllmd -image-fetch, server.Config.Images.FetchRemote)")

// decodeImage decodes b, a picture whose declared media type is mt ("" for
// none), within p's bounds.
func (p ImagePolicy) decodeImage(b []byte, mt string) (image.Image, error) {
	if int64(len(b)) > p.MaxBytes {
		return nil, fmt.Errorf("is %d bytes, over the limit of %d", len(b), p.MaxBytes)
	}
	cfg, format, err := image.DecodeConfig(bytes.NewReader(b))
	if err != nil {
		return nil, fmt.Errorf("is not a PNG, JPEG, WebP or GIF picture: %v", err)
	}
	got, ok := imageFormats[format]
	if !ok {
		return nil, fmt.Errorf("is %s; the pictures accepted are PNG, JPEG, WebP and GIF", format)
	}
	if mt != "" && mt != got {
		return nil, fmt.Errorf("is declared %s and its bytes are %s", mt, got)
	}
	if cfg.Width <= 0 || cfg.Height <= 0 {
		return nil, fmt.Errorf("is %dx%d, an empty picture", cfg.Width, cfg.Height)
	}
	if px := int64(cfg.Width) * int64(cfg.Height); px > p.MaxPixels {
		return nil, fmt.Errorf("is %dx%d, %d pixels, over the limit of %d", cfg.Width, cfg.Height, px, p.MaxPixels)
	}
	img, _, err := image.Decode(bytes.NewReader(b))
	if err != nil {
		return nil, fmt.Errorf("does not decode: %v", err)
	}
	return img, nil
}

// decodeBase64Image decodes a base64 payload declared as mt, sizing nothing
// from the payload before its length is checked.
func (p ImagePolicy) decodeBase64Image(data, mt string) (image.Image, error) {
	data = strings.TrimSpace(data)
	if int64(base64.StdEncoding.DecodedLen(len(data))) > p.MaxBytes+2 {
		return nil, fmt.Errorf("is about %d bytes, over the limit of %d",
			base64.StdEncoding.DecodedLen(len(data)), p.MaxBytes)
	}
	b, err := base64.StdEncoding.DecodeString(data)
	if err != nil {
		// Some clients send the unpadded or URL-safe alphabet.
		if b, err = base64.RawStdEncoding.DecodeString(strings.TrimRight(data, "=")); err != nil {
			if b, err = base64.RawURLEncoding.DecodeString(strings.TrimRight(data, "=")); err != nil {
				return nil, errors.New("is not valid base64")
			}
		}
	}
	return p.decodeImage(b, mt)
}

// decodeDataURL decodes a data: URL, data:<media type>[;params];base64,<data>.
func (p ImagePolicy) decodeDataURL(u string) (image.Image, error) {
	rest, ok := cutPrefixFold(u, "data:")
	if !ok {
		return nil, errors.New("is not a data: URL")
	}
	head, data, ok := strings.Cut(rest, ",")
	if !ok {
		return nil, errors.New("is a data: URL with no comma before its data")
	}
	params := strings.Split(head, ";")
	mt, ok := imageMediaType(params[0])
	if !ok {
		return nil, fmt.Errorf("is a data: URL of type %q; the pictures accepted are image/png, "+
			"image/jpeg, image/webp and image/gif", params[0])
	}
	b64 := false
	for _, x := range params[1:] {
		if strings.EqualFold(strings.TrimSpace(x), "base64") {
			b64 = true
		}
	}
	if !b64 {
		return nil, errors.New("is a data: URL that is not base64; send ;base64,")
	}
	return p.decodeBase64Image(data, mt)
}

func cutPrefixFold(s, prefix string) (string, bool) {
	if len(s) >= len(prefix) && strings.EqualFold(s[:len(prefix)], prefix) {
		return s[len(prefix):], true
	}
	return s, false
}

// imageURL decodes the picture a URL names: a data: URL inline, an http(s)
// one fetched when p allows it.
func (p ImagePolicy) imageURL(ctx context.Context, u string) (image.Image, error) {
	u = strings.TrimSpace(u)
	if _, ok := cutPrefixFold(u, "data:"); ok {
		return p.decodeDataURL(u)
	}
	_, http1 := cutPrefixFold(u, "http://")
	_, https := cutPrefixFold(u, "https://")
	if !http1 && !https {
		return nil, errors.New("is neither a data: URL nor an http(s) URL")
	}
	if !p.FetchRemote {
		return nil, errImageFetchOff
	}
	b, err := p.fetch(ctx, u)
	if err != nil {
		return nil, err
	}
	return p.decodeImage(b, "")
}

// fetch reads u's body, at most MaxBytes of it, within FetchTimeout.
func (p ImagePolicy) fetch(ctx context.Context, u string) ([]byte, error) {
	ctx, cancel := context.WithTimeout(ctx, p.FetchTimeout)
	defer cancel()
	d := &net.Dialer{Timeout: p.FetchTimeout}
	if !p.allowPrivate {
		// The address is checked as dialled, after resolution, so a name that
		// resolves to a public address for the check and a private one for
		// the connection has no window to do it in.
		d.Control = refusePrivate
	}
	c := &http.Client{
		Transport: &http.Transport{Proxy: nil, DialContext: d.DialContext,
			TLSHandshakeTimeout: p.FetchTimeout, ResponseHeaderTimeout: p.FetchTimeout},
		CheckRedirect: func(r *http.Request, via []*http.Request) error {
			if len(via) >= 3 {
				return errors.New("more than 3 redirects")
			}
			if r.URL.Scheme != "http" && r.URL.Scheme != "https" {
				return fmt.Errorf("a redirect to a %s URL", r.URL.Scheme)
			}
			return nil
		},
	}
	defer c.CloseIdleConnections()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, fmt.Errorf("is not a URL: %v", err)
	}
	resp, err := c.Do(req)
	if err != nil {
		return nil, fmt.Errorf("could not be fetched: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("could not be fetched: %s", resp.Status)
	}
	if resp.ContentLength > p.MaxBytes {
		return nil, fmt.Errorf("is %d bytes, over the limit of %d", resp.ContentLength, p.MaxBytes)
	}
	b, err := io.ReadAll(io.LimitReader(resp.Body, p.MaxBytes+1))
	if err != nil {
		return nil, fmt.Errorf("could not be fetched: %v", err)
	}
	if int64(len(b)) > p.MaxBytes {
		return nil, fmt.Errorf("is over the limit of %d bytes", p.MaxBytes)
	}
	return b, nil
}

// cgnat is the shared address space (RFC 6598), private in all but name.
var cgnat = netip.PrefixFrom(netip.AddrFrom4([4]byte{100, 64, 0, 0}), 10)

// refusePrivate is a dialer's Control: it refuses an address that is not a
// public unicast one.
func refusePrivate(network, address string, _ syscall.RawConn) error {
	ap, err := netip.ParseAddrPort(address)
	if err != nil {
		return fmt.Errorf("refusing %s: %v", address, err)
	}
	if !publicAddr(ap.Addr()) {
		return fmt.Errorf("refusing %s: a remote image may not be fetched from a private, loopback "+
			"or link-local address", ap.Addr())
	}
	return nil
}

func publicAddr(a netip.Addr) bool {
	a = a.Unmap()
	return a.IsValid() && a.IsGlobalUnicast() && !a.IsPrivate() && !cgnat.Contains(a)
}

// imageError names the part of a request a picture came from.
type imageError struct {
	part string
	err  error
}

func (e *imageError) Error() string { return e.part + " " + e.err.Error() }
func (e *imageError) Unwrap() error { return e.err }

// pictures collects a request's pictures in order, against p's bounds.
type pictures struct {
	p    ImagePolicy
	ctx  context.Context
	imgs []image.Image
}

// add decodes one picture with load, naming part on failure.
func (ps *pictures) add(part string, load func() (image.Image, error)) error {
	if len(ps.imgs) >= ps.p.MaxImages {
		return &imageError{part, fmt.Errorf("is picture %d; a request carries at most %d",
			len(ps.imgs)+1, ps.p.MaxImages)}
	}
	img, err := load()
	if err != nil {
		return &imageError{part, err}
	}
	ps.imgs = append(ps.imgs, img)
	return nil
}
