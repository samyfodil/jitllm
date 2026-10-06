// Package safetensors is a parser for the safetensors container, and nothing
// else. It produces a tensor directory -- name, dtype, shape, byte range -- and
// reads a tensor's bytes on demand.
//
// Like gguf/, it is a converter input that knows nothing about the engine;
// nothing here imports jlm, model or nn. A safetensors file carries no
// hyperparameters or tokenizer (those live in config.json and tokenizer.json
// beside it), so convert/ takes the directory.
//
// The layout is: a little-endian u64 header length N, N bytes of JSON mapping
// tensor name to {dtype, shape, data_offsets: [begin, end)}, then the raw
// tensor bytes. Offsets are relative to the first byte after the header.
package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
)

// MaxHeader bounds the JSON header. It is a sanity bound; the real check is
// that the header fits inside the file, which is measured rather than asserted
// by the file. Real headers reach a few MB at most.
const MaxHeader = 256 << 20

// DType is a safetensors element type, spelled exactly as the format spells it.
//
// It is a string, as the format uses, so an unsupported dtype can be refused
// by name in the converter.
type DType string

const (
	F64    DType = "F64"
	F32    DType = "F32"
	F16    DType = "F16"
	BF16   DType = "BF16"
	I64    DType = "I64"
	I32    DType = "I32"
	I16    DType = "I16"
	I8     DType = "I8"
	U8     DType = "U8"
	BOOL   DType = "BOOL"
	F8E4M3 DType = "F8_E4M3"
	F8E5M2 DType = "F8_E5M2"
)

// Size is the byte width of one element, and ok=false for a dtype this parser
// does not know. An unknown dtype is not assumed to be one byte: a wrong width
// turns into a wrong bounds check, which is the one thing a parser of untrusted
// input must not get wrong.
func (d DType) Size() (int, bool) {
	switch d {
	case F64, I64:
		return 8, true
	case F32, I32:
		return 4, true
	case F16, BF16, I16:
		return 2, true
	case I8, U8, BOOL, F8E4M3, F8E5M2:
		return 1, true
	}
	return 0, false
}

// Tensor is one entry of the directory.
type Tensor struct {
	Name  string
	DType DType
	// Shape is in the format's own order: slowest axis first, C-contiguous, so
	// a [out, in] Linear weight is stored row by row of `in` elements. That is
	// the reverse of the ne0-fastest convention the container uses; reversing
	// it is the converter's business, not this package's.
	Shape []uint64
	// Begin and End are relative to the first byte after the header.
	Begin, End uint64
}

// Elems is the logical element count.
func (t *Tensor) Elems() uint64 {
	n := uint64(1)
	for _, d := range t.Shape {
		n *= d
	}
	return n
}

// NBytes is End-Begin, which Parse has already proved equals Elems*width.
func (t *Tensor) NBytes() uint64 { return t.End - t.Begin }

// File is an open safetensors file.
//
// It reads, it does not map, for the reasons format/jlm/read.go measures; it also
// keeps this package free of per-GOOS files.
type File struct {
	Path string
	// Meta is the "__metadata__" entry, a free-form string map. Every writer
	// puts {"format":"pt"} in it and most put nothing else; it is not where
	// hyperparameters live.
	Meta map[string]string
	// Tensors is the directory, sorted by Begin so a caller reading them in
	// order reads the file in order.
	Tensors []Tensor

	f         io.ReaderAt
	c         io.Closer // what Open opened; nil for OpenAt
	dataStart uint64
	dataLen   uint64
	index     map[string]int
}

// Open parses path's header and leaves the file open for ReadTensor.
func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	st, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	sf, err := OpenAt(path, f, st.Size())
	if err != nil {
		f.Close()
		return nil, err
	}
	sf.c = f
	return sf, nil
}

// OpenAt parses a safetensors file of size bytes read through r, which need
// not be on the disk: a range reader over the Hub is the case it exists for,
// and a local file is the same bytes through os.File. name is for messages.
// OpenAt does not close r.
func OpenAt(name string, r io.ReaderAt, size int64) (*File, error) {
	if size < 8 {
		return nil, fmt.Errorf("safetensors: %s is %d bytes, too short for a header length", name, size)
	}
	var lb [8]byte
	if _, err := r.ReadAt(lb[:], 0); err != nil {
		return nil, fmt.Errorf("safetensors: %s: reading the header length: %w", name, err)
	}
	n := binary.LittleEndian.Uint64(lb[:])
	// The bound comes from the file, not the file's claim: n is untrusted,
	// uint64(size)-8 is measured.
	if n > uint64(size)-8 {
		return nil, fmt.Errorf("safetensors: %s claims a %d-byte header in a %d-byte file", name, n, size)
	}
	if n > MaxHeader {
		return nil, fmt.Errorf("safetensors: %s has a %d-byte header, over the %d-byte bound", name, n, MaxHeader)
	}
	hdr := make([]byte, n)
	if _, err := r.ReadAt(hdr, 8); err != nil {
		return nil, fmt.Errorf("safetensors: %s: reading the header: %w", name, err)
	}
	dataLen := uint64(size) - 8 - n
	meta, tensors, err := Parse(hdr, dataLen)
	if err != nil {
		return nil, fmt.Errorf("safetensors: %s: %w", name, err)
	}
	sf := &File{Path: name, Meta: meta, Tensors: tensors, f: r,
		dataStart: 8 + n, dataLen: dataLen}
	sf.index = make(map[string]int, len(tensors))
	for i := range tensors {
		sf.index[tensors[i].Name] = i
	}
	return sf, nil
}

// Close releases the file handle. Buffers handed out by ReadTensor are copies
// and stay valid.
func (f *File) Close() error {
	if f.c == nil {
		return nil
	}
	return f.c.Close()
}

// Get looks a tensor up by its exact name.
func (f *File) Get(name string) (*Tensor, bool) {
	i, ok := f.index[name]
	if !ok {
		return nil, false
	}
	return &f.Tensors[i], true
}

// ReadTensor returns a fresh buffer holding t's bytes.
//
// It re-checks the bound rather than inheriting Parse's proof: convert/gguf/validate.go
// holds the same standard, and it is what keeps this safe if a caller ever
// hands in a Tensor it built itself.
func (f *File) ReadTensor(t *Tensor) ([]byte, error) {
	if t.End < t.Begin || t.End > f.dataLen {
		return nil, fmt.Errorf("safetensors: %s: %q wants bytes [%d,%d) of %d",
			f.Path, t.Name, t.Begin, t.End, f.dataLen)
	}
	b := make([]byte, t.End-t.Begin)
	if len(b) == 0 {
		return b, nil
	}
	if _, err := f.f.ReadAt(b, int64(f.dataStart+t.Begin)); err != nil {
		return nil, fmt.Errorf("safetensors: %s: reading %q: %w", f.Path, t.Name, err)
	}
	return b, nil
}

// Headers is f with its data replaced by zeros: the same directory, and a
// ReadTensor that answers a zeroed buffer of the tensor's length without
// reading. A converter run over it decides everything the headers decide --
// every name, shape and type, the container's layout -- and transfers
// nothing past them, which is how a conversion is planned before it is run.
func (f *File) Headers() *File {
	g := *f
	g.f, g.c = zeroData{}, nil
	return &g
}

// zeroData reads as zeros everywhere.
type zeroData struct{}

func (zeroData) ReadAt(p []byte, off int64) (int, error) {
	clear(p)
	return len(p), nil
}

// hdrEntry is one JSON value of the header map.
type hdrEntry struct {
	DType   DType    `json:"dtype"`
	Shape   []uint64 `json:"shape"`
	Offsets []uint64 `json:"data_offsets"`
}

// Parse reads a header and validates every entry against dataLen, the number of
// bytes that actually follow it.
//
// It is exported for the fuzzer.
//
// Every number in here is untrusted: a shape is multiplied out with an
// overflow check, the declared byte range must equal the element count times
// the dtype width (not merely contain it), and no two tensors may overlap,
// which would alias two weights onto one range.
func Parse(hdr []byte, dataLen uint64) (map[string]string, []Tensor, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(hdr, &raw); err != nil {
		return nil, nil, fmt.Errorf("the header is not a JSON object: %w", err)
	}
	var meta map[string]string
	if m, ok := raw["__metadata__"]; ok {
		if err := json.Unmarshal(m, &meta); err != nil {
			return nil, nil, fmt.Errorf("__metadata__ is not a string map: %w", err)
		}
		delete(raw, "__metadata__")
	}
	if len(raw) == 0 {
		return nil, nil, fmt.Errorf("the header names no tensors")
	}
	// len(raw) is derived from bytes we have, so sizing by it is bounded by the
	// header already read rather than by a count the header asserts.
	out := make([]Tensor, 0, len(raw))
	for name, rm := range raw {
		var e hdrEntry
		if err := json.Unmarshal(rm, &e); err != nil {
			return nil, nil, fmt.Errorf("%q: %w", name, err)
		}
		w, ok := e.DType.Size()
		if !ok {
			return nil, nil, fmt.Errorf("%q has dtype %q, which this parser does not define", name, e.DType)
		}
		if len(e.Offsets) != 2 {
			return nil, nil, fmt.Errorf("%q has %d data_offsets, want 2", name, len(e.Offsets))
		}
		begin, end := e.Offsets[0], e.Offsets[1]
		if begin > end {
			return nil, nil, fmt.Errorf("%q spans [%d,%d), which runs backwards", name, begin, end)
		}
		if end > dataLen {
			return nil, nil, fmt.Errorf("%q ends at %d, past the %d bytes of data", name, end, dataLen)
		}
		// A scalar has shape []; every real weight has at least one axis. The
		// product is built with an explicit overflow check because a crafted
		// shape of [2^40, 2^40] otherwise wraps to something small and passes
		// the length test below.
		n := uint64(1)
		for _, d := range e.Shape {
			if d != 0 && n > ^uint64(0)/d {
				return nil, nil, fmt.Errorf("%q has shape %v, whose product overflows", name, e.Shape)
			}
			n *= d
		}
		if n != 0 && uint64(w) > ^uint64(0)/n {
			return nil, nil, fmt.Errorf("%q has %d elements of %d bytes, which overflows", name, n, w)
		}
		if want := n * uint64(w); end-begin != want {
			return nil, nil, fmt.Errorf("%q spans %d bytes and its %s%v wants %d",
				name, end-begin, e.DType, e.Shape, want)
		}
		out = append(out, Tensor{Name: name, DType: e.DType, Shape: e.Shape, Begin: begin, End: end})
	}
	// Sorted by Begin, which makes the overlap check one pass and makes a
	// caller that walks the directory read the file forwards.
	sort.Slice(out, func(i, j int) bool {
		if out[i].Begin != out[j].Begin {
			return out[i].Begin < out[j].Begin
		}
		return out[i].Name < out[j].Name
	})
	for i := 1; i < len(out); i++ {
		// Empty tensors share an offset legitimately; a non-empty overlap does
		// not, and it would make two names resolve to one weight.
		if out[i].Begin < out[i-1].End {
			return nil, nil, fmt.Errorf("%q [%d,%d) overlaps %q [%d,%d)",
				out[i].Name, out[i].Begin, out[i].End,
				out[i-1].Name, out[i-1].Begin, out[i-1].End)
		}
	}
	return meta, out, nil
}
