package jlm

import (
	"encoding/binary"
	"fmt"
)

// The tensor table. Lengths are derived, not stored, so the table cannot
// disagree with the packer about a span's size: a packed tensor's three spans
// come from the writer's arithmetic, a verbatim one's from its dims and type.
//
//	u16 role | i32 block | i32 index | u8 type | u8 ndim | u64 dims[ndim]
//	| u64 qsOff | u64 dOff | u64 scOff | u16 nameLen | name
//
// The name is a label for dumps; (role, block, index) is the identity.

func encodeTable(es []Entry) []byte {
	var b []byte
	var n8 [8]byte
	put := func(v uint64) { binary.LittleEndian.PutUint64(n8[:], v); b = append(b, n8[:]...) }
	for _, e := range es {
		b = binary.LittleEndian.AppendUint16(b, uint16(e.Role))
		b = binary.LittleEndian.AppendUint32(b, uint32(e.Block))
		b = binary.LittleEndian.AppendUint32(b, uint32(e.Index))
		b = append(b, byte(e.Type), e.NDim)
		for i := 0; i < int(e.NDim); i++ {
			put(e.Dims[i])
		}
		put(e.QSOff)
		put(e.DOff)
		put(e.SCOff)
		b = binary.LittleEndian.AppendUint16(b, uint16(len(e.Name)))
		b = append(b, e.Name...)
	}
	return b
}

func decodeTable(b []byte, n uint32, size uint64) ([]Entry, error) {
	es := make([]Entry, 0, n)
	le := binary.LittleEndian
	for len(b) > 0 {
		const head = 2 + 4 + 4 + 1 + 1
		if len(b) < head {
			return nil, fmt.Errorf("jlm: table: %d trailing bytes", len(b))
		}
		e := Entry{
			Role:  Role(le.Uint16(b)),
			Block: int32(le.Uint32(b[2:])),
			Index: int32(le.Uint32(b[6:])),
			Type:  Type(b[10]),
			NDim:  b[11],
		}
		b = b[head:]
		// ndim is from the file, so it is bounded before it indexes anything
		// (RULE 9).
		if e.NDim == 0 || int(e.NDim) > len(e.Dims) {
			return nil, fmt.Errorf("jlm: %v: %d dimensions, want 1..%d", e.Role, e.NDim, len(e.Dims))
		}
		if !e.Role.Valid() {
			return nil, fmt.Errorf("jlm: role code %d is not one this format defines", e.Role)
		}
		if !e.Type.Valid() {
			return nil, fmt.Errorf("jlm: %v: type code %d is not one this format defines", e.Role, e.Type)
		}
		want := int(e.NDim)*8 + 24 + 2
		if len(b) < want {
			return nil, fmt.Errorf("jlm: %v wants %d bytes, %d left", e.Role, want, len(b))
		}
		for i := 0; i < int(e.NDim); i++ {
			e.Dims[i] = le.Uint64(b[i*8:])
		}
		b = b[int(e.NDim)*8:]
		e.QSOff, e.DOff, e.SCOff = le.Uint64(b), le.Uint64(b[8:]), le.Uint64(b[16:])
		b = b[24:]
		nl := int(le.Uint16(b))
		b = b[2:]
		if len(b) < nl {
			return nil, fmt.Errorf("jlm: %v: name wants %d bytes, %d left", e.Role, nl, len(b))
		}
		e.Name = string(b[:nl])
		b = b[nl:]
		for _, o := range []uint64{e.QSOff, e.DOff, e.SCOff} {
			if o > size {
				return nil, fmt.Errorf("jlm: %v: span at %d is past the %d-byte file", e.Role, o, size)
			}
		}
		es = append(es, e)
	}
	if uint32(len(es)) != n {
		return nil, fmt.Errorf("jlm: table holds %d entries, the header says %d", len(es), n)
	}
	return es, nil
}

func encodeFP(f Fingerprint) []byte {
	var b []byte
	for _, s := range []string{f.Host, f.Device, f.Writer} {
		b = binary.LittleEndian.AppendUint16(b, uint16(len(s)))
		b = append(b, s...)
	}
	return binary.LittleEndian.AppendUint64(b, f.VRAM)
}

func decodeFP(b []byte) (Fingerprint, error) {
	var f Fingerprint
	le := binary.LittleEndian
	for _, p := range []*string{&f.Host, &f.Device, &f.Writer} {
		if len(b) < 2 {
			return f, fmt.Errorf("jlm: fingerprint truncated")
		}
		n := int(le.Uint16(b))
		b = b[2:]
		if len(b) < n {
			return f, fmt.Errorf("jlm: fingerprint field wants %d bytes, %d left", n, len(b))
		}
		*p = string(b[:n])
		b = b[n:]
	}
	if len(b) < 8 {
		return f, fmt.Errorf("jlm: fingerprint missing its vram field")
	}
	f.VRAM = le.Uint64(b)
	return f, nil
}
