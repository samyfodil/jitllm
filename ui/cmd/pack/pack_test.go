package main

import (
	"bytes"
	"debug/pe"
	"encoding/binary"
	"image/png"
	"testing"
)

// TestSysoIsAnObjectTheLinkerReads opens the object with debug/pe, as the
// linker's loader does: one .rsrc section, a relocation per resource, and the
// three resource types at its root.
func TestSysoIsAnObjectTheLinkerReads(t *testing.T) {
	for _, arch := range []string{"amd64", "arm64"} {
		b, err := syso(arch, "1.2.3-next")
		if err != nil {
			t.Fatal(err)
		}
		f, err := pe.NewFile(bytes.NewReader(b))
		if err != nil {
			t.Fatalf("%s: %v", arch, err)
		}
		s := f.Section(".rsrc")
		if s == nil {
			t.Fatalf("%s: no .rsrc section", arch)
		}
		if want := len(icoSizes) + 2; len(s.Relocs) != want {
			t.Errorf("%s: %d relocations, want %d", arch, len(s.Relocs), want)
		}
		types, err := resourceTypes(f)
		if err != nil {
			t.Fatal(err)
		}
		for _, want := range []uint32{rtIcon, rtGroupIcon, rtVersion} {
			if !types[want] {
				t.Errorf("%s: resource type %d missing: %v", arch, want, types)
			}
		}
	}
	if _, err := syso("386", "1.0.0"); err == nil {
		t.Error("an unsupported GOARCH was accepted")
	}
}

func TestQuad(t *testing.T) {
	for in, want := range map[string][4]uint16{
		"1.2.3": {1, 2, 3, 0}, "v0.9.1-rc1": {0, 9, 1, 0}, "1.2.3.4": {1, 2, 3, 4},
	} {
		got, err := quad(in)
		if err != nil || got != want {
			t.Errorf("quad(%q) = %v, %v; want %v", in, got, err, want)
		}
	}
	for _, bad := range []string{"1.2.3.4.5", "x.y", "70000.0.0"} {
		if _, err := quad(bad); err == nil {
			t.Errorf("quad(%q) accepted", bad)
		}
	}
}

// TestIcnsElementsAreTheirSize walks the .icns as macOS reads it: every
// element's length lands on the next, and each PNG is the size its type says.
func TestIcnsElementsAreTheirSize(t *testing.T) {
	b, err := icns()
	if err != nil {
		t.Fatal(err)
	}
	if string(b[:4]) != "icns" || int(binary.BigEndian.Uint32(b[4:])) != len(b) {
		t.Fatalf("header %q %d, file %d bytes", b[:4], binary.BigEndian.Uint32(b[4:]), len(b))
	}
	at, n := 8, 0
	for at < len(b) {
		tag, l := string(b[at:at+4]), int(binary.BigEndian.Uint32(b[at+4:]))
		img, err := png.DecodeConfig(bytes.NewReader(b[at+8 : at+l]))
		if err != nil {
			t.Fatalf("%s: %v", tag, err)
		}
		if want := icnsTypes[n].size; img.Width != want || img.Height != want || tag != icnsTypes[n].tag {
			t.Errorf("%s is %dx%d, want %s at %d", tag, img.Width, img.Height, icnsTypes[n].tag, want)
		}
		at += l
		n++
	}
	if at != len(b) || n != len(icnsTypes) {
		t.Errorf("walked %d of %d bytes, %d of %d elements", at, len(b), n, len(icnsTypes))
	}
}
