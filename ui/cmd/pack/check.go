package main

import (
	"bytes"
	"debug/pe"
	"encoding/binary"
	"fmt"
)

// imageSubsystemWindowsGUI is the optional header's Subsystem for a program
// Windows starts without a console (IMAGE_SUBSYSTEM_WINDOWS_GUI).
const imageSubsystemWindowsGUI = 2

// checkExe refuses a Windows executable that would open a console window when
// started, or that carries no icon or version information.
func checkExe(path string) error {
	f, err := pe.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	var sub uint16
	switch h := f.OptionalHeader.(type) {
	case *pe.OptionalHeader64:
		sub = h.Subsystem
	case *pe.OptionalHeader32:
		sub = h.Subsystem
	default:
		return fmt.Errorf("%s has no optional header", path)
	}
	if sub != imageSubsystemWindowsGUI {
		return fmt.Errorf("%s: subsystem %d, want %d (GUI): link it with -ldflags -H=windowsgui", path, sub, imageSubsystemWindowsGUI)
	}
	types, err := resourceTypes(f)
	if err != nil {
		return fmt.Errorf("%s: %v", path, err)
	}
	for _, want := range []uint32{rtIcon, rtGroupIcon, rtVersion} {
		if !types[want] {
			return fmt.Errorf("%s: no resource of type %d: build it beside the .syso from `pack syso`", path, want)
		}
	}
	// The tree can be whole with its data entries pointing at the wrong
	// bytes, if the linker did not relocate them: read the version back.
	if err := versionReadsBack(f); err != nil {
		return fmt.Errorf("%s: %v", path, err)
	}
	fmt.Printf("%s: subsystem %d (GUI), resources %v\n", path, sub, keys(types))
	return nil
}

// versionReadsBack follows the version resource's directory to its data entry
// and checks the RVA there lands on a VS_VERSION_INFO.
func versionReadsBack(f *pe.File) error {
	s := f.Section(".rsrc")
	b, err := s.Data()
	if err != nil {
		return err
	}
	le := binary.LittleEndian
	// child finds id in the directory at off, or the first entry for id 0.
	child := func(off uint32, id uint32) (uint32, error) {
		if int(off)+16 > len(b) {
			return 0, fmt.Errorf("resource directory at %#x runs past .rsrc", off)
		}
		n := int(le.Uint16(b[off+12:])) + int(le.Uint16(b[off+14:]))
		for i := range n {
			at := int(off) + 16 + 8*i
			if at+8 > len(b) {
				break
			}
			if id == 0 || le.Uint32(b[at:]) == id {
				return le.Uint32(b[at+4:]) &^ 0x80000000, nil
			}
		}
		return 0, fmt.Errorf("resource %d not found", id)
	}
	off := uint32(0)
	for _, id := range []uint32{rtVersion, 0, 0} {
		if off, err = child(off, id); err != nil {
			return err
		}
	}
	if int(off)+8 > len(b) {
		return fmt.Errorf("version data entry runs past .rsrc")
	}
	at := int64(le.Uint32(b[off:])) - int64(s.VirtualAddress)
	key := wstr("VS_VERSION_INFO")
	if at < 0 || int(at)+6+len(key) > len(b) || !bytes.Equal(b[at+6:int(at)+6+len(key)], key) {
		return fmt.Errorf("the version resource's RVA %#x does not land on VS_VERSION_INFO", le.Uint32(b[off:]))
	}
	return nil
}

// resourceTypes reads the type IDs at the root of the .rsrc section.
func resourceTypes(f *pe.File) (map[uint32]bool, error) {
	s := f.Section(".rsrc")
	if s == nil {
		return nil, fmt.Errorf("no .rsrc section")
	}
	b, err := s.Data()
	if err != nil {
		return nil, err
	}
	if len(b) < 16 {
		return nil, fmt.Errorf(".rsrc is %d bytes", len(b))
	}
	le := binary.LittleEndian
	named, ids := int(le.Uint16(b[12:])), int(le.Uint16(b[14:]))
	out := map[uint32]bool{}
	for i := named; i < named+ids; i++ {
		at := 16 + 8*i
		if at+8 > len(b) {
			return nil, fmt.Errorf(".rsrc root directory runs past the section")
		}
		out[le.Uint32(b[at:])] = true
	}
	return out, nil
}

func keys(m map[uint32]bool) []uint32 {
	var out []uint32
	for _, k := range []uint32{rtIcon, rtGroupIcon, rtVersion} {
		if m[k] {
			out = append(out, k)
		}
	}
	return out
}
