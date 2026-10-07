package main

import (
	"bytes"
	"encoding/binary"
	"image/png"

	"github.com/samyfodil/jitllm/ui/app"
)

// iconPNG is the app icon at size pixels square, PNG-encoded. Both formats
// carry PNG: Windows has read PNG icon images since Vista, and every .icns
// type used below is PNG.
func iconPNG(size int) ([]byte, error) {
	var b bytes.Buffer
	if err := png.Encode(&b, app.IconAt(size)); err != nil {
		return nil, err
	}
	return b.Bytes(), nil
}

// icnsTypes are the .icns element types for PNG images, with their pixel
// size: the 1x and 2x pairs of the 16, 32, 128, 256 and 512 point icons.
var icnsTypes = []struct {
	tag  string
	size int
}{
	{"icp4", 16}, {"ic11", 32}, {"icp5", 32}, {"ic12", 64},
	{"ic07", 128}, {"ic13", 256}, {"ic08", 256}, {"ic14", 512},
	{"ic09", 512}, {"ic10", 1024},
}

// icns is the macOS icon file: "icns", the file's length, then one element
// per image, each its type, its length with the eight header bytes, and the
// PNG. Written here so a Linux runner can build the bundle without iconutil.
func icns() ([]byte, error) {
	var body bytes.Buffer
	cache := map[int][]byte{}
	for _, t := range icnsTypes {
		p, ok := cache[t.size]
		if !ok {
			var err error
			if p, err = iconPNG(t.size); err != nil {
				return nil, err
			}
			cache[t.size] = p
		}
		body.WriteString(t.tag)
		binary.Write(&body, binary.BigEndian, uint32(8+len(p)))
		body.Write(p)
	}
	var out bytes.Buffer
	out.WriteString("icns")
	binary.Write(&out, binary.BigEndian, uint32(8+body.Len()))
	out.Write(body.Bytes())
	return out.Bytes(), nil
}
