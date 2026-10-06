package widgets

import (
	"fmt"
	"image/color"
	"os"
	"regexp"
	"strconv"
	"testing"
)

// The window icon is the website's favicon, pixel for pixel: testdata holds
// the website's favicon.svg, a 23x23 grid of one-pixel rects.
func TestLogoIconIsTheFavicon(t *testing.T) {
	svg, err := os.ReadFile("testdata/favicon.svg")
	if err != nil {
		t.Fatal(err)
	}
	want := map[[2]int]string{}
	re := regexp.MustCompile(`<rect x="(\d+)" y="(\d+)" width="1" height="1" fill="#([0-9a-f]{6})"/>`)
	for _, m := range re.FindAllStringSubmatch(string(svg), -1) {
		x, _ := strconv.Atoi(m[1])
		y, _ := strconv.Atoi(m[2])
		want[[2]int{x, y}] = m[3]
	}
	if len(want) < 50 {
		t.Fatalf("read %d pixels out of the favicon: the parse is broken, not the icon", len(want))
	}
	hex := func(h uint32) color.RGBA { return color.RGBA{uint8(h >> 16), uint8(h >> 8), uint8(h), 0xFF} }
	bands := [5]color.Color{hex(0xDAECC6), hex(0xBBDD97), hex(0x9ECE6A), hex(0x678549), hex(0x39482E)}
	img := LogoIcon(23, hex(0x0E0E14), bands)
	for y := range 23 {
		for x := range 23 {
			r, g, b, _ := img.At(x, y).RGBA()
			got := fmt.Sprintf("%02x%02x%02x", r>>8, g>>8, b>>8)
			w, ok := want[[2]int{x, y}]
			if !ok {
				w = "0e0e14"
			}
			if got != w {
				t.Errorf("pixel (%d,%d) is #%s, the favicon has #%s", x, y, got, w)
			}
		}
	}
}
