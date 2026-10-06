package model

import (
	"fmt"
	"image"

	"github.com/samyfodil/jitllm/format/jlm"
)

// A picture in a prompt is rows like any other.
//
// A Span carries a Picture where it would carry tokens: the preprocessed
// pixels, named before a single block runs (ImageKey). Prefilling the span
// runs the State's vision segment over them -- the tower's blocks through the
// one runner, on the State's JIT and device -- unless the model's image cache
// already holds the picture's rows, and then the rows enter the text segment
// exactly as an embedding span's do. There is no encode stage for a caller to
// run first, and a prompt restored from the prefix cache past the picture
// runs no tower block at all: the picture's rows are named by the picture, so
// the pages that hold them are found without its rows being computed.

// Picture is one picture as the vision segment reads it.
type Picture struct {
	// Pixels is the preprocessed picture, [y][x][channel]: GH*PatchSz x
	// GW*PatchSz pixels, or a tiling tower's squares back to back.
	Pixels []float32
	// GH and GW are the patch grid one pass of the tower covers.
	GH, GW int
	// Rows is how many rows the segment's head emits for the picture: the
	// positions it takes in a prompt.
	Rows int
	// Grid is the rows' grid, where they turn on an M-RoPE text model.
	Grid ImageGrid
	// Key names the picture: its preprocessing, its grid and its pixels.
	Key ImageKey
}

// Picture preprocesses img for this State's vision segment and names it. The
// result goes into a prompt as Span.Picture (or Image.Picture), which encodes
// it when the span is prefilled.
func (s *State) Picture(img image.Image) (*Picture, error) {
	vs, err := s.Vision()
	if err != nil {
		return nil, err
	}
	px, err := vs.PreprocessImage(img)
	if err != nil {
		return nil, err
	}
	gh, gw := vs.patchGrid()
	return vs.vis.t.Cfg.picture(px, gh, gw)
}

// picture names the preprocessed pixels px of passes over a gh x gw grid.
func (c TowerConfig) picture(px []float32, gh, gw int) (*Picture, error) {
	one := gh * c.PatchSz * gw * c.PatchSz * 3
	if one == 0 || len(px)%one != 0 {
		return nil, fmt.Errorf("model: %d pixels is not a whole number of %dx%d patch grids", len(px), gh, gw)
	}
	passes := len(px) / one
	if passes > 1 && (c.Dynamic() || passes > c.MaxTiles+1) {
		return nil, fmt.Errorf("model: %d passes of a %dx%d grid; this tower takes %d", passes, gh, gw, c.MaxTiles+1)
	}
	sc := max(c.Scale, 1)
	grid := c.gridOf(gh/sc, gw/sc)
	per := c.Queries
	switch {
	case c.Kind == jlm.ProjPixtral:
		per = c.pixtralRows(gh, gw)
	case c.Kind == jlm.ProjHunyuanVL:
		// Its newline column and its begin and end rows.
		per = grid.Rows()
	case per == 0:
		per = gh * gw / (c.Scale * c.Scale)
	}
	return &Picture{Pixels: px, GH: gh, GW: gw, Rows: passes * per, Grid: grid,
		Key: c.imageKey(px, gh, gw)}, nil
}

// image is the picture's encoded rows as a chat Image: the merged rows, and a
// deepstack tower's taps behind them (splitDeep).
func (p *Picture) image(rows []float32, projDim int) Image {
	embd, deep := splitDeep(rows, p.Rows, projDim)
	return Image{Embd: embd, Grid: p.Grid, Key: &p.Key, Deep: deep}
}

// PicturePieces is Picture for a tower that cuts a picture into an overview
// and slices (MiniCPM-V): each piece of lay, preprocessed and named, in the
// order PictureSpans lays them out. Pillow's resampler needs no generated
// code, so it needs no State.
func (t *Tower) PicturePieces(img image.Image, lay Layout) ([]*Picture, error) {
	px, err := t.Pieces(img, lay)
	if err != nil {
		return nil, err
	}
	c := t.Cfg
	out := make([]*Picture, len(px))
	for i := range px {
		src := i
		if c.fault == faultSliceOrder && i > 0 {
			src = len(px) - i // the slices reversed, the overview kept
		}
		gh, gw := lay.Pieces[src].Grid(c.PatchSz)
		if out[i], err = c.picture(px[src], gh, gw); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// encodePicture is the picture's rows, on the vision segment's State: the
// image cache's when it holds them, the tower's otherwise. The rows are the
// caller's. The caller holds the pager (enterPager).
func (s *State) encodePicture(p *Picture) ([]float32, error) {
	c := s.vis.t.Cfg
	rows, _, err := s.encodeNamedKey(p.Key, p.GH, p.GW, func() ([]float32, error) {
		if c.Dynamic() || c.PosBuckets > 0 {
			return s.encodeGrid(p.Pixels, p.GH, p.GW)
		}
		return s.encode(p.Pixels)
	})
	if err != nil {
		return nil, err
	}
	if len(rows) != p.Rows*c.RowWidth() {
		return nil, fmt.Errorf("model: the tower emitted %d rows for a picture of %d", len(rows)/c.RowWidth(), p.Rows)
	}
	return rows, nil
}

// pictureRows resolves every Picture span of spans whose rows reach past
// position from (relative to the spans' start) into an embedding span: the
// vision segment encodes it, or the image cache answers. A span the prefix
// cache restored whole is left as it is -- its rows are never read. spans is
// not changed; the result is a copy when anything was encoded.
func (s *State) pictureRows(spans []Span, from int) ([]Span, error) {
	var out []Span
	at := 0
	for i, sp := range spans {
		n := sp.n(s.c.NEmbd)
		if sp.Picture != nil && sp.Embd == nil && at+n > from {
			vs, err := s.Vision()
			if err != nil {
				return nil, err
			}
			rows, err := vs.encodePicture(sp.Picture)
			if err != nil {
				return nil, err
			}
			if out == nil {
				out = append([]Span(nil), spans...)
			}
			out[i].Embd, out[i].Deep = splitDeep(rows, n, s.c.NEmbd)
		}
		at += n
	}
	if out == nil {
		return spans, nil
	}
	return out, nil
}
