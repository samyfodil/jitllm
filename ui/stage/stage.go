// Package stage renders the app headlessly the way the window does, frame by
// frame, and checks each frame for what a person would see as broken.
//
// offscreen.Render is not that: it never mounts the tree, so no signal binding
// exists and nothing a binding should re-measure ever is. A screenshot taken
// with it can only show data that was in place before the first layout -- the
// one order the running app never uses. A Stage mounts once with a scheduler,
// drains the shell's queue before every frame as the window's OnUpdate does,
// and keeps the tree and its layout caches between frames, so a widget that
// fails to re-measure when its data arrives fails here too.
package stage

import (
	"fmt"
	"image"
	"image/draw"
	"strings"

	"github.com/gogpu/gg"
	"github.com/gogpu/ui/geometry"
	"github.com/gogpu/ui/primitives"
	"github.com/gogpu/ui/render"
	"github.com/gogpu/ui/state"
	"github.com/gogpu/ui/transition"
	"github.com/gogpu/ui/widget"

	"github.com/jitllm/jitllm/ui/app"
)

// Stage is one window's tree, mounted.
type Stage struct {
	sh   *app.Shell
	root widget.Widget
	ctx  *widget.ContextImpl
	w, h int
}

// New mounts the shell's whole window -- tabs, status bar and dialogs -- at
// w x h. Register the screens and attach their dependencies first.
func New(sh *app.Shell, w, h int) *Stage {
	// A Box, because a root that is not one never has its bounds set off the
	// window (UPSTREAM.md #9).
	return NewWith(sh, primitives.Box(sh.Build()), w, h)
}

// NewWith mounts any tree, for a stage of one screen.
func NewWith(sh *app.Shell, root widget.Widget, w, h int) *Stage {
	ctx := widget.NewContext()
	ctx.SetThemeProvider(sh.M3)
	ctx.SetScale(1)
	ctx.SetWindowSize(geometry.Sz(float32(w), float32(h)))
	// The scheduler's flush is the window's redraw request; a stage draws every
	// frame anyway. What matters is that bindings exist, and they bind to it.
	ctx.SetScheduler(state.NewScheduler(func([]widget.Widget) {}))
	widget.MountTree(root, ctx)
	return &Stage{sh: sh, root: root, ctx: ctx, w: w, h: h}
}

// Text is one string the frame painted, with the box its glyphs cover and the
// clip it was drawn under, both in window coordinates.
type Text struct {
	S         string
	Ink, Clip geometry.Rect
	// Seq is when it was painted, against Fill.Seq.
	Seq int
}

// Fill is an opaque shape painted over whatever was there, clipped as the
// canvas clips shapes. A string a later fill covers cannot be seen.
type Fill struct {
	Rect geometry.Rect
	Seq  int
}

// Frame is one rendered frame.
type Frame struct {
	Image *image.RGBA
	Texts []Text
	Fills []Fill
	W, H  int
}

// Frame runs what the window runs for one frame: the posted work, a layout of
// the root, and a draw.
func (s *Stage) Frame() Frame {
	s.sh.DrainQueue(0)
	s.root.Layout(s.ctx, geometry.Loose(geometry.Sz(float32(s.w), float32(s.h))))

	dc := gg.NewContext(s.w, s.h)
	defer dc.Close()
	cv := render.NewCanvas(dc, s.w, s.h)
	cv.Clear(s.sh.P.Background())
	rec := &recorder{Canvas: cv}
	widget.DrawTree(s.root, s.ctx, rec)

	img := image.NewRGBA(image.Rect(0, 0, s.w, s.h))
	draw.Draw(img, img.Bounds(), dc.Image(), image.Point{}, draw.Src)
	return Frame{Image: img, Texts: rec.texts, Fills: rec.fills, W: s.w, H: s.h}
}

// Paints reports whether the frame painted s, whole, as one string.
func (f Frame) Paints(s string) bool {
	for _, t := range f.Texts {
		if strings.Contains(t.S, s) {
			return true
		}
	}
	return false
}

// Problems is everything wrong with the frame that a person would see:
//
//   - two strings painted over each other -- the symptom of a widget whose
//     layout did not grow with its content, or of a caption that does not fit
//   - a string painted outside its clip, which the toolkit does not enforce for
//     text (UPSTREAM.md #8), so it lands on whatever is beside it
//   - a string past the window's edge
//
// Only what is still visible counts: a stray piece that an opaque fill painted
// later covers -- the status bar over a half-scrolled row -- is not seen.
func (f Frame) Problems() []string {
	var out []string
	win := geometry.NewRect(0, 0, float32(f.W), float32(f.H))
	for i, a := range f.Texts {
		if p, ok := f.escaped(a, a.Clip); ok {
			out = append(out, fmt.Sprintf("%q paints outside its clip at %v", a.S, p))
		}
		if p, ok := f.escaped(a, win); ok {
			out = append(out, fmt.Sprintf("%q paints past the window at %v", a.S, p))
		}
		for _, b := range f.Texts[i+1:] {
			// They clash unless the first painted was erased before the second.
			if o, ok := intersect(a.Ink, b.Ink); ok && !f.covered(o, min(a.Seq, b.Seq)) {
				out = append(out, fmt.Sprintf("%q and %q paint over each other at %v", a.S, b.S, o))
			}
		}
	}
	return out
}

// escaped is the first visible part of t's ink outside outer.
func (f Frame) escaped(t Text, outer geometry.Rect) (geometry.Rect, bool) {
	r := t.Ink
	for _, p := range []geometry.Rect{
		geometry.NewRect(r.Min.X, r.Min.Y, outer.Min.X-r.Min.X, r.Height()),     // left
		geometry.NewRect(outer.Max.X, r.Min.Y, r.Max.X-outer.Max.X, r.Height()), // right
		geometry.NewRect(r.Min.X, r.Min.Y, r.Width(), outer.Min.Y-r.Min.Y),      // above
		geometry.NewRect(r.Min.X, outer.Max.Y, r.Width(), r.Max.Y-outer.Max.Y),  // below
	} {
		if p.Width() > slack && p.Height() > slack && !f.covered(p, t.Seq) {
			return p, true
		}
	}
	return geometry.Rect{}, false
}

// covered reports whether one opaque fill painted after seq hides r.
func (f Frame) covered(r geometry.Rect, seq int) bool {
	for _, fl := range f.Fills {
		if fl.Seq > seq && inside(r, fl.Rect) {
			return true
		}
	}
	return false
}

// slack is how far a glyph box may cross a boundary before it is a problem:
// the ink box is an estimate from the font size, not the glyphs' outlines.
const slack = 2

func inside(r, outer geometry.Rect) bool {
	return r.Min.X >= outer.Min.X-slack && r.Min.Y >= outer.Min.Y-slack &&
		r.Max.X <= outer.Max.X+slack && r.Max.Y <= outer.Max.Y+slack
}

func intersect(a, b geometry.Rect) (geometry.Rect, bool) {
	x0, y0 := max(a.Min.X, b.Min.X), max(a.Min.Y, b.Min.Y)
	w, h := min(a.Max.X, b.Max.X)-x0, min(a.Max.Y, b.Max.Y)-y0
	return geometry.NewRect(x0, y0, w, h), w > slack && h > slack
}

// recorder forwards every call to the real canvas and notes each string it
// paints. The optional interfaces the toolkit asserts for are forwarded too,
// or wrapping the canvas would quietly switch those features off.
type recorder struct {
	widget.Canvas
	texts []Text
	fills []Fill
	seq   int
}

// fill notes an opaque shape, as the canvas clips it.
func (r *recorder) fill(rect geometry.Rect, c widget.Color) {
	if c.A < 0.999 {
		return
	}
	if b, ok := intersect(rect.Translate(r.TransformOffset()), r.ClipBounds()); ok {
		r.seq++
		r.fills = append(r.fills, Fill{Rect: b, Seq: r.seq})
	}
}

func (r *recorder) DrawRect(rect geometry.Rect, c widget.Color) {
	r.fill(rect, c)
	r.Canvas.DrawRect(rect, c)
}

func (r *recorder) FillRectDirect(rect geometry.Rect, c widget.Color) {
	r.fill(rect, c)
	r.Canvas.FillRectDirect(rect, c)
}

func (r *recorder) DrawRoundRect(rect geometry.Rect, c widget.Color, radius float32) {
	// The corners are not covered; a rounded fill hides its inset square.
	r.fill(geometry.NewRect(rect.Min.X+radius/2, rect.Min.Y+radius/2, rect.Width()-radius, rect.Height()-radius), c)
	r.Canvas.DrawRoundRect(rect, c, radius)
}

func (r *recorder) note(s string, bounds geometry.Rect, size float32, bold bool, align widget.TextAlign) {
	b := bounds.Translate(r.TransformOffset())
	clip := r.ClipBounds()
	// The canvas skips text whose bounds miss the clip, so it never painted.
	if s == "" || !clip.Intersects(b) {
		return
	}
	// Placed as Canvas.DrawText places it: centred vertically in the bounds,
	// shifted by the alignment when it is narrower than them. The height is the
	// font size, a little under ascent+descent, so two tight lines do not read
	// as overlapping.
	w := r.MeasureText(s, size, bold)
	x := b.Min.X
	if w < b.Width() {
		x += (b.Width() - w) * float32(align.Float64())
	}
	y := b.Min.Y + (b.Height()-size)/2
	r.seq++
	r.texts = append(r.texts, Text{S: s, Ink: geometry.NewRect(x, y, w, size), Clip: clip, Seq: r.seq})
}

func (r *recorder) DrawText(s string, bounds geometry.Rect, size float32, c widget.Color, bold bool, align widget.TextAlign) {
	r.note(s, bounds, size, bold, align)
	r.Canvas.DrawText(s, bounds, size, c, bold, align)
}

func (r *recorder) DrawStyledText(s string, bounds geometry.Rect, st widget.TextStyle) {
	r.note(s, bounds, st.FontSize, st.Bold, st.Align)
	if d, ok := r.Canvas.(widget.StyledTextDrawer); ok {
		d.DrawStyledText(s, bounds, st)
		return
	}
	r.Canvas.DrawText(s, bounds, st.FontSize, st.Color, st.Bold, st.Align)
}

// The optional interfaces, forwarded to the real canvas when it has them.

func (r *recorder) MeasureStyledText(s string, st widget.TextStyle) float32 {
	if d, ok := r.Canvas.(widget.StyledTextDrawer); ok {
		return d.MeasureStyledText(s, st)
	}
	return r.Canvas.MeasureText(s, st.FontSize, st.Bold)
}

func (r *recorder) StrokeArcStyled(c geometry.Point, radius float32, start, sweep float64, col widget.Color, w float32, cap widget.LineCap) {
	if a, ok := r.Canvas.(widget.ArcStroker); ok {
		a.StrokeArcStyled(c, radius, start, sweep, col, w, cap)
		return
	}
	r.Canvas.StrokeArc(c, radius, start, sweep, col, w)
}

func (r *recorder) FillSVGPath(d string, viewBox float32, b geometry.Rect, c widget.Color) {
	if f, ok := r.Canvas.(widget.SVGFiller); ok {
		f.FillSVGPath(d, viewBox, b, c)
	}
}

func (r *recorder) RenderSVG(x []byte, b geometry.Rect, c widget.Color) {
	if f, ok := r.Canvas.(widget.SVGRenderer); ok {
		f.RenderSVG(x, b, c)
	}
}

func (r *recorder) SetDamageTracking(on bool) {
	if d, ok := r.Canvas.(widget.DamageController); ok {
		d.SetDamageTracking(on)
	}
}

func (r *recorder) IsBoundaryRecording() bool {
	b, ok := r.Canvas.(widget.BoundaryRecorder)
	return ok && b.IsBoundaryRecording()
}

func (r *recorder) PushOpacity(o float64) {
	if p, ok := r.Canvas.(transition.OpacityPusher); ok {
		p.PushOpacity(o)
	}
}

func (r *recorder) PopOpacity() {
	if p, ok := r.Canvas.(transition.OpacityPusher); ok {
		p.PopOpacity()
	}
}
