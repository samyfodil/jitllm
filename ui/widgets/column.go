package widgets

import (
	"github.com/gogpu/ui/event"
	"github.com/gogpu/ui/geometry"
	"github.com/gogpu/ui/state"
	"github.com/gogpu/ui/widget"
)

// Column lays a signal-driven list of children out vertically.
//
// It replaces core/listview, whose rows cannot receive events (the decorator
// returns false and the list's click recognizer wins the arena), whose height
// cache survives a resize, and whose RepaintBoundary textures outlive their
// rows. It gives up virtualisation knowingly: a conversation is tens of turns,
// and if one grows long enough to matter, cap what is built here.
type Column struct {
	widget.WidgetBase

	count state.ReadonlySignal[int]
	build func(i int) widget.Widget
	gap   float32

	// built is the children for the current count, rebuilt when it changes.
	built []widget.Widget
	// captured is the child that took the last press, so the rest of a drag
	// reaches it even once the pointer leaves its bounds.
	captured widget.Widget
	ctx      widget.Context
	n        int
	// key, when set, rebuilds every row when it moves: the same count of
	// different rows, as when another conversation is shown.
	key     state.ReadonlySignal[int]
	builtAt int
}

// KeyedBy rebuilds the rows whenever key moves, whatever the count.
func (c *Column) KeyedBy(key state.ReadonlySignal[int]) *Column { c.key = key; return c }

// NewColumn builds one child per index, rebuilding when count changes.
func NewColumn(count state.ReadonlySignal[int], gap float32, build func(i int) widget.Widget) *Column {
	c := &Column{count: count, build: build, gap: gap, n: -1}
	c.SetVisible(true)
	c.SetEnabled(true)
	return c
}

// Mount binds the count to the layout: adding a turn changes the height of
// everything below it, which a repaint alone cannot express.
func (c *Column) Mount(ctx widget.Context) {
	c.ctx = ctx
	if c.count != nil && ctx != nil {
		if sched := ctx.Scheduler(); sched != nil {
			c.AddBinding(state.BindToSchedulerLayout(c.count, c, sched))
			if c.key != nil {
				c.AddBinding(state.BindToSchedulerLayout(c.key, c, sched))
			}
		}
	}
	c.sync(ctx)
}

func (c *Column) Unmount() {
	for _, w := range c.built {
		widget.UnmountTree(w)
	}
	c.built, c.n = nil, -1
}

// sync rebuilds the children only when the count has moved; rebuilding every
// layout would discard every child's state each frame.
func (c *Column) sync(ctx widget.Context) {
	n := 0
	if c.count != nil {
		n = c.count.Get()
	}
	k := 0
	if c.key != nil {
		k = c.key.Get()
	}
	if n == c.n && k == c.builtAt {
		return
	}
	c.builtAt = k
	for _, w := range c.built {
		widget.UnmountTree(w)
	}
	c.built = make([]widget.Widget, 0, n)
	for i := 0; i < n; i++ {
		w := c.build(i)
		if w == nil {
			continue
		}
		if s, ok := w.(interface{ SetParent(widget.Widget) }); ok {
			s.SetParent(c)
		}
		if ctx != nil {
			widget.MountTree(w, ctx)
		}
		c.built = append(c.built, w)
	}
	c.n = n
}

func (c *Column) Children() []widget.Widget { return c.built }

func (c *Column) Layout(ctx widget.Context, cons geometry.Constraints) geometry.Size {
	c.sync(ctx)

	w := cons.MaxWidth
	if w >= geometry.Infinity {
		w = cons.MinWidth
	}
	child := geometry.Constraints{MinWidth: w, MaxWidth: w, MaxHeight: geometry.Infinity}

	var y float32
	for i, ch := range c.built {
		if i > 0 {
			y += c.gap
		}
		if s, ok := ch.(interface{ SetPosition(geometry.Point) }); ok {
			s.SetPosition(geometry.Pt(0, y))
		}
		sz := widget.LayoutChild(ch, ctx, child)
		if s, ok := ch.(interface{ SetBounds(geometry.Rect) }); ok {
			s.SetBounds(geometry.FromPointSize(geometry.Pt(0, y), geometry.Sz(w, sz.Height)))
		}
		y += sz.Height
	}
	sz := geometry.Sz(w, y)
	c.SetBounds(geometry.FromPointSize(c.Position(), sz))
	return sz
}

// Draw paints the children in the column's own frame: Layout places them at
// local (0, y) and Event already subtracts Bounds().Min, so Draw translates to
// match.
func (c *Column) Draw(ctx widget.Context, canvas widget.Canvas) {
	if !c.IsVisible() {
		return
	}
	canvas.PushTransform(c.Bounds().Min)
	for _, ch := range c.built {
		widget.StampScreenOrigin(ch, canvas)
		widget.DrawChild(ch, ctx, canvas)
	}
	canvas.PopTransform()
}

// Event dispatches to the child under the pointer, in that child's
// coordinates. The toolkit's convention is that a child's Bounds() are in its
// parent's space and a container subtracts its own origin before descending
// (see primitives.Box.dispatchMouseEvent).
func (c *Column) Event(ctx widget.Context, e event.Event) bool {
	me, isMouse := e.(*event.MouseEvent)
	if !isMouse {
		// Keyboard and focus go to everyone: the selection's ctrl+C has to
		// reach whichever paragraph holds it, and that is not a position.
		for _, ch := range c.built {
			if ch.Event(ctx, e) {
				return true
			}
		}
		return false
	}

	local := *me
	local.Position = me.Position.Sub(c.Bounds().Min)

	// A drag that leaves the child still belongs to it until the release, as
	// the window does with a captured widget; otherwise a text selection
	// could not grow past its first line.
	if c.captured != nil {
		switch local.MouseType {
		case event.MouseMove, event.MouseRelease:
			// Clamped into the captured child, because primitives.Box in
			// between refuses a position outside a child's bounds; the
			// selection lands on the nearest edge.
			clamped := local
			if b, ok := c.captured.(interface{ Bounds() geometry.Rect }); ok {
				clamped.Position = clampInto(local.Position, b.Bounds())
			}
			consumed := c.captured.Event(ctx, &clamped)
			if local.MouseType == event.MouseRelease {
				c.captured = nil
			}
			if consumed {
				return true
			}
		}
	}

	for _, ch := range c.built {
		if b, ok := ch.(interface{ Bounds() geometry.Rect }); ok {
			if !b.Bounds().Contains(local.Position) {
				// A press outside is still delivered, without consuming, so a
				// selection in another bubble can clear itself.
				if local.MouseType == event.MousePress {
					ch.Event(ctx, &local)
				}
				continue
			}
		}
		if ch.Event(ctx, &local) {
			if local.MouseType == event.MousePress {
				c.captured = ch
			}
			return true
		}
	}
	return false
}

// clampInto pulls a point inside r, staying one pixel in so a Contains check
// that excludes the far edge still passes.
func clampInto(p geometry.Point, r geometry.Rect) geometry.Point {
	if r.Width() <= 0 || r.Height() <= 0 {
		return p
	}
	x, y := p.X, p.Y
	if x < r.Min.X {
		x = r.Min.X
	}
	if x > r.Max.X-1 {
		x = r.Max.X - 1
	}
	if y < r.Min.Y {
		y = r.Min.Y
	}
	if y > r.Max.Y-1 {
		y = r.Max.Y - 1
	}
	return geometry.Pt(x, y)
}
