package app

import (
	"github.com/gogpu/ui/dnd"
	"github.com/gogpu/ui/geometry"
)

// dropTarget takes OS file drops for the whole window. It registers with the
// window's dnd.Manager because gogpu.App.OnDragDrop is a single-slot setter
// that desktop.Run overwrites with its own.
type dropTarget struct{ s *Shell }

func (d dropTarget) CanAccept(data dnd.DragData) bool { return data.Kind == dnd.KindFile }
func (d dropTarget) DragEnter(dnd.DragData)           {}
func (d dropTarget) DragLeave()                       {}

func (d dropTarget) DragOver(dnd.DragData, geometry.Point) dnd.DropEffect {
	return dnd.DropCopy
}

func (d dropTarget) Drop(data dnd.DragData, _ geometry.Point) bool {
	f, ok := data.Payload.(dnd.FilePayload)
	if !ok || len(f.Paths) == 0 {
		return false
	}
	d.s.mu.Lock()
	fn := d.s.onDrop
	d.s.mu.Unlock()
	if fn == nil {
		return false
	}
	fn(f.Paths)
	return true
}

// acceptDrops registers the window-wide drop target. The bounds are the
// window's, refreshed on every resize by the shell.
func (s *Shell) acceptDrops(w, h float32) {
	if s.UI == nil || s.UI.Window() == nil {
		return
	}
	mgr := s.UI.Window().DndManager()
	if mgr == nil {
		return
	}
	bounds := geometry.NewRect(0, 0, w, h)
	if s.drop == nil {
		s.drop = dropTarget{s: s}
		mgr.RegisterTarget(s.drop, bounds)
		return
	}
	mgr.UpdateTargetBounds(s.drop, bounds)
}
