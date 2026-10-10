package screen

import (
	"strings"

	"github.com/gogpu/ui/core/chip"
	"github.com/gogpu/ui/core/progress"
	"github.com/gogpu/ui/primitives"
	"github.com/gogpu/ui/state"
	"github.com/gogpu/ui/widget"

	"github.com/jitllm/jitllm/ui/app"
	"github.com/jitllm/jitllm/ui/widgets"
)

// modelSlots is how many open models the bar can show.
const modelSlots = 6

// modelBar is the row of open models, one chip each, the active one selected.
// It is a switch, not a list: clicking a chip hands that model the session,
// the device and the worker pool, since two live pools or two tiers over one
// card must never coexist. Switching costs a full prefill: each turn is
// rendered through the incoming model's own template, and the KV namespace is
// per model, so no cached pages carry over.
func modelBar(sh *app.Shell) widget.Widget {
	row := make([]widget.Widget, 0, modelSlots+2)

	for i := 0; i < modelSlots; i++ {
		i := i
		at := func() (app.LoadedModel, bool) {
			ms := sh.Store.Models.Get()
			if i >= len(ms) {
				return app.LoadedModel{}, false
			}
			return ms[i], true
		}
		show := state.NewComputed(func() bool { _, ok := at(); return ok },
			sh.Store.Models.AsReadonly())

		c := chip.New(
			chip.LabelFn(func() string {
				m, ok := at()
				if !ok {
					return ""
				}
				return chipLabel(m.Name)
			}),
			chip.SelectedReadonlySignal(state.NewComputed(func() bool {
				m, ok := at()
				return ok && m.Active
			}, sh.Store.Models.AsReadonly())),
			chip.OnClick(func() {
				m, ok := at()
				if !ok {
					return
				}
				eng := depsOf(sh).Engine
				if eng == nil {
					sh.Store.Status.Set("no engine is attached: see screen.Attach")
					return
				}
				sh.Store.SetChatModel(m.Path)
				eng.Use(m.Path)
			}),
			chip.PainterOpt(sh.P.Chip),
		)
		row = append(row, widgets.Hide(show, c))
	}

	for i := 0; i < loadSlots; i++ {
		row = append(row, loadChip(sh, i))
	}
	// A strip, so models that do not fit are left off rather than painted over
	// the controls beside them; the Models tab lists every one.
	bar := widgets.NewStrip(sh.P.Space.S, row...)
	return widgets.Hide(state.NewComputed(func() bool {
		return len(sh.Store.Models.Get()) > 0 || sh.Store.Loading()
	}, sh.Store.Models.AsReadonly(), sh.Store.Loads.AsReadonly()), bar)
}

// loadSlots is how many loads the bar shows at once.
const loadSlots = 3

// loadChip is the i-th model on its way in: a spinner, its name and the stage
// it is at, shaped like the open models' chips it will become one of.
func loadChip(sh *app.Shell, i int) widget.Widget {
	p := sh.P
	at := func() (app.Loading, bool) {
		ls := sh.Store.Loads.Get()
		if i >= len(ls) {
			return app.Loading{}, false
		}
		return ls[i], true
	}
	show := state.NewComputed(func() bool { _, ok := at(); return ok }, sh.Store.Loads.AsReadonly())
	label := reactive(sh, sh.Store.Loads.AsReadonly(), func() string {
		l, ok := at()
		if !ok {
			return ""
		}
		// The first is the one being opened, and the pill says at what stage;
		// the rest are waiting their turn.
		if i > 0 {
			return chipLabel(modelLabel(l.Path)) + " \u00b7 waiting"
		}
		return chipLabel(modelLabel(l.Path))
	})
	text := p.Role(primitives.Text("").ContentSignal(label).Color(p.Muted()), p.Type.LabelLarge)
	var inner widget.Widget = text
	if i == 0 {
		inner = primitives.HBox(
			progress.New(
				progress.Indeterminate(true),
				progress.Size(14),
				progress.StrokeWidth(2),
				progress.PainterOpt(p.Progress),
			),
			text,
		).Gap(p.Space.S)
	}
	return widgets.Hide(show, primitives.Box(inner).
		PaddingXY(p.Space.M, 6).
		Rounded(p.Shape.Small).
		BorderStyle(1, p.Colors.OutlineVariant))
}

// turnPalette is the colours an assistant turn is named in, one per open
// model in load order. The name is coloured rather than the bubble, whose
// background already carries the role.
func turnPalette(sh *app.Shell) []widget.Color {
	c := sh.P.Colors
	return []widget.Color{c.Primary, c.Tertiary, c.Secondary, c.Error, c.OnSurfaceVariant, c.Outline}
}

// turnColour is the colour for one turn's name, or fallback when the turn
// names no model.
func turnColour(sh *app.Shell, t app.Turn, fallback widget.Color) widget.Color {
	if t.Role != app.RoleAssistant || t.Model == "" {
		return fallback
	}
	p := turnPalette(sh)
	return p[t.Colour%len(p)]
}

// chipLabel is a model name at the width a chip in a row of chips can have.
// An HBox positions a sibling by the width its child measured, and
// Canvas.DrawText does not clip, so an overlong label draws the next widget
// over it; shortening the label is the only fix. The full name is on the
// Models tab and in the summary row.
func chipLabel(name string) string {
	name = strings.TrimSuffix(name, ".jlm")
	if r := []rune(name); len(r) > chipLabelRunes {
		return string(r[:chipLabelRunes-1]) + "\u2026"
	}
	return name
}

// chipLabelRunes is sized so modelSlots chips, the phase, a 160px dropdown and
// Clear all fit the 1000px minimum window width, with room for the chip's
// padding.
const chipLabelRunes = 16
