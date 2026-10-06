# gogpu/ui issues found while building this app

Against `github.com/gogpu/ui@v0.1.54` and `github.com/gogpu/gg@v0.52.3`.
Each was found by a real defect in this app, and each is recorded with the
symptom first, because the symptom is what makes the case.

One has been reported: #16 is gogpu/ui#236. The rest have not.

---

## 1. `itemDecorator` does not stamp its child's screen origin

**Symptom.** Nothing inside a `core/listview` row can be clicked or hovered. A
`core/collapsible` in a row never opens; a `core/button` never fires.

**Cause.** `core/listview/decorator.go:70-78` calls `d.child.Draw(ctx, canvas)`
directly after `SetBounds`, without `widget.StampScreenOrigin(d.child, canvas)`.
Every other container in the toolkit stamps before drawing —
`primitives/box.go:676`, `primitives/expanded.go:81`,
`primitives/themescope.go:126`, `core/scrollview/widget.go:180`,
`core/listview/virtual_content.go:115`, `core/collapsible/collapsible.go:225`.

So a row's child keeps `screenOrigin = (0,0)` and `widget/base.go:424-429` gives
it a `ScreenBounds` at the WINDOW's top-left with the child's size. Both hit
tests walk by `ScreenBounds` and return early there:
`app/window.go:1443-1448` (gesture) and `:1670-1675` (hover).

**Fix.** One line before `decorator.go:77`:

    widget.StampScreenOrigin(d.child, canvas)

and probably `child.SetParent(d)` in `newItemDecorator` (`decorator.go:30`), to
match `primitives/box.go:89-91`.

---

## 2. `ListView` claims every click with an empty handler

**Symptom.** Even with #1 fixed, a row's children still never win a gesture.

**Cause.** `core/listview/widget.go:286-295` returns a `ClickRecognizer` for the
whole list from `GestureHitTest`, and its `OnClick` body is a comment
(`widget.go:117-126`). The walk appends a parent's recognizers before
descending (`app/window.go:1452-1468`), `Route` iterates in add order
(`gesture/arena.go:188`), and on `PointerUp` the first resolver wins outright
(`gesture/click.go:226`, `arena.go:118-119`) — so a no-op takes every click.

**Fix.** Return no recognizer when there is no handler, or make the list's
recognizer defer to descendants.

---

## 3. `ListView` never invalidates on a WIDTH change

**Symptom.** After a window resize, rows are the wrong height: text too tall for
its box, or a box too tall for its text. Long content appears truncated with
blank space under it.

**Cause.** `core/listview/widget.go:137-143` invalidates the item cache only
when the item COUNT changes. `widget.go:168` assigns `w.viewportWidth = size.Width`
with no comparison against the previous value, and `heights.measured`
(`heights.go:33-34`) is written only by `setMeasured` and cleared only by
`initLazy`, which a resize never calls. Off-screen rows keep heights measured at
the old width, and `currentEstimate()` (`heights.go:324-329`) averages
old-width and new-width measurements into one number that drives `totalHeight`,
`offsetAt` and `visibleRange`.

**Fix.** At `widget.go:137`, compare the resolved width against `w.viewportWidth`
and on a change do what `InvalidateData` already does at `:338-344` —
`w.cache.invalidate()` and, in lazy mode, `heights.initLazy()`.

**Workaround in this app.** `gogpu.App.OnResize` (the one callback
`desktop.Run` does not claim) ticks a signal the transcript watches, which calls
`lv.InvalidateData()`.

---

## 4. No selectable read-only text anywhere in the toolkit

**Symptom.** Text in a chat transcript cannot be selected or copied.

**Cause.** The only selection implementation is
`core/textfield/selection.go:10-49`, unexported, single-line, reachable only as
`textfield.Widget.sel`. Its click-to-rune mapping is
`internal/textmetrics/textmetrics.go:49` (`RuneIndexFromX`), and `internal/` is
not importable from another module. `core/textfield/options.go` has no
`ReadOnly` and no `Multiline`, and `Layout` hard-returns a one-line height
(`core/textfield/widget.go:115-133`).

**Fix.** Either export a selection primitive with a public coordinate mapping,
or add `ReadOnly` + `Multiline` to `textfield`.

**Note.** `widget.ClipboardWrite`/`ClipboardRead` (`widget/clipboard.go:24,34`)
are fine and already used here.

---

## 5. `desktop.Run` silently overwrites application callbacks

**Symptom.** `gpuApp.OnClose(...)` and `gpuApp.OnDragDrop(...)` set by an
application have no effect. In this app that meant NO SETTING WAS EVER SAVED:
window size, theme, model directories, chat mode, system prompt, sampling and
last model were all written to a config that was never flushed.

**Cause.** Both are single-slot setters on `gogpu.App` (`app.go:214`, `:233`),
and `desktop.Run` installs its own at `desktop/desktop.go:81` and `:90`, after
the application has set them.

**Fix.** Chain rather than replace, or document that `Run` owns these two and
provide `desktop.OnClose`/`OnDrop` hooks. At minimum the doc comment on
`gogpu.App.OnClose` should say `desktop.Run` will take it.

**Workaround in this app.** Run the shutdown after `desktop.Run` returns, and
register a `dnd.DropTarget` on the window's manager instead of using the
callback.

---

## 6. `gg/text` shapes no complex script

**Symptom.** Arabic renders as isolated letterforms. Devanagari, Thai, Khmer and
Myanmar would be worse.

**Cause.** `gg/text/shaper_own.go:300-307` hardcodes the GSUB feature set to
`{ccmp, liga, clig, rlig, dlig}` — all Latin typography — and
`ot_layout.go:551-564` collects a lookup only if its tag is in that set. The
Arabic joining features `init`/`medi`/`fina`/`isol` appear nowhere in the
module. `gsub.go:112-118` implements lookup types 1-4 only; 5, 6 and 8 are the
contextual ones Arabic and Indic need. `gsub.go:52` carries one buffer-wide
feature list, where joining needs a per-GLYPH feature mask.

Bidi is present but unused for placement: `text/segment.go:57-90` runs the UBA
through `x/text/unicode/bidi` and produces correct levels and per-segment
`Direction`, but `shapeSegments` (`layout.go:266-306`) places runs in logical
order and only copies `seg.Direction`. Nothing reverses anything.

**Verified empirically.** Shaping `مرحبا` with DejaVu Sans through
`text.NewFontSource(...).Face(14).Glyphs(...)` gives meem as glyph 1390 both
alone and inside the word — the base cmap glyph. Its initial form is 5341.

**Fix.** Ask the FONT which features it declares for the run's script instead of
a fixed list; add a per-glyph feature mask threaded through `applyGSUB`; add
lookup type 6; reverse RTL runs by the level `segment.go` already computes.

**Note.** The script-specific INPUT is one Unicode property, `Joining_Type`, and
it is data — shared by Arabic, Syriac, Mongolian, N'Ko and Adlam. This is one
engine, not a table per script.

---

## 7. `gogpu/ui` never builds a `text.MultiFace`

**Symptom.** No per-rune font fallback. A family that lacks a glyph draws
`.notdef` rather than falling through to one that has it.

**Cause.** `gg/text.MultiFace` (`text/multi.go`) is real per-rune fallback —
`faceForRune` picks the first face with the glyph — and implements the whole
`Face` interface, which `scene.DrawText` accepts. `gogpu/ui` never constructs a
`text.Face` anywhere (zero non-test hits): `internal/render/scene_canvas.go:415`
resolves ONE `*FontSource` and calls `source.Face(size)`.

**Fix.** Build a `MultiFace` over the registered families in
`internal/render/fontregistry.go`. Per-rune fallback then works for every widget
with no application code at all.

**Workaround in this app.** `app.FamilyFor` walks a chain and `widgets.Paragraph`
splits a line into runs by coverage — a weaker reimplementation of `MultiFace`,
one layer too high.

---

## 8. `Canvas.PushClip` does not clip TEXT

**Symptom.** A data-table cell wider than its column is painted over the next
column, with a clip pushed around it. The Models table read
`gemma-2-2b-it-Q4_K_M.jlmcontainer`.

**Cause.** `internal/render/canvas.go:319-336` `PushClip` calls
`dc.ClipRect(...)`, and its own comment says gg uses that "for CPU-side
ClipCoverage masking". `Canvas.DrawText` (`canvas.go:450-491`) ends in
`c.dc.DrawString(s, x, baselineY)`, which does not consult it. Shapes are
clipped; glyphs are not.

**Why it is worth fixing rather than documenting.** A painter that pushes a
clip and draws text has no way to know it did nothing — and a test that asserts
the clip was pushed passes. That is exactly how this app shipped a
non-functioning fix.

**Workaround in this app.** Measure with `Canvas.MeasureText` and truncate the
string with an ellipsis before handing it to the painter.

---

## 9. `offscreen.Render` does not set the root widget's bounds

**Symptom.** Rendering a screen whose root is a `core/splitview` produces a
blank image. Roots built from `primitives.Box`/`VBox` render fine.

**Cause.** `offscreen/renderer.go:120-151` calls `w.Layout(ctx, constraints)`
and then `widget.DrawTree(w, ctx, canvas)`, and never `SetBounds` on the root.
A widget that derives its bounds from a parent-set position therefore has the
zero rect, and `splitview.Draw` (`core/splitview/splitview.go:341-345`) returns
early on `bounds.IsEmpty()`.

**Fix.** `Render` should set the root's bounds to the render rect before
drawing, the way `app.Window` does for the real tree.

**Workaround in this app.** Wrap the root in a `primitives.Box`, which sets its
own bounds in Layout.

---

## 10. `primitives.Text.Ellipsis()` does not truncate anything

**Symptom.** A long string in a bounded box is drawn at its natural width, over
whatever is beside it. With `MaxLines(1).Ellipsis()` set. Rendered at 1000px
this app showed the model summary under the Chat dropdown, the pager line under
two buttons, and a table cell over the next column — three symptoms, one cause.

**Cause.** `Ellipsis()` sets `style.Overflow = TextOverflowEllipsis`
(`primitives/text.go:175-177`). The only readers are the MEASURE path
(`text.go:333` says so) and tests. `TextWidget.Draw` (`text.go:235-263`) takes
`t.Content()` and hands the whole string to `canvas.DrawStyledText` or
`canvas.DrawText`. Neither truncates, and neither clips — see #8.

So the widget MEASURES as if it were truncated and DRAWS as if it were not,
which is the worst of the two: the layout reserves the small size and the paint
takes the large one.

**Fix.** Truncate in `Draw`, against the bounds, when `Overflow` is
`TextOverflowEllipsis` — the width is known there and `Canvas.MeasureText`
exists for the cut.

**Workaround in this app.** `widgets.Paragraph.MaxLines(n)` truncates in Draw
where the real width is known. Every single-line readout in the app goes
through it now.

---

## 11. `collapsible` cannot be opened by anything except its own click

**Symptom.** A disclosure bound to app state renders the right chevron and the
right title, and does not open. Not when the state is written by a button
elsewhere, not when it is written by a background worker, not when the state
was true before the widget was built. Only a click on its own header works.

**Cause — two, and either alone is enough.**

1. `ExpandedReadonlySignal` is a ONE-WAY bind, which the option name does not
   say and the doc comment does not mention. `setExpandedState`
   (`collapsible.go:375-402`) writes to `cfg.expandedSignal`, which that option
   leaves nil, while `cfg.isExpanded()` (`config.go:52-57`) reads
   `readonlyExpandedSignal` in PREFERENCE to it. So the press is received, the
   widget toggles nothing, and the next frame reads the same value back. A
   disclosure bound this way is decoration.

2. Open-ness is an ANIMATION, not a boolean. `Layout` returns
   `headerH + contentH*w.progress` (`collapsible.go:204`), and `progress` is
   reconciled with the configured state in exactly two places: once in `Mount`
   (`:87-90`) and inside `setExpandedState`. The signal binding is
   `BindToSchedulerLayout` (`:334-340`), which invalidates the layout and
   touches `progress` not at all — so an external write re-measures to the
   SAME height and the section stays shut. This one bites `ExpandedSignal`,
   the two-way option, just as hard.

**Why the obvious fix is also wrong.** Swapping to `ExpandedSignal` fixes (1)
and not (2), and `SetExpanded` cannot rescue it: that method short-circuits on
`current == expanded` (`:154-160`) and `current` comes from `isExpanded()`,
which reads the signal the caller has just written. Every external write is a
no-op by construction.

**Fix.** Reconcile `progress` with `ResolvedExpanded()` at the top of `Layout`
when no animation is active — the state is already there, it is simply only
consulted at Mount. And either make `ExpandedReadonlySignal` drive
`setExpandedState` on change, or document it as display-only.

**Workaround in this app.** Do not bind the signal at all. Build with
`Expanded(initial)` + `OnToggle(sig.Set)` so `cfg.expanded` stays the widget's
OWN field, and drive it from a watcher that calls `SetExpanded` when the signal
moves. The equality short-circuit then compares against the widget's state
rather than against the value just written, so an external write genuinely
differs and the transition runs; a click round-trips through `OnToggle` and the
watcher's callback short-circuits, which is what it is for. See
`screen/session_transcript.go`'s `thinkingSection` and
`screen.TestAWriteFromOutsideOpensTheDisclosure`, which is red against both the
one-way bind and the naive two-way one.

**Also.** `handleMouseEvent` rejects any press whose `Button` is not
`ButtonLeft` (`event.go:69-72`), and `event.ButtonNone` is the ZERO VALUE of
`event.Button`. A synthesised `event.MouseEvent{MouseType: event.MousePress}`
is silently ignored, which in a test reads exactly like the bug above.

---

## 12. `primitives.Image` draws a placeholder and has no way to carry pixels

**Symptom.** An image widget built from `primitives.Image(...)` renders a light
grey rectangle with a cross through it, whatever source it is given.

**Cause.** `ImageWidget.Draw` (`primitives/image.go:165-203`) paints the
placeholder unconditionally and never calls `canvas.DrawImage`; its own doc
(`:159-164`) says full rendering "will be provided by the rendering backend".
It cannot be, because the source has nothing to hand over:
`ImageSource` (`primitives/style.go:223`) is `{ Bounds() [2]float32 }` — the
natural size and no pixels — and no concrete implementation ships.

Meanwhile `widget.Canvas.DrawImage(img image.Image, at geometry.Point)`
(`widget/canvas.go:133`) is real on both render paths
(`internal/render/canvas.go:588`, `internal/render/scene_canvas.go:479`) and is
recorded by `uitest.MockCanvas` (`DrawImageCall{Image, At}`), so the missing
piece is only the widget's side.

**And `DrawImage` takes a point, not a destination rect.** Both canvases blit
at the image's own pixel size, so a widget laid out at 96x96 over a 4000x3000
photo would draw 4000x3000 — scaling is left to every caller.

**Fix.** Let `ImageSource` expose an `image.Image` (or add a concrete
`ImageSource` over one), have `Draw` call `canvas.DrawImage` when it has
pixels, and either add a scaled `DrawImageRect` to `Canvas` or scale the
source to the laid-out size once, in the widget.

**Workaround in this app.** `widgets.Thumb` decodes the file, scales it once
on the CPU with `golang.org/x/image/draw` (already in the module graph through
gg), caches the result by path, and calls `DrawImage` itself.
`widgets.TestThumbDrawsTheScaledPicture` asserts the call, its origin, its
size and the colour at the centre of the picture — the pixel, because a
placeholder of the right size would pass any check on bounds alone.

---

## 13. `core/dialog` has a fixed height and never lays out its content

**Symptom.** A dialog's message runs past the right edge of the dialog, and a
message of more than two lines is drawn over the action buttons.

**Cause.** `computeDialogBounds` (`core/dialog/widget.go:201-215`) sizes the
surface as `titleAreaHeight + actionAreaHeight + 2*contentPadding` — 160 px
whatever the content is — and `surfaceWidget.Draw` (`:340-350`) only calls
`SetBounds` and `Draw` on the content widget; `Layout` is never called on it.
The content band is therefore 48 px tall, and a `primitives.Text` in it paints
at its natural width (see #10).

**Fix.** Lay the content out against `maxW - 2*contentPadding` and add its
height to `dialogH` (still capped by `MaxHeight` and the window margin).

**Workaround in this app.** `widgets.DialogHost` renders the message as a
wrapping `widgets.Paragraph` capped at two lines, and messages are written to
fit.

---

## 14. `core/datatable` has no double-click or row-activate callback

**Symptom.** Double-clicking a model in the Models table selects it and does
nothing else. There is no way to open a row by double-click, which is what a
file list is expected to do.

**Cause.** The table's click recognizer is built with `MaxClickCount: 1`
(`core/datatable/datatable.go:365`) and its `OnClick` body is a comment, so
the gesture layer's click count (`gesture/click.go:22`, default 3) never
reaches anything. A row press goes to `handleContentMousePress`
(`:1182-1197`), which only selects. The one callback, `OnRowSelect`
(`:228`), fires on every selection change (`:1130`, `:1224`) and also on
Enter/Space (`:1075`), so a handler cannot tell "moved to this row" from
"open this row".

**Fix.** An `OnRowActivate(func(row int))` option, fired on a double-click of
a row (raise `MaxClickCount` and pass `ClickCount` through) and on Enter,
with `OnRowSelect` left to mean selection only.

**Workaround in this app.** None: the Models screen does not offer
double-click, and Load is the button.

---

## 15. `primitives.HBox` ignores `CrossAlign`

**Symptom.** A 2 px coloured `Box` beside a paragraph in an `HBox`, meant as a
left rule, is never drawn.

**Cause.** `layoutHorizontalSimple` and `layoutHorizontalTwoPass`
(`primitives/box.go:510-620`) lay every child out with the row's constraints
and place it at `pad.Top` with the size it measured; neither reads
`b.crossAlign`, which only the vertical layout honours (`box.go:351`). An
empty box therefore measures, and stays, 0 tall.

**Fix.** After the row's height is known, re-lay out children under
`CrossAxisStretch` with a tight height, as the vertical layout does for width.

**Workaround in this app.** The notes' rule was removed; they are indented and
muted instead.

---

## 16. The text cursor is placed by shaping the prefix in ISOLATION

**Symptom.** In `core/textfield` the caret sits about a pixel to the right of
where the character actually starts, so it overlaps the glyph instead of
sitting in the gap before it. It is not uniform: it depends on which letters
are on either side of the caret, so it looks intermittent and unreproducible
until you notice which words it happens in. Typing `To The Top` shows it;
typing `what is 1+1?` never does.

**Cause.** `core/textfield/widget.go:265`:

```go
cursorRelative = tm.Canvas.MeasureText(string(runes[:runePos]), tm.FontSize, false)
```

The prefix is shaped **on its own**. The text itself is drawn by shaping the
WHOLE string — `internal/render/canvas.go:450` `DrawText` ends in
`dc.DrawString(s, x, baselineY)` — so any kerning pair spanning the cut is
applied to the glyphs and missing from the caret. Kern pairs such as `To`,
`AV`, `Ya` are negative: in context the second glyph is pulled left, and the
isolated measurement never sees that pull.

★ **AND THE KERNING IS ONLY HALF OF IT, WHICH THE FIRST VERSION OF THIS ENTRY
GOT WRONG.** `Canvas.MeasureText` reaches `text.Measure` -> `Face.Advance`,
which sums **grid-fitted** per-glyph advances and applies **no GPOS at all**;
the renderer positions from `text.Shape`, which uses the font's own advances
plus GPOS. So the two disagree even where nothing kerns, and the error
ACCUMULATES along the string rather than appearing at one pair.

Measured against Inter-Regular at 14px, caret position against the pen position
of the same cluster in the shaped run:

| string | worst caret error | the kerning term alone |
|---|---|---|
| `AV Wave` | **3.366 px** | 0.957 px |
| `Vy Ya P.` | 3.310 px | 1.025 px |
| `common sense` | **2.898 px** | **0.000 px** |
| `Yo, Tavi.` | 2.670 px | 1.073 px |
| `To The Top` | 1.399 px | 1.094 px |

**`common sense` is the control and it is the whole point: it has no kerning
pairs at all and still drifts 2.9 px.** That is the advance-source term on its
own. The earlier control here was `what is 1+1?` at 0.000 px, which is a clean
control for the KERNING term and not for the error -- it drifts 0.895 px from
the advance source, so quoting it as "exactly right" understated the defect by
roughly three times.

A second, smaller term rides along: `DrawText` snaps the text origin with
`x = math.Round(x)` (canvas.go:487) while the caret's base is not rounded, so
the two disagree by up to half a pixel independently of the kerning.

**Fix.** The shaper already answers this and says so in its own doc comment --
`gg/text.ShapedGlyph.Cluster` is *"the source character index in the original
text. Used for hit testing and cursor positioning."* Shape `DisplayText` ONCE,
then take the caret x as the sum of `XAdvance` over the glyphs whose `Cluster`
is below the cursor's rune index. That is the same loop `MeasureText` already
runs, stopped at a cluster instead of applied to a substring, and it is cheaper
than the current code because the whole string is shaped once per paint rather
than a prefix re-shaped per cursor move. It also fixes hit testing, which has
the same error with the opposite sign: a click lands on the wrong side of a
kerned pair.

It additionally makes a caret position inside a ligature or a combining
sequence expressible, which substring measurement cannot represent at all.

**Workaround in this app.** None available. `CursorRect` reaches the painter
already computed (`core/textfield/painter.go:52`), so a custom painter cannot
correct it without re-shaping the text itself -- and that needs the same font
the canvas draws with, which is `internal/render/fonts.InterRegular` and not
reachable from outside the module.

**Status: SUBMITTED, gogpu/ui#236** -- the first entry in this file to be sent.
An optional `TextCaretPositioner` interface rather than a method on `Canvas`,
which would have broken every implementor and which `docs/VERSIONING.md`
forbids; `internal/textmetrics` type-asserts it and falls back to the old
prefix measurement, so a canvas that does not implement it is unchanged. The
regression test uses only pre-existing API and therefore runs against
unmodified `main`, where it fails on 49 assertions -- which is the half that
matters, since a test that cannot fail proves nothing. Hit testing is fixed by
the same change: `RuneIndexFromX` compared a click against prefix widths the
renderer never used, so a click on the exact start of a character could resolve
to the character before it.
