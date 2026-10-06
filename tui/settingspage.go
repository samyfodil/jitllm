package main

import (
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/charmbracelet/bubbles/key"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/huh"
	"github.com/samyfodil/jitllm/common/hardware"
	"github.com/samyfodil/jitllm/common/session"
)

// settingsPage is the window's settings as a huh form over the same file:
// a change here is a change there. The form edits a draft; submitting it
// (enter on the last field) applies and saves the draft, esc throws it away.
type settingsPage struct {
	env  *env
	form *huh.Form
	d    *draft
	w, h int
}

// draft is the settings as the form edits them: text for every number, so a
// half-typed value is the form's to validate rather than lost.
type draft struct {
	chat, kvCache     bool
	system, spec, ctx string
	knobs             []string
	seed, folders     string
}

func newSettingsPage(env *env) *settingsPage {
	p := &settingsPage{env: env}
	p.rebuild()
	return p
}

func theme() *huh.Theme {
	t := huh.ThemeCharm()
	t.Focused.Base = t.Focused.Base.BorderForeground(cBrand)
	t.Focused.Title = t.Focused.Title.Foreground(cBrand).Bold(true)
	t.Focused.Description = t.Focused.Description.Foreground(cDim)
	t.Focused.TextInput.Cursor = t.Focused.TextInput.Cursor.Foreground(cBrand)
	t.Focused.TextInput.Prompt = t.Focused.TextInput.Prompt.Foreground(cBrand)
	t.Focused.TextInput.Text = t.Focused.TextInput.Text.Foreground(cFg)
	t.Focused.FocusedButton = t.Focused.FocusedButton.Background(cBrand).Foreground(cBg)
	t.Focused.BlurredButton = t.Focused.BlurredButton.Background(cRaised).Foreground(cMuted)
	t.Focused.ErrorIndicator = t.Focused.ErrorIndicator.Foreground(cRed)
	t.Focused.ErrorMessage = t.Focused.ErrorMessage.Foreground(cRed)
	t.Blurred = t.Focused
	t.Blurred.Base = t.Blurred.Base.BorderForeground(cLine)
	t.Blurred.Title = t.Blurred.Title.Foreground(cMuted).Bold(false)
	t.Blurred.TextInput.Text = t.Blurred.TextInput.Text.Foreground(cMuted)
	t.Blurred.FocusedButton = t.Blurred.FocusedButton.Background(cRaised).Foreground(cMuted)
	t.Group.Title = t.Group.Title.Foreground(cBrand).Bold(true)
	return t
}

// rebuild starts a fresh form over the settings as they are now.
func (p *settingsPage) rebuild() {
	cfg := p.env.cfg
	d := &draft{
		chat: cfg.Chat, kvCache: !cfg.NoKVCache,
		system: cfg.System, spec: cfg.DeviceSpec, ctx: strconv.Itoa(cfg.MaxSeq),
		seed: strconv.FormatInt(cfg.Sampling.Seed, 10), folders: strings.Join(cfg.ModelDirs, "\n"),
	}
	knobs := make([]huh.Field, len(session.SamplingKnobs))
	d.knobs = make([]string, len(session.SamplingKnobs))
	for i, k := range session.SamplingKnobs {
		k := k
		d.knobs[i] = strconv.FormatFloat(float64(k.Get(cfg.Sampling)), 'f', -1, 32)
		knobs[i] = huh.NewInput().Title(k.Name).Value(&d.knobs[i]).
			Description(fmt.Sprintf("%g to %g", k.Min, k.Max)).
			Validate(func(s string) error {
				v, err := strconv.ParseFloat(strings.TrimSpace(s), 32)
				if err != nil || float32(v) < k.Min || float32(v) > k.Max {
					return fmt.Errorf("a number from %g to %g", k.Min, k.Max)
				}
				return nil
			})
	}
	knobs = append(knobs, huh.NewInput().Title("Seed").Description("0 is random").Value(&d.seed).
		Validate(func(s string) error {
			if _, err := strconv.ParseInt(strings.TrimSpace(s), 10, 64); err != nil {
				return errors.New("a whole number")
			}
			return nil
		}))

	p.d = d
	p.form = huh.NewForm(
		huh.NewGroup(
			huh.NewConfirm().Title("Chat template").Description("off is a raw completion of the prompt, as the benchmarks run").
				Affirmative("on").Negative("off").Value(&d.chat),
			huh.NewText().Title("System prompt").Lines(3).Value(&d.system),
			huh.NewConfirm().Title("Prompt cache on disk").Description("a prompt seen before resumes from its cached keys and values").
				Affirmative("on").Negative("off").Value(&d.kvCache),
		).Title("Chat"),
		huh.NewGroup(
			huh.NewInput().Title("Run models on").Description("auto, all, cpu, gpu, gpu:0, cuda:0, vulkan:1, metal · the next load").
				Value(&d.spec).Validate(func(s string) error {
				if strings.TrimSpace(s) == "" {
					return nil
				}
				_, err := hardware.ValidateSpec(strings.TrimSpace(s))
				return err
			}),
			huh.NewInput().Title("Context").Description("tokens · the next load").Value(&d.ctx).
				Validate(func(s string) error {
					if n, err := strconv.Atoi(strings.TrimSpace(s)); err != nil || n < 256 {
						return errors.New("a whole number of tokens, 256 or more")
					}
					return nil
				}),
			huh.NewText().Title("Model folders").Description("one per line; the first is where conversions go").
				Lines(4).Value(&d.folders).Validate(func(s string) error {
				if strings.TrimSpace(s) == "" {
					return errors.New("at least one folder")
				}
				return nil
			}),
		).Title("Engine"),
		huh.NewGroup(knobs...).Title("Sampling"),
	).WithTheme(theme()).WithShowHelp(false).WithLayout(huh.LayoutColumns(3))
	p.form.SubmitCmd = func() tea.Msg { return settingsDoneMsg{} }
	p.form.CancelCmd = func() tea.Msg { return settingsCancelMsg{} }
	p.layout()
}

type (
	settingsDoneMsg   struct{}
	settingsCancelMsg struct{}
)

// apply writes the draft into the settings and the engine's state, and saves.
func (p *settingsPage) apply() tea.Cmd {
	cfg, st, d := p.env.cfg, p.env.f.st, p.d
	cfg.Chat = d.chat
	st.Chat.Set(d.chat)
	cfg.NoKVCache = !d.kvCache
	st.KVCache.Set(d.kvCache)
	cfg.System = strings.TrimSpace(d.system)
	cfg.DeviceSpec = strings.TrimSpace(d.spec)
	if cfg.DeviceSpec == "" {
		cfg.DeviceSpec = "auto"
	}
	st.DeviceSpec.Set(cfg.DeviceSpec)
	if n, err := strconv.Atoi(strings.TrimSpace(d.ctx)); err == nil {
		cfg.MaxSeq = n
		st.MaxSeq.Set(n)
	}
	for i, k := range session.SamplingKnobs {
		if v, err := strconv.ParseFloat(strings.TrimSpace(d.knobs[i]), 32); err == nil {
			k.Set(&cfg.Sampling, float32(v))
		}
	}
	if v, err := strconv.ParseInt(strings.TrimSpace(d.seed), 10, 64); err == nil {
		cfg.Sampling.Seed = v
	}
	var dirs []string
	for _, l := range strings.Split(d.folders, "\n") {
		if l = strings.TrimSpace(l); l != "" {
			dirs = append(dirs, l)
		}
	}
	folders := strings.Join(dirs, "\n") != strings.Join(cfg.ModelDirs, "\n")
	cfg.ModelDirs = dirs
	p.env.dirs = dirs
	cmds := []tea.Cmd{saveConfig(p.env), notify("settings saved")}
	if folders {
		cmds = append(cmds, rescan())
	}
	return tea.Batch(cmds...)
}

func (p *settingsPage) Help() []key.Binding {
	return []key.Binding{bind("tab", "next field"), bind("shift+tab", "previous"), bind("enter", "next · save on the last"), bind("esc", "discard")}
}

func (p *settingsPage) Focus() tea.Cmd { return p.form.Init() }
func (p *settingsPage) Blur()          {}

// Capturing: the form takes tab and every printable key.
func (p *settingsPage) Capturing() bool { return true }
func (p *settingsPage) Animating() bool { return false }

func (p *settingsPage) SetSize(w, h int) {
	p.w, p.h = w, h
	p.layout()
}

func (p *settingsPage) layout() {
	iw, ih := inner(p.w, p.h)
	// The columns' borders and gaps are the form's own; it is given a little
	// less than the panel so they land inside it.
	p.form.WithWidth(iw - 4).WithHeight(ih - 1)
	if iw < 120 {
		p.form.WithLayout(huh.LayoutDefault)
	} else {
		p.form.WithLayout(huh.LayoutColumns(3))
	}
}

func (p *settingsPage) Update(msg tea.Msg) tea.Cmd {
	switch msg.(type) {
	case settingsDoneMsg:
		cmd := p.apply()
		p.rebuild()
		return tea.Batch(cmd, p.form.Init())
	case settingsCancelMsg:
		p.rebuild()
		return tea.Batch(notify("changes discarded"), p.form.Init())
	}
	m, cmd := p.form.Update(msg)
	if f, ok := m.(*huh.Form); ok {
		p.form = f
	}
	return cmd
}

func (p *settingsPage) View() string {
	foot := sDim.Render("saved to " + p.env.cfg.Path() + " · shared with the window")
	iw, _ := inner(p.w, p.h)
	return panel("settings", p.form.View()+"\n"+truncate(foot, iw), p.w, p.h, cBrand)
}
