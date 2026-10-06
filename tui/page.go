package main

import (
	"time"

	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/spinner"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/samyfodil/jitllm/common/config"
	"github.com/samyfodil/jitllm/common/engine"
)

// screen is which page fills the body.
type screen int

const (
	scrChat screen = iota
	scrModels
	scrDiscover
	scrEngine
	scrSettings
)

var screenNames = []string{"Chat", "Models", "Discover", "Engine", "Settings"}

// page is one screen. The root sizes it, focuses it, routes it every message
// and draws its View; a page asks the root for anything beyond itself with a
// message (showMsg, notifyMsg, rescanMsg).
type page interface {
	Update(msg tea.Msg) tea.Cmd
	View() string
	SetSize(w, h int)
	Focus() tea.Cmd
	Blur()
	// Help is the page's keys, for the footer.
	Help() []key.Binding
	// Capturing is true while the page takes text or has something esc
	// undoes (a filter), so esc goes to the page instead of back to the chat.
	Capturing() bool
	// Animating is true while the page draws something that moves, which is
	// when the root runs its frame clock.
	Animating() bool
}

// env is what every page shares: the engine and its state, the settings,
// the frame clock and the spinner.
type env struct {
	f    *front
	e    *engine.Engine
	cfg  *config.Config
	dirs []string
	keys *globalKeys
	jobs *jobs
	tele *telemetry
	// frame counts animation ticks; spin is the one spinner every page shows.
	frame int
	spin  spinner.Model
}

// Messages a page sends the root.
type (
	showMsg   struct{ s screen }
	notifyMsg struct{ text string }
	// rescanMsg asks every page that lists the model folders to list them
	// again, after a job wrote into one or a folder was added.
	rescanMsg struct{}
	// loadMsg asks the root to open path (or switch to it) and show the chat.
	loadMsg struct{ path string }
)

func show(s screen) tea.Cmd      { return func() tea.Msg { return showMsg{s} } }
func notify(text string) tea.Cmd { return func() tea.Msg { return notifyMsg{text} } }
func rescan() tea.Cmd            { return func() tea.Msg { return rescanMsg{} } }
func load(path string) tea.Cmd   { return func() tea.Msg { return loadMsg{path} } }

// fps is the frame clock's rate while something animates.
const fps = 12

type animMsg struct{}

func frameTick() tea.Cmd {
	return tea.Tick(time.Second/fps, func(time.Time) tea.Msg { return animMsg{} })
}

// globalKeys are the bindings the root takes before any page.
type globalKeys struct {
	Quit, Palette, Back key.Binding
	Screens             []key.Binding
}

func bind(help, desc string, ks ...string) key.Binding {
	if len(ks) == 0 {
		ks = []string{help}
	}
	return key.NewBinding(key.WithKeys(ks...), key.WithHelp(help, desc))
}

func newGlobalKeys() *globalKeys {
	return &globalKeys{
		Quit:    bind("ctrl+c", "quit", "ctrl+c"),
		Palette: bind("ctrl+k", "commands", "ctrl+k", "ctrl+p"),
		Back:    bind("esc", "back to chat", "esc"),
		Screens: []key.Binding{
			bind("f1", "chat", "f1", "alt+1"),
			bind("f2", "models", "f2", "alt+2", "ctrl+o"),
			bind("f3", "discover", "f3", "alt+3"),
			bind("f4", "engine", "f4", "alt+4", "ctrl+e"),
			bind("f5", "settings", "f5", "alt+5"),
		},
	}
}
