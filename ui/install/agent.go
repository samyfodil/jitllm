package install

import (
	"errors"
	"fmt"
	"html"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

// AgentLabel names the launchd user agent that starts jitllmd at login. It
// is the app's bundle identifier (ui/cmd/pack) with the program's name.
const AgentLabel = "io.github.samyfodil.jitllm.jitllmd"

// AgentAddr is where the agent's jitllmd listens: loopback only, as the app's
// own API defaults to, and on the same port, so the two are not both on --
// the settings screen says so beside the switch.
const AgentAddr = "127.0.0.1:8080"

// Agent is the launchd user agent for jitllmd, rooted at a home directory.
type Agent struct {
	Home string
}

// Plist is where the agent's property list is written.
func (a Agent) Plist() string {
	return filepath.Join(a.Home, "Library", "LaunchAgents", AgentLabel+".plist")
}

// Log is where jitllmd's output goes while it runs under launchd.
func (a Agent) Log() string {
	return filepath.Join(a.Home, "Library", "Logs", "jitllm", "jitllmd.log")
}

// Installed reports whether the agent's property list is in place.
func (a Agent) Installed() bool {
	_, err := os.Stat(a.Plist())
	return err == nil
}

// PlistFor is the property list that runs jitllmd serve on AgentAddr over
// the models directory, at login and after a crash (KeepAlive on an
// unsuccessful exit only, so `jitllmd` exiting cleanly is not restarted).
func (a Agent) PlistFor(jitllmd, models string) string {
	e := html.EscapeString
	args := []string{jitllmd, "serve", "-addr", AgentAddr, "-models", models}
	var argv strings.Builder
	for _, s := range args {
		argv.WriteString("\t\t<string>" + e(s) + "</string>\n")
	}
	return `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>Label</key>
	<string>` + AgentLabel + `</string>
	<key>ProgramArguments</key>
	<array>
` + argv.String() + `	</array>
	<key>RunAtLoad</key>
	<true/>
	<key>KeepAlive</key>
	<dict>
		<key>SuccessfulExit</key>
		<false/>
	</dict>
	<key>ProcessType</key>
	<string>Background</string>
	<key>StandardOutPath</key>
	<string>` + e(a.Log()) + `</string>
	<key>StandardErrorPath</key>
	<string>` + e(a.Log()) + `</string>
</dict>
</plist>
`
}

// Install writes the agent and starts it in the person's GUI session
// (launchctl bootstrap gui/<uid>). An agent already there is replaced, so a
// moved app or a new models directory takes effect.
func (a Agent) Install(jitllmd, models string) error {
	if err := os.MkdirAll(filepath.Dir(a.Plist()), 0o755); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(a.Log()), 0o755); err != nil {
		return err
	}
	if a.Installed() {
		// Not loaded is not a failure: the file can outlive a session.
		launchctl("bootout", domain()+"/"+AgentLabel)
	}
	if err := os.WriteFile(a.Plist(), []byte(a.PlistFor(jitllmd, models)), 0o644); err != nil {
		return err
	}
	return launchctl("bootstrap", domain(), a.Plist())
}

// Remove stops the agent and deletes its property list.
func (a Agent) Remove() error {
	if !a.Installed() {
		return nil
	}
	// As in Install: an agent that is not loaded has nothing to stop.
	launchctl("bootout", domain()+"/"+AgentLabel)
	if err := os.Remove(a.Plist()); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return nil
}

func domain() string { return "gui/" + strconv.Itoa(os.Getuid()) }

func launchctl(args ...string) error {
	out, err := exec.Command("launchctl", args...).CombinedOutput()
	if err != nil {
		return fmt.Errorf("launchctl %s: %v: %s", args[0], err, strings.TrimSpace(string(out)))
	}
	return nil
}
