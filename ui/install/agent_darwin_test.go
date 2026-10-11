package install

import (
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

// TestAgentRunsUnderLaunchd installs the real agent in this login session,
// waits for jitllmd to answer on AgentAddr, and removes it again. It changes
// the machine (a LaunchAgents file, a running daemon for its duration), so it
// runs only when JITLLM_LAUNCHD_JITLLMD names the jitllmd to start; it is the
// check run on a Mac before a release changes this file.
func TestAgentRunsUnderLaunchd(t *testing.T) {
	jitllmd := os.Getenv("JITLLM_LAUNCHD_JITLLMD")
	if jitllmd == "" {
		t.Skip("set JITLLM_LAUNCHD_JITLLMD to a jitllmd binary to run the agent under launchd")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	a := Agent{Home: home}
	if a.Installed() {
		t.Fatalf("%s is already installed; this test would replace it", a.Plist())
	}
	if c, err := net.DialTimeout("tcp", AgentAddr, time.Second); err == nil {
		c.Close()
		t.Fatalf("%s is already taken", AgentAddr)
	}
	models := t.TempDir()
	if err := a.Install(jitllmd, models); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := a.Remove(); err != nil {
			t.Error(err)
		}
		if a.Installed() {
			t.Error("Remove left the plist")
		}
		if out, err := exec.Command("launchctl", "print", domain()+"/"+AgentLabel).CombinedOutput(); err == nil {
			t.Errorf("the agent is still loaded after Remove:\n%s", out)
		}
	}()
	up := false
	for i := 0; i < 60 && !up; i++ {
		if c, err := net.DialTimeout("tcp", AgentAddr, time.Second); err == nil {
			c.Close()
			up = true
		} else {
			time.Sleep(500 * time.Millisecond)
		}
	}
	if !up {
		log, err := os.ReadFile(a.Log())
		t.Fatalf("jitllmd never listened on %s; log %s (%v):\n%s", AgentAddr, filepath.Base(a.Log()), err, log)
	}
}
