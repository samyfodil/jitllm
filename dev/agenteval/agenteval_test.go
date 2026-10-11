//go:build unix

package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestTasksDiscriminate runs every task twice: with its reference solution,
// which must complete it with every required check seen running, and with an
// agent that does nothing, which must complete none and run none. A task
// whose checks pass on an empty workspace measures nothing; one whose
// reference fails cannot be completed.
func TestTasksDiscriminate(t *testing.T) {
	work := t.TempDir()
	bin := filepath.Join(work, "bin")
	root, err := repoRoot()
	if err != nil {
		t.Fatal(err)
	}
	if err := buildBinaries(root, bin); err != nil {
		t.Fatal(err)
	}
	for _, arm := range []struct {
		agent string
		want  bool
	}{{"reference", true}, {"true", false}} {
		out := filepath.Join(work, arm.agent+".json")
		cfg := config{agent: arm.agent, runs: 1, timeout: 5 * time.Minute, toolPattern: `"type":"tool_use"`}
		if err := run(cfg, "", filepath.Join(work, arm.agent), bin, out); err != nil {
			t.Fatalf("-agent %s: %v", arm.agent, err)
		}
		b, err := os.ReadFile(out)
		if err != nil {
			t.Fatal(err)
		}
		var rep Report
		if err := json.Unmarshal(b, &rep); err != nil {
			t.Fatal(err)
		}
		if len(rep.Results) != len(tasks()) {
			t.Fatalf("-agent %s: %d results for %d tasks", arm.agent, len(rep.Results), len(tasks()))
		}
		for _, r := range rep.Results {
			if r.Error != "" {
				t.Errorf("-agent %s, %s: %s", arm.agent, r.Task, r.Error)
				continue
			}
			if r.Completed != arm.want || r.RequiredRan != arm.want {
				t.Errorf("-agent %s, %s: completed %v, required checks ran %v; want %v\nchecks %+v\nrequired %+v",
					arm.agent, r.Task, r.Completed, r.RequiredRan, arm.want, r.Checks, r.Required)
			}
			// Every check fails on the empty workspace, not only one.
			if !arm.want {
				for _, c := range r.Checks {
					if c.Passed && c.Name != "neither file was modified" && c.Name != "the server was not restarted" &&
						c.Name != "the server still answers correctly" && c.Name != "the server module builds and vets" {
						t.Errorf("%s: %q passed with an agent that did nothing", r.Task, c.Name)
					}
				}
			}
		}
	}
}
