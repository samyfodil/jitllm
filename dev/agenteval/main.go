//go:build unix

// Command agenteval measures whether an agent can use and change jitllm: it
// sets up each task in a fresh workspace, hands the task's prompt to the agent
// command it is given, then checks the result by machine and writes a JSON
// report. dev/agenteval/README.md is how to run it.
//
//	go run ./dev/agenteval -agent 'my-agent --prompt-file "$AGENTEVAL_PROMPT_FILE"'
//	go run ./dev/agenteval -agent reference      # the built-in solutions: the checks can pass
//	go run ./dev/agenteval -list
//
// It calls no model API and no remote service: the agent is whatever command
// the caller names, and with none named it does nothing and says so.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// AgentEnv is the variable that names the agent command when -agent does not.
const AgentEnv = "JITLLM_AGENTEVAL_CMD"

func main() {
	agent := flag.String("agent", os.Getenv(AgentEnv), "the agent command, run with bash -c in the task's workspace; the prompt is on stdin, in $AGENTEVAL_PROMPT and in the file $AGENTEVAL_PROMPT_FILE. \"reference\" runs the built-in solutions (default $"+AgentEnv+")")
	only := flag.String("tasks", "", "comma-separated task ids to run (default: all)")
	runs := flag.Int("runs", 1, "times to run each task")
	out := flag.String("out", "agenteval-report.json", "where the JSON report goes")
	work := flag.String("work", "", "directory the workspaces are made in (default: a fresh temporary one, removed afterwards)")
	keep := flag.Bool("keep", false, "keep the workspaces after the run")
	bin := flag.String("bin", "", "a directory holding jitllm and jitllmd to use (default: build them from this tree)")
	timeout := flag.Duration("timeout", 20*time.Minute, "how long the agent gets for one task")
	toolPattern := flag.String("tool-call-pattern", `"type":"tool_use"`, "a string counted in the agent's stdout as one tool call each; a run with none is reported as unknown")
	list := flag.Bool("list", false, "list the tasks and exit")
	flag.Parse()

	if *list {
		for _, t := range tasks() {
			fmt.Printf("%-18s %s\n", t.ID, t.Title)
		}
		return
	}
	if strings.TrimSpace(*agent) == "" {
		fmt.Fprintf(os.Stderr, `agenteval: no agent configured, so nothing was run.
Name the agent command with -agent or %s; it runs in each task's workspace with
the prompt on stdin and in $AGENTEVAL_PROMPT_FILE. -agent reference runs the
built-in solutions, which checks the tasks themselves. See dev/agenteval/README.md.
`, AgentEnv)
		return
	}
	cfg := config{
		agent:       *agent,
		runs:        *runs,
		timeout:     *timeout,
		toolPattern: *toolPattern,
		keep:        *keep,
	}
	if err := run(cfg, *only, *work, *bin, *out); err != nil {
		fmt.Fprintf(os.Stderr, "agenteval: %v\n", err)
		os.Exit(1)
	}
}

func run(cfg config, only, work, bin, out string) error {
	root, err := repoRoot()
	if err != nil {
		return err
	}
	cfg.root = root
	selected, err := selectTasks(only)
	if err != nil {
		return err
	}
	if work == "" {
		if work, err = os.MkdirTemp("", "agenteval-"); err != nil {
			return err
		}
		if !cfg.keep {
			defer os.RemoveAll(work)
		}
	}
	if bin == "" {
		bin = filepath.Join(work, "bin")
		fmt.Fprintf(os.Stderr, "agenteval: building jitllm and jitllmd into %s\n", bin)
		if err := buildBinaries(root, bin); err != nil {
			return err
		}
	}
	if cfg.bin, err = filepath.Abs(bin); err != nil {
		return err
	}
	rep := Report{Agent: cfg.agent, Started: time.Now().UTC().Format(time.RFC3339), Revision: revision(root)}
	for _, t := range selected {
		for i := range cfg.runs {
			dir := filepath.Join(work, fmt.Sprintf("%s-%d", t.ID, i+1))
			r := runTask(cfg, t, dir)
			fmt.Fprintf(os.Stderr, "agenteval: %-18s run %d: completed=%v required=%v %.1fs\n",
				t.ID, i+1, r.Completed, r.RequiredRan, r.WallSeconds)
			rep.Results = append(rep.Results, r)
		}
	}
	rep.summarise()
	b, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(out, append(b, '\n'), 0o644); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "agenteval: %d of %d run(s) completed, %d with every required check run; report in %s\n",
		rep.Completed, len(rep.Results), rep.RequiredRan, out)
	return nil
}

func selectTasks(only string) ([]task, error) {
	all := tasks()
	if only == "" {
		return all, nil
	}
	var out []task
	for _, id := range strings.Split(only, ",") {
		found := false
		for _, t := range all {
			if t.ID == strings.TrimSpace(id) {
				out = append(out, t)
				found = true
			}
		}
		if !found {
			return nil, fmt.Errorf("no task %q (-list lists them)", id)
		}
	}
	return out, nil
}
