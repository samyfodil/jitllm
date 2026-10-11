//go:build unix

package main

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"
)

type config struct {
	agent       string
	root        string // the repository
	bin         string // jitllm and jitllmd
	runs        int
	timeout     time.Duration
	toolPattern string
	keep        bool
}

// Report is the JSON the runner writes.
type Report struct {
	Agent       string   `json:"agent"`
	Revision    string   `json:"revision"`
	Started     string   `json:"started"`
	Completed   int      `json:"completed"`
	RequiredRan int      `json:"required_checks_ran"`
	Results     []Result `json:"results"`
}

func (r *Report) summarise() {
	for _, x := range r.Results {
		if x.Completed {
			r.Completed++
		}
		if x.RequiredRan {
			r.RequiredRan++
		}
	}
}

// Result is one run of one task.
type Result struct {
	Task      string `json:"task"`
	Workspace string `json:"workspace"`
	// Completed is every success check passing.
	Completed bool    `json:"completed"`
	Checks    []Check `json:"checks"`
	// RequiredRan is every validation the task requires the agent to run
	// having been seen running (through the command shims' log).
	RequiredRan bool    `json:"required_checks_ran"`
	Required    []Check `json:"required"`
	WallSeconds float64 `json:"wall_seconds"`
	TimedOut    bool    `json:"timed_out"`
	AgentExit   int     `json:"agent_exit"`
	// ToolCalls is counted from the agent's stdout; null when its output
	// carries no recognisable tool calls.
	ToolCalls *int `json:"tool_calls"`
	// Commands is every jitllm, jitllmd, go and curl the agent ran, from the
	// shims; it is a lower bound on its tool use and is always known.
	Commands []string `json:"commands"`
	Error    string   `json:"error,omitempty"`
}

// Check is one machine-checked criterion and its outcome.
type Check struct {
	Name   string `json:"name"`
	Passed bool   `json:"passed"`
	Detail string `json:"detail,omitempty"`
}

// env is what a task's setup, agent and checks share.
type env struct {
	cfg  config
	dir  string // the workspace
	port string // the task's server port
	log  string // the shims' log
	path string // PATH with the shims first
	pid  int    // the server the setup started, if it started one
	pgid int    // the agent's process group, which what it backgrounded stays in
}

func runTask(cfg config, t task, dir string) Result {
	res := Result{Task: t.ID, Workspace: dir}
	fail := func(err error) Result {
		res.Error = err.Error()
		return res
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fail(err)
	}
	port, err := freePort()
	if err != nil {
		return fail(err)
	}
	e := &env{cfg: cfg, dir: dir, port: port, log: filepath.Join(dir, ".agenteval", "commands.log")}
	if err := e.shims(); err != nil {
		return fail(err)
	}
	defer e.reap()
	if t.Setup != nil {
		if err := t.Setup(e); err != nil {
			return fail(fmt.Errorf("setup: %w", err))
		}
	}
	prompt := e.expand(t.Prompt)
	pf := filepath.Join(dir, ".agenteval", "PROMPT.md")
	if err := os.WriteFile(pf, []byte(prompt), 0o644); err != nil {
		return fail(err)
	}

	agent := cfg.agent
	if agent == "reference" {
		agent = e.expand(t.Reference)
	}
	ctx, cancel := context.WithTimeout(context.Background(), cfg.timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, "bash", "-c", agent)
	cmd.Dir = dir
	cmd.Env = append(e.environ(), "AGENTEVAL_PROMPT="+prompt, "AGENTEVAL_PROMPT_FILE="+pf, "AGENTEVAL_TASK="+t.ID)
	cmd.Stdin = strings.NewReader(prompt)
	// Files, not pipes: a server the agent leaves running holds whatever
	// it inherited, and a pipe would keep the run waiting on it.
	outPath := filepath.Join(dir, ".agenteval", "agent.stdout")
	stdout, err := os.Create(outPath)
	if err != nil {
		return fail(err)
	}
	defer stdout.Close()
	stderr, err := os.Create(filepath.Join(dir, ".agenteval", "agent.stderr"))
	if err != nil {
		return fail(err)
	}
	defer stderr.Close()
	cmd.Stdout, cmd.Stderr = stdout, stderr
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error { return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL) }
	start := time.Now()
	err = cmd.Start()
	if err == nil {
		e.pgid = cmd.Process.Pid
		err = cmd.Wait()
	}
	res.WallSeconds = time.Since(start).Seconds()
	res.TimedOut = ctx.Err() != nil
	var ee *exec.ExitError
	switch {
	case err == nil:
	case errors.As(err, &ee):
		res.AgentExit = ee.ExitCode()
	default:
		return fail(fmt.Errorf("agent: %w", err))
	}
	transcript, err := os.ReadFile(outPath)
	if err != nil {
		return fail(err)
	}
	if n := strings.Count(string(transcript), cfg.toolPattern); n > 0 && cfg.toolPattern != "" {
		res.ToolCalls = &n
	}
	res.Commands = e.commands()

	res.Checks = t.Check(e)
	res.Completed = len(res.Checks) > 0
	for _, c := range res.Checks {
		res.Completed = res.Completed && c.Passed
	}
	res.RequiredRan = true
	for _, r := range t.Required {
		c := Check{Name: r.Name}
		for _, line := range res.Commands {
			if r.Match(line) {
				c.Passed, c.Detail = true, line
				break
			}
		}
		res.Required = append(res.Required, c)
		res.RequiredRan = res.RequiredRan && c.Passed
	}
	return res
}

// requirement is a validation the agent must be seen running.
type requirement struct {
	Name  string
	Match func(command string) bool
}

// ran is a requirement met by a logged command of tool whose arguments
// contain every one of words.
func ran(name, tool string, words ...string) requirement {
	return requirement{Name: name, Match: func(line string) bool {
		t, args, _ := strings.Cut(line, "\t")
		if t != tool {
			return false
		}
		for _, w := range words {
			if !strings.Contains(args, w) {
				return false
			}
		}
		return true
	}}
}

// expand fills a task's text with its workspace's values.
func (e *env) expand(s string) string {
	return strings.NewReplacer("{port}", e.port, "{dir}", e.dir).Replace(s)
}

// shims puts logging wrappers for jitllm, jitllmd, go and curl first on the
// agent's PATH: each appends its tool name and arguments to the log and runs
// the real one.
func (e *env) shims() error {
	sd := filepath.Join(e.dir, ".agenteval", "bin")
	if err := os.MkdirAll(sd, 0o755); err != nil {
		return err
	}
	goTool := filepath.Join(runtime.GOROOT(), "bin", "go")
	if _, err := os.Stat(goTool); err != nil {
		goTool, _ = exec.LookPath("go")
	}
	curl, _ := exec.LookPath("curl")
	real := map[string]string{
		"jitllm":  filepath.Join(e.cfg.bin, "jitllm"),
		"jitllmd": filepath.Join(e.cfg.bin, "jitllmd"),
		"go":      goTool,
		"curl":    curl,
	}
	for _, name := range []string{"python3", "node"} {
		if p, err := exec.LookPath(name); err == nil {
			real[name] = p
		}
	}
	for name, target := range real {
		if target == "" {
			continue
		}
		sh := fmt.Sprintf("#!/bin/sh\nprintf '%%s\\t%%s\\n' %s \"$*\" >> %s\nexec %s \"$@\"\n",
			shellQuote(name), shellQuote(e.log), shellQuote(target))
		if err := os.WriteFile(filepath.Join(sd, name), []byte(sh), 0o755); err != nil {
			return err
		}
	}
	e.path = sd + string(os.PathListSeparator) + os.Getenv("PATH")
	return nil
}

func (e *env) environ() []string {
	var out []string
	for _, kv := range os.Environ() {
		k, _, _ := strings.Cut(kv, "=")
		if k == "PATH" || k == "JITLLM_MODELS" || k == AgentEnv {
			continue
		}
		out = append(out, kv)
	}
	return append(out, "PATH="+e.path)
}

// commands is the shims' log, one "tool\targs" a line.
func (e *env) commands() []string {
	f, err := os.Open(e.log)
	if err != nil {
		return nil
	}
	defer f.Close()
	var out []string
	s := bufio.NewScanner(f)
	for s.Scan() {
		out = append(out, s.Text())
	}
	return out
}

// run runs a command for the setup or a check, with the shims' PATH but
// without logging into the agent's record.
func (e *env) run(dir string, name string, args ...string) (string, error) {
	// exec resolves a bare name on this process's PATH, not the command's.
	if name == "jitllm" || name == "jitllmd" {
		name = filepath.Join(e.cfg.bin, name)
	}
	cmd := exec.Command(name, args...)
	cmd.Dir = dir
	cmd.Env = append(e.environ(), "PATH="+e.cfg.bin+string(os.PathListSeparator)+os.Getenv("PATH"))
	out, err := cmd.CombinedOutput()
	return string(out), err
}

// start leaves a server running for the agent to work against; reap stops it.
func (e *env) start(args ...string) error {
	cmd := exec.Command(filepath.Join(e.cfg.bin, "jitllmd"), args...)
	cmd.Dir = e.dir
	cmd.Env = e.environ()
	f, err := os.Create(filepath.Join(e.dir, ".agenteval", "server.log"))
	if err != nil {
		return err
	}
	cmd.Stdout, cmd.Stderr = f, f
	if err := cmd.Start(); err != nil {
		return err
	}
	e.pid = cmd.Process.Pid
	go cmd.Wait()
	return e.waitHealthy()
}

func (e *env) waitHealthy() error {
	for range 120 {
		if _, err := httpGet(e.url("/healthz")); err == nil {
			return nil
		}
		time.Sleep(250 * time.Millisecond)
	}
	return fmt.Errorf("no server answered on 127.0.0.1:%s", e.port)
}

func (e *env) url(path string) string { return "http://127.0.0.1:" + e.port + path }

// reap stops every process still running in the workspace: the servers the
// setup started and any the agent left behind, however it detached them.
func (e *env) reap() {
	// Everywhere: the agent's process group (a server it backgrounded with
	// nohup or & stays in it) and the server the setup started.
	if e.pgid > 0 {
		syscall.Kill(-e.pgid, syscall.SIGKILL)
	}
	if e.pid > 0 {
		syscall.Kill(e.pid, syscall.SIGKILL)
	}
	// On Linux, also whatever left the group (setsid) but runs in the
	// workspace; other systems have no /proc to find it by.
	procs, err := os.ReadDir("/proc")
	if err != nil {
		return
	}
	for _, p := range procs {
		pid, err := strconv.Atoi(p.Name())
		if err != nil || pid == os.Getpid() {
			continue
		}
		cwd, err := os.Readlink(filepath.Join("/proc", p.Name(), "cwd"))
		if err != nil {
			continue
		}
		if cwd == e.dir || strings.HasPrefix(cwd, e.dir+string(filepath.Separator)) {
			syscall.Kill(pid, syscall.SIGKILL)
		}
	}
}

func freePort() (string, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return "", err
	}
	defer l.Close()
	return strconv.Itoa(l.Addr().(*net.TCPAddr).Port), nil
}

func shellQuote(s string) string { return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'" }

// repoRoot walks up from the working directory to the go.mod of the module.
func repoRoot() (string, error) {
	d, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for {
		b, err := os.ReadFile(filepath.Join(d, "go.mod"))
		if err == nil && bytes.HasPrefix(b, []byte("module github.com/jitllm/jitllm\n")) {
			return d, nil
		}
		up := filepath.Dir(d)
		if up == d {
			return "", errors.New("not inside the jitllm repository: run it from the checkout")
		}
		d = up
	}
}

func revision(root string) string {
	out, err := exec.Command("git", "-C", root, "rev-parse", "HEAD").Output()
	if err != nil {
		return "unknown"
	}
	return strings.TrimSpace(string(out))
}

func buildBinaries(root, bin string) error {
	goTool := filepath.Join(runtime.GOROOT(), "bin", "go")
	if _, err := os.Stat(goTool); err != nil {
		if goTool, err = exec.LookPath("go"); err != nil {
			return err
		}
	}
	for _, b := range []struct{ dir, pkg, out string }{
		{root, "./cmd/jitllm", "jitllm"},
		{filepath.Join(root, "server"), "./cmd/jitllmd", "jitllmd"},
	} {
		cmd := exec.Command(goTool, "build", "-o", filepath.Join(bin, b.out), b.pkg)
		cmd.Dir = b.dir
		cmd.Env = append(os.Environ(), "CGO_ENABLED=0")
		if out, err := cmd.CombinedOutput(); err != nil {
			return fmt.Errorf("building %s: %v\n%s", b.out, err, out)
		}
	}
	return nil
}
