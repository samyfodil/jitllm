package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/samyfodil/jitllm/engine/model"
	"github.com/samyfodil/jitllm/format/jlm"
)

// decideCmd is `jitllm decide <model.jlm> <request.json|->`: a decision
// model's answers to a TypeSafe /v1/systemone request body, printed as the
// response body the server returns for it (docs/design/decision-models.md).
//
// With -lines, every line of stdin is one request and gets one response line,
// which is how a stream of decisions is answered without paying the load per
// request.
func decideCmd(args []string) error {
	fs := flag.NewFlagSet("decide", flag.ExitOnError)
	lines := fs.Bool("lines", false, "answer every line of stdin as one request")
	maxSeq := fs.Int("max-seq", 8192, "the longest prompt one question may render to, in tokens")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if err := misplacedFlag(fs); err != nil {
		return err
	}
	if fs.NArg() != 2 && !(*lines && fs.NArg() == 1) {
		return fmt.Errorf("usage: jitllm decide [-lines] [-max-seq N] <model.jlm> <request.json|->")
	}
	t0 := time.Now()
	m, err := model.Open(fs.Arg(0), loadOpts(nil, 0)...)
	if err != nil {
		return err
	}
	defer m.Close()
	if m.Decision() == jlm.DecisionNone {
		return fmt.Errorf("%s is not a decision model: its container states no decision readout", fs.Arg(0))
	}
	d, err := m.NewDecider(*maxSeq)
	if err != nil {
		return err
	}
	defer d.Close()
	fmt.Fprintf(os.Stderr, "%s: %s, %v readout, loaded in %v\n",
		fs.Arg(0), m.Cfg.Arch, m.Decision(), time.Since(t0).Round(time.Millisecond))
	out := bufio.NewWriter(os.Stdout)
	defer out.Flush()
	one := func(body []byte) error {
		_, state, qs, err := model.ParseDecisionRequest(body)
		if err != nil {
			return err
		}
		t := time.Now()
		ans, err := d.Decide(state, qs)
		if err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "%d questions, %d tokens in %v\n", len(qs), d.InputTokens(), time.Since(t).Round(time.Microsecond))
		if _, err := out.Write(model.DecisionResponse(fs.Arg(0), qs, ans, d.InputTokens())); err != nil {
			return err
		}
		return out.WriteByte('\n')
	}
	if !*lines {
		var body []byte
		if fs.Arg(1) == "-" {
			body, err = io.ReadAll(os.Stdin)
		} else {
			body, err = os.ReadFile(fs.Arg(1))
		}
		if err != nil {
			return err
		}
		return one(body)
	}
	sc := bufio.NewScanner(os.Stdin)
	sc.Buffer(make([]byte, 1<<20), 1<<26)
	for sc.Scan() {
		if err := one(sc.Bytes()); err != nil {
			return err
		}
		if err := out.Flush(); err != nil {
			return err
		}
	}
	return sc.Err()
}
