package main

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"
)

// TestEveryVerbRefusesAFlagAfterItsFirstPositional. Go's flag package stops
// at the first positional, so `jitllmd run "hi" -n 50` would silently ignore
// the -n; every verb with a positional has the same hole, so all are walked.
// The guard fires inside parse, before any network call.
//
// VIOLATION SIGNATURE. Delete the `misplacedFlag` call from parse and this
// fails with
//
//	run: [m-1 -n 50] was accepted; the -n is silently swallowed into the
//	positional arguments
func TestEveryVerbRefusesAFlagAfterItsFirstPositional(t *testing.T) {
	// A dead port: the guard must fire before anything is dialled, so these
	// never reach a socket. If one ever does, the error will say "unavailable"
	// instead and the assertion below catches it.
	const dead = "127.0.0.1:1"

	cases := []struct {
		verb string
		fn   func(context.Context, cli, []string) error
		args []string
	}{
		{"run", cmdRun, []string{"-addr", dead, "-model", "m", "hello", "-n", "50"}},
		{"run/attached value", cmdRun, []string{"-addr", dead, "-model", "m", "hello", "-n=50"}},
		{"run/double dash", cmdRun, []string{"-addr", dead, "-model", "m", "hello", "--chat"}},
		{"models", cmdModels, []string{"-addr", dead, "m-1", "-loaded"}},
		{"devices", cmdDevices, []string{"-addr", dead, "cuda:0", "-addr", dead}},
		{"sessions", cmdSessions, []string{"-addr", dead, "s-1", "-model", "m"}},
		{"place", cmdPlace, []string{"-addr", dead, "x", "-session", "s"}},
		{"stats", cmdStats, []string{"-addr", dead, "x", "-watch"}},
	}
	for _, c := range cases {
		t.Run(c.verb, func(t *testing.T) {
			var o, e syncBuffer
			err := runVerb(t, c.fn, cli{out: &o, err: &e}, c.args, 10*time.Second)
			if err == nil {
				t.Fatalf("%v was accepted; the trailing flag is silently swallowed into the "+
					"positional arguments", c.args)
			}
			if !strings.Contains(err.Error(), "came after a positional argument") {
				t.Fatalf("%v failed for the wrong reason: %v", c.args, err)
			}
			// It must exit 2, like an unknown flag: a script that retries
			// on 1 would retry a command line that can never work.
			var ue usageErr
			if !errors.As(err, &ue) {
				t.Fatalf("%v is not reported as a usage error, so it exits 1 rather than 2: %v",
					c.args, err)
			}
			t.Logf("%s", strings.SplitN(err.Error(), "\n", 2)[0])
		})
	}
}

// TestAPromptThatMerelyLooksLikeAFlagIsNotRefused: the guard matches the
// flags the set defines, not a leading dash, so "-42 degrees is" still runs.
func TestAPromptThatMerelyLooksLikeAFlagIsNotRefused(t *testing.T) {
	allowed := [][]string{
		{"-addr", "127.0.0.1:1", "-model", "m", "-42 degrees is"},
		{"-addr", "127.0.0.1:1", "-model", "m", "hi", "-notaflag", "there"},
		{"-addr", "127.0.0.1:1", "-model", "m", "--", "a literal dash dash"},
	}
	for _, args := range allowed {
		var o, e syncBuffer
		err := runVerb(t, cmdRun, cli{out: &o, err: &e}, args, 10*time.Second)
		if err != nil && strings.Contains(err.Error(), "came after a positional argument") {
			t.Errorf("%v was refused as a misplaced flag: %v", args, err)
		}
	}
}

// TestParseReportsAnUnknownFlagOnceRatherThanTwice. flag already writes the
// reason and the usage to the verb's stderr, so the error that comes back is a
// marker for the exit status and not a second copy of the message.
func TestParseReportsAnUnknownFlagOnceRatherThanTwice(t *testing.T) {
	var o, e syncBuffer
	err := runVerb(t, cmdRun, cli{out: &o, err: &e},
		[]string{"-nosuchflag", "hi"}, 10*time.Second)
	if err == nil {
		t.Fatal("an unknown flag was accepted")
	}
	if err != errHandled {
		t.Fatalf("an unknown flag returned %v, want the already-explained marker", err)
	}
	if !strings.Contains(e.String(), "nosuchflag") {
		t.Fatalf("the verb's stderr does not name the unknown flag:\n%s", e.String())
	}
}
