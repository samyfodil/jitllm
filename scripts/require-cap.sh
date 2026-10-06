#!/usr/bin/env bash
# PreToolUse hook: refuse any build/test/benchmark command not wrapped in
# scripts/cap.
#
# AGENTS.md RULE 3 says every heavy job runs capped. Saying so in a document
# failed twice in one day, both times killing the user's GNOME session: once
# via uncapped subagents, once via a bare `go test` that "only ran two tiny
# tests" and still had to compile the package first. A rule that depends on
# remembering it is not a rule.
#
# Reads the hook JSON on stdin. Exit 0 allows, exit 2 blocks and shows stderr.
# Anything unexpected allows: a broken hook must never wedge the session.
exec python3 "$(dirname "$0")/require-cap.py"
