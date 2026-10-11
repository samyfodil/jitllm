//go:build !unix

// Command agenteval runs its tasks' agents and servers as Unix process groups;
// on other systems it says so.
package main

import (
	"fmt"
	"os"
)

func main() {
	fmt.Fprintln(os.Stderr, "agenteval: runs on Linux and macOS only (its tasks are shell workspaces and process groups)")
	os.Exit(1)
}
