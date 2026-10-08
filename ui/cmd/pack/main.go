// Command pack makes jitllm-ui a desktop application on the two systems that
// tell one from a command-line program by how it is packaged.
//
//	go run ./cmd/pack syso -arch amd64 -version 1.2.3 -o rsrc_windows_amd64.syso
//	go run ./cmd/pack check jitllm-ui.exe
//	go run ./cmd/pack app -bin jitllm-ui -version 1.2.3 -o dist-app/jitllm.app
//
// syso writes the Windows resource object (the icon and the version
// information) that the Go linker links into a binary built in the same
// directory; the binary itself is linked with -ldflags -H=windowsgui so
// Windows opens no console beside it. check refuses an .exe that is not a GUI
// program or carries no icon and version. app builds the macOS bundle around a
// darwin binary: Info.plist, the .icns, and on macOS an ad-hoc signature with
// the JIT entitlements.
//
// Every icon is drawn from app.IconAt, the window's own icon, so there is no
// image file to keep in step. Everything here is pure Go and runs on any host:
// no iconutil, no rsrc tool, no cgo.
package main

import (
	"flag"
	"fmt"
	"os"
)

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	var err error
	switch os.Args[1] {
	case "syso":
		err = runSyso(os.Args[2:])
	case "check":
		err = runCheck(os.Args[2:])
	case "app":
		err = runApp(os.Args[2:])
	default:
		usage()
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "pack:", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: pack syso|check|app [flags]")
	os.Exit(2)
}

func runSyso(args []string) error {
	fs := flag.NewFlagSet("syso", flag.ExitOnError)
	arch := fs.String("arch", "amd64", "target GOARCH: amd64 or arm64")
	version := fs.String("version", "0.0.0", "the version the resource states")
	out := fs.String("o", "", "output file (default rsrc_windows_<arch>.syso)")
	fs.Parse(args)
	if *out == "" {
		*out = "rsrc_windows_" + *arch + ".syso"
	}
	b, err := syso(*arch, *version)
	if err != nil {
		return err
	}
	return os.WriteFile(*out, b, 0o644)
}

func runCheck(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("check takes one .exe")
	}
	return checkExe(args[0])
}

func runApp(args []string) error {
	fs := flag.NewFlagSet("app", flag.ExitOnError)
	bin := fs.String("bin", "", "the darwin binary to bundle")
	version := fs.String("version", "0.0.0", "the version the bundle states")
	out := fs.String("o", "jitllm.app", "the bundle to write")
	ent := fs.String("entitlements", "", "entitlements for the signature (default: the repo's .github/release/entitlements.plist)")
	fs.Parse(args)
	if *bin == "" {
		return fmt.Errorf("app needs -bin")
	}
	return bundle(*bin, *version, *out, *ent)
}
