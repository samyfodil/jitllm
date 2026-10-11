package main

import (
	"fmt"
	"html"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

// bundleID names the app to macOS: Launch Services, the Dock and the
// settings an app keeps under ~/Library are keyed on it.
const bundleID = "org.jitllm.app"

// minMacOS is the oldest macOS the Go toolchain this module builds with runs
// on.
const minMacOS = "12.0"

// bundle writes out as a macOS application around the darwin binary bin:
//
//	jitllm.app/Contents/Info.plist
//	jitllm.app/Contents/MacOS/<bin's name>
//	jitllm.app/Contents/Resources/jitllm.icns
//	jitllm.app/Contents/Resources/bin/<each of clis>
//
// clis are the command-line programs the app carries (jitllm, jitllmd): the
// app links them into PATH on request (ui/install), and the Homebrew cask
// links them from there. They sit under Resources rather than MacOS because
// MacOS is where Launch Services looks for the one executable it starts.
//
// On macOS the bundle is then signed ad-hoc under the hardened runtime with
// the JIT entitlements, the release's own (.github/release), since the engine
// maps the code it generates executable. Elsewhere there is no codesign: the
// binary keeps the ad-hoc signature Go's linker gives every darwin/arm64
// binary, which runs, and a release signs the bundle where it is notarized.
func bundle(bin string, clis []string, version, out, entitlements string) error {
	exe := filepath.Base(bin)
	if err := os.RemoveAll(out); err != nil {
		return err
	}
	macos := filepath.Join(out, "Contents", "MacOS")
	res := filepath.Join(out, "Contents", "Resources")
	cliDir := filepath.Join(res, "bin")
	for _, d := range []string{macos, res, cliDir} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			return err
		}
	}
	if err := copyExe(bin, filepath.Join(macos, exe)); err != nil {
		return err
	}
	var signFirst []string
	for _, c := range clis {
		dst := filepath.Join(cliDir, filepath.Base(c))
		if err := copyExe(c, dst); err != nil {
			return err
		}
		signFirst = append(signFirst, dst)
	}
	ic, err := icns()
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(res, "jitllm.icns"), ic, 0o644); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(out, "Contents", "Info.plist"), []byte(infoPlist(exe, version)), 0o644); err != nil {
		return err
	}
	if runtime.GOOS != "darwin" {
		fmt.Printf("%s: written, not signed (codesign runs on macOS only)\n", out)
		return nil
	}
	if entitlements == "" {
		entitlements, err = repoEntitlements()
		if err != nil {
			return err
		}
	}
	// Nested code is signed before the bundle that seals it: codesign refuses
	// to seal a bundle whose Resources hold an unsigned Mach-O.
	for _, p := range append(signFirst, out) {
		args := []string{"--force", "--sign", "-", "--options", "runtime", "--entitlements", entitlements}
		if id := signingID(filepath.Base(p)); id != "" {
			args = append(args, "--identifier", id)
		}
		cmd := exec.Command("codesign", append(args, p)...)
		cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
		if err := cmd.Run(); err != nil {
			return fmt.Errorf("codesign %s: %v", p, err)
		}
	}
	fmt.Printf("%s: written and signed ad-hoc\n", out)
	return nil
}

// signingID is the code-signing identifier of a program the release ships on
// macOS: the bundle's own executable takes bundleID from Info.plist, and each
// command-line program is named here so its signature does not carry a bare
// file name. "" leaves codesign's default.
func signingID(name string) string {
	switch name {
	case "jitllm":
		return "org.jitllm.cli"
	case "jitllmd":
		return "org.jitllm.service"
	case "jitllm-tui":
		return "org.jitllm.tui"
	}
	return ""
}

// copyExe copies the executable src to dst, mode 0755.
func copyExe(src, dst string) error {
	b, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	return os.WriteFile(dst, b, 0o755)
}

// repoEntitlements finds .github/release/entitlements.plist above the
// working directory, which is ui/ or the repository root.
func repoEntitlements() (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for {
		p := filepath.Join(dir, ".github", "release", "entitlements.plist")
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
		up := filepath.Dir(dir)
		if up == dir {
			return "", fmt.Errorf("no .github/release/entitlements.plist above the working directory; pass -entitlements")
		}
		dir = up
	}
}

// infoPlist is the bundle's Info.plist. NSHighResolutionCapable keeps the
// window at the display's own scale rather than magnified; the category files
// the app in Finder and the App Store's sense of developer tools.
// The version keys take numbers only, so a snapshot's or a release
// candidate's suffix is dropped there.
func infoPlist(exe, version string) string {
	e := html.EscapeString
	version, _, _ = strings.Cut(strings.TrimPrefix(version, "v"), "-")
	return `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleDevelopmentRegion</key>
	<string>en</string>
	<key>CFBundleDisplayName</key>
	<string>jitllm</string>
	<key>CFBundleName</key>
	<string>jitllm</string>
	<key>CFBundleExecutable</key>
	<string>` + e(exe) + `</string>
	<key>CFBundleIconFile</key>
	<string>jitllm</string>
	<key>CFBundleIdentifier</key>
	<string>` + bundleID + `</string>
	<key>CFBundleInfoDictionaryVersion</key>
	<string>6.0</string>
	<key>CFBundlePackageType</key>
	<string>APPL</string>
	<key>CFBundleShortVersionString</key>
	<string>` + e(version) + `</string>
	<key>CFBundleVersion</key>
	<string>` + e(version) + `</string>
	<key>LSMinimumSystemVersion</key>
	<string>` + minMacOS + `</string>
	<key>LSApplicationCategoryType</key>
	<string>public.app-category.developer-tools</string>
	<key>NSHighResolutionCapable</key>
	<true/>
	<key>NSPrincipalClass</key>
	<string>NSApplication</string>
</dict>
</plist>
`
}
