# Packaging

How a release becomes `jitllm.app` in a disk image, a Windows setup, a Homebrew
tap and a winget manifest. Everything here runs in
`.github/workflows/release.yml` on a `v*` tag; nothing in it publishes to a
package manager unless the secrets below are set, and winget is never
submitted from CI.

| artifact | built by | where | signed when |
|---|---|---|---|
| `jitllm_<v>_darwin_<arch>.dmg` | `ui/cmd/pack app`, then the `macos-app` job adds the CLIs, signs, and runs `macos/dmg.sh` | macOS runner (hdiutil, codesign, notarytool) | `MACOS_SIGN_P12` set: Developer ID; plus `MACOS_NOTARY_*`: notarized and stapled. Otherwise ad-hoc |
| `jitllm-setup_<v>_windows_<arch>.exe` | `windows/build.sh` (makensis on `windows/jitllm.nsi`) | Linux runner | `WINDOWS_SIGN_PFX` set: Authenticode on every .exe, the uninstaller and the setup. Otherwise unsigned |
| `Casks/jitllm.rb`, `Formula/jitllm.rb` | `homebrew/generate.sh <tag> checksums.txt <dir>` | the `packages` job, after the release is published | -- |
| `manifests/j/jitllm/jitllm/<v>/*.yaml` | `winget/generate.sh <tag> checksums.txt <dir>` | the `packages` job | -- |

The generated cask, formula and manifests are the `packaging` artifact of
every release run. They are not committed here: they name one release's
checksums.

## jitllm.app

`ui/cmd/pack app -bin jitllm-desktop -cli jitllm -cli jitllmd` writes the
bundle:

```
jitllm.app/Contents/Info.plist           bundle id io.github.samyfodil.jitllm, version from the tag
jitllm.app/Contents/MacOS/jitllm-desktop the desktop app
jitllm.app/Contents/Resources/jitllm.icns drawn from app.IconAt, the window icon
jitllm.app/Contents/Resources/bin/jitllm  the CLI
jitllm.app/Contents/Resources/bin/jitllmd the server
```

In a release, goreleaser builds the bundle around the desktop binary and the
`macos-app` job puts the CLI and the server in from their own darwin
archives. Nested code is signed before the bundle that seals it, each with the
hardened runtime and `.github/release/entitlements.plist` (the engine maps the
code it generates executable).

The app (`ui/install`, `ui/screen/cli.go`):

- on its first launch, when the programs are not already linked (a cask links
  them), asks to link `jitllm` and `jitllmd` into `/usr/local/bin`, through
  macOS's administrator prompt when that folder is root's, and into
  `~/.local/bin` if the prompt is declined;
- **Settings > Command line** links and unlinks them, and **Start the server
  at login** installs or removes the launchd user agent
  `~/Library/LaunchAgents/io.github.samyfodil.jitllm.jitllmd.plist`, which
  runs `jitllmd serve -addr 127.0.0.1:8080 -models <first model folder>`,
  restarting it if it crashes, logging to `~/Library/Logs/jitllm/jitllmd.log`.

There is no menu-bar icon. gogpu has one only in a separate module
(`gogpu/systray`), and the desktop module's toolkit versions are pinned
(`ui/README.md`, "Do not float the toolkit"); adding it is its own change.

The bundle is per architecture (`darwin_arm64`, `darwin_amd64`), as every
other release archive is; the cask picks one with `arch arm:/intel:`.

`macos/dmg.sh <app> <out.dmg>` stages the bundle beside a link to
`/Applications` and writes a compressed read-only image with `hdiutil`. It
runs on macOS only.

## The Windows setup

`windows/build.sh <version> <arch> <dist> <out.exe>` unpacks the three
programs from the release's zips, writes the icon (`ui/cmd/pack ico`) and runs
`makensis` on `windows/jitllm.nsi`. Needs NSIS 3.08 or later (`apt install
nsis`), `unzip` and Go. The setup:

- installs for the current user only (`RequestExecutionLevel user`) into
  `%LOCALAPPDATA%\Programs\jitllm`, the same directory `scripts/install.ps1`
  uses;
- adds that directory to the user `PATH` (a component, on by default);
- creates Start menu shortcuts for the app and the uninstaller;
- optionally starts `jitllmd serve -addr 127.0.0.1:8080 -models
  %LOCALAPPDATA%\jitllm\models` at login, from the `Run` key through a small
  script that starts it without a console window (`jitllmd` is a console
  program). The script is VBScript run by `wscript.exe`, which Windows ships
  today and has announced it will make optional and later remove; when it
  does, this entry needs a GUI-subsystem launcher instead;
- registers under Add/Remove Programs (per user), with an uninstaller that
  stops this install's processes, removes the programs, the shortcuts, the
  PATH entry and the login entry, and keeps models, chats and settings.

`/S` installs silently with the defaults; `/D=<dir>` (last) picks the
directory.

## Secrets and what the user provides

None of these exist yet. With none set, a release still builds every artifact,
ad-hoc signed on macOS and unsigned on Windows.

| secret | what | used by |
|---|---|---|
| `MACOS_SIGN_P12` | Developer ID Application certificate and key, `.p12`, base64 | goreleaser (bare binaries) and `macos-app` (bundle, dmg) |
| `MACOS_SIGN_PASSWORD` | the `.p12`'s password | same |
| `MACOS_NOTARY_ISSUER_ID` | App Store Connect API key's issuer id | notarization |
| `MACOS_NOTARY_KEY_ID` | that key's id | notarization |
| `MACOS_NOTARY_KEY` | that key's `.p8` contents | notarization |
| `WINDOWS_SIGN_PFX` | an Authenticode code-signing certificate and key, `.pfx`, base64 | `windows-setup` |
| `WINDOWS_SIGN_PASSWORD` | the `.pfx`'s password | `windows-setup` |
| `HOMEBREW_TAP_TOKEN` | a token that can push to the tap repository | `packages` |

and one optional repository variable, `HOMEBREW_TAP_REPO` (default
`jitllm/homebrew-tap`).

A Windows certificate whose key lives in a hardware token or a cloud signing
service cannot be exported as a `.pfx`; `windows/sign.sh` is the one place to
swap `osslsigncode` for that service's signer.

To publish, in order:

1. **Apple**: an Apple Developer Program membership, a *Developer ID
   Application* certificate exported as `.p12`, and an App Store Connect API
   key with the Developer role. Set the five `MACOS_*` secrets.
2. **Windows**: a code-signing certificate (OV or EV) usable as a `.pfx`, or a
   signing service wired into `windows/sign.sh`. Set the two `WINDOWS_*`
   secrets.
3. **Homebrew**: create the GitHub repository `jitllm/homebrew-tap` (the
   `homebrew-` prefix is what makes `brew install jitllm/tap/...` find it),
   and a fine-grained token with contents read/write on it alone, as
   `HOMEBREW_TAP_TOKEN`. The next release commits `Casks/jitllm.rb` and
   `Formula/jitllm.rb`; or push a release's `packaging` artifact by hand.
4. **winget**: from a release's `packaging` artifact, validate with
   `winget validate --manifest manifests/j/jitllm/jitllm/<v>` on Windows and
   open a pull request adding that directory to `microsoft/winget-pkgs`
   (`wingetcreate submit` does both). Later versions can use `wingetcreate
   update jitllm.jitllm`.
5. Then drop "once published" from the Homebrew and winget lines in
   `README.md` and `website/src/content/docs/docs/install.md`.

## First launch of an unsigned build

- **macOS, ad-hoc signed and not notarized.** Gatekeeper refuses a
  downloaded app it cannot check: *"jitllm" cannot be opened because Apple
  cannot check it for malicious software* (or, on recent macOS, *"jitllm" is
  damaged*). Control-click the app, **Open**, **Open**; or **System Settings >
  Privacy & Security > Open Anyway**; or remove the quarantine attribute with
  `xattr -dr com.apple.quarantine /Applications/jitllm.app`. The CLIs inside
  are covered once the app is allowed; linked elsewhere and run from a
  terminal, they carry no quarantine of their own.
- **Windows, unsigned.** SmartScreen shows *Windows protected your PC* for a
  downloaded unsigned setup: **More info**, **Run anyway**. A signed setup
  with an OV certificate still warns until the certificate builds reputation;
  an EV certificate does not.
- **Homebrew** quarantines what a cask installs, so an unsigned cask meets
  the same Gatekeeper prompt; and Homebrew's own audit requires casks in
  `homebrew/cask` to be signed and notarized, which is one reason this is a
  tap of its own.

## Checking a change here

- `ui`: `../scripts/cap 8G -- go test ./cmd/pack ./install -count=1`.
- The installer, on Linux: `windows/build.sh` against the release's zips,
  then under wine (32-bit wine is needed: the NSIS stub is x86) run the setup
  with `/S`, check `%LOCALAPPDATA%\Programs\jitllm`, the Uninstall key and the
  Start menu, run `uninstall.exe /S` and check they are gone. Under wine the
  PATH step does nothing (no PowerShell); the release's `smoke` job checks it
  on Windows.
- The bundle and the dmg, on a Mac: sign as `macos-app` does, run `dmg.sh`,
  mount the image, `codesign --verify --deep`, run
  `jitllm.app/Contents/Resources/bin/jitllm version`, `open` the app, and run
  `ui/install`'s test binary with `JITLLM_LAUNCHD_JITLLMD` set to the
  bundle's `jitllmd`, which starts the real agent under launchd and removes it.
- The winget manifests: `.github/workflows/winget.yml`, on a push to
  `installers` and in every release before it is published. On windows-2025
  and windows-11-arm it serves the setups from the runner
  (`JITLLM_INSTALLER_BASE`), generates the manifests with their real
  checksums, and runs `winget validate`, `winget install --manifest`,
  `jitllm version` from a new shell's PATH and `winget uninstall`. Where the
  image has no winget it is installed with `Repair-WinGetPackageManager`. The
  unsigned setup needs SmartScreen and the attachment manager's prompt turned
  off on the runner; a signed one would not. A manifest installed from disk is
  in no source, so the check finds the install by its Add/Remove Programs
  name, not by `jitllm.jitllm`.
- `shellcheck packaging/*/*.sh` and `actionlint .github/workflows/*.yml`.
