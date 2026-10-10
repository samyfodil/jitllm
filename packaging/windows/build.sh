#!/bin/sh
# Builds jitllm's Windows setup from a release's archives, on Linux.
#
#   packaging/windows/build.sh 1.2.3 amd64 dist dist/jitllm-setup_1.2.3_windows_amd64.exe
#
# dist holds the release's jitllm, jitllmd and jitllm-desktop zips for
# windows/<arch> (goreleaser's names). Needs makensis (NSIS 3.08 or later),
# unzip, and Go for the icon (ui/cmd/pack ico). Run from the repository root.
#
# Signing is optional. With WINDOWS_SIGN_PFX (a path to a PKCS#12 file) and
# WINDOWS_SIGN_PASSWORD set, every .exe -- the three programs, the
# uninstaller and the setup itself -- is Authenticode-signed with
# osslsigncode and timestamped; without them nothing is signed, and Windows
# SmartScreen warns on first run (packaging/README.md).
set -eu

[ $# -eq 4 ] || { echo "usage: $0 <version> <amd64|arm64> <dist> <out.exe>" >&2; exit 2; }
version=${1#v}
arch=$2
dist=$3
out=$4
case $arch in amd64 | arm64) ;; *) echo "build: arch $arch is not amd64 or arm64" >&2; exit 2 ;; esac
command -v makensis >/dev/null || { echo "build: needs makensis (apt install nsis)" >&2; exit 1; }

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
src=$(mktemp -d)
trap 'rm -rf "$src"' EXIT

for prog in jitllm jitllmd jitllm-desktop; do
	zip="$dist/${prog}_${version}_windows_${arch}.zip"
	[ -f "$zip" ] || { echo "build: $zip is missing" >&2; exit 1; }
	unzip -q -o -j "$zip" "$prog.exe" -d "$src"
done
cp "$root/LICENSE" "$src/LICENSE.txt"
(cd "$root/ui" && go run ./cmd/pack ico -o "$src/jitllm.ico")

# The file version resource takes four numbers: a suffix (-rc1, -next) is
# dropped and the missing places are zeros.
version4=$(echo "${version%%-*}.0.0.0" | cut -d. -f1-4)

# The positional parameters carry makensis' SIGN define, or nothing. The
# quotes around %1 are makensis', for a path with spaces.
set --
if [ -n "${WINDOWS_SIGN_PFX:-}" ]; then
	command -v osslsigncode >/dev/null || { echo "build: signing needs osslsigncode" >&2; exit 1; }
	for exe in "$src"/*.exe; do
		"$here/sign.sh" "$exe"
	done
	set -- "-DSIGN=$here/sign.sh \"%1\""
fi

mkdir -p "$(dirname "$out")"
out=$(cd "$(dirname "$out")" && pwd)/$(basename "$out")
makensis -V2 -INPUTCHARSET UTF8 \
	-DVERSION="$version" -DVERSION4="$version4" -DARCH="$arch" \
	-DSRC="$src" -DOUTFILE="$out" "$@" \
	"$here/jitllm.nsi"
echo "$out"
