#!/bin/sh
# Builds the disk image a Mac user drags jitllm.app out of.
#
#   packaging/macos/dmg.sh jitllm.app jitllm_1.2.3_darwin_arm64.dmg
#
# The image holds the bundle and a link to /Applications beside it, which is
# the whole install. It is compressed (UDZO) and read-only. macOS only:
# hdiutil has no counterpart elsewhere that writes a dmg Finder trusts.
#
# Signing and notarizing the image are the release workflow's steps
# (.github/workflows/release.yml, macos-app), run after this one, because
# they need the Developer ID that only the workflow holds.
set -eu

[ $# -eq 2 ] || { echo "usage: $0 <jitllm.app> <out.dmg>" >&2; exit 2; }
app=$1
out=$2
[ -d "$app/Contents/MacOS" ] || { echo "dmg: $app is not an app bundle" >&2; exit 1; }
[ "$(uname -s)" = Darwin ] || { echo "dmg: hdiutil runs on macOS only" >&2; exit 1; }

stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
# ditto keeps the bundle's signature, extended attributes and links intact,
# where cp -R can drop them.
ditto "$app" "$stage/jitllm.app"
ln -s /Applications "$stage/Applications"

rm -f "$out"
hdiutil create -volname jitllm -srcfolder "$stage" -fs HFS+ -format UDZO -ov "$out" >/dev/null
hdiutil verify "$out" >/dev/null
echo "$out"
