#!/bin/sh
# Re-renders the desktop app's Discover and Convert screenshots and copies
# them into public/shots. The frames come from the app's own mock scenarios
# (ui/cmd/shots): the real window, mounted headlessly, against fixtures, so
# they need no model and no GPU and change only when the app does.
#
# chat-, models- and machine-*.png are not from here: they are the real app
# on an Xvfb display (DISPLAY=:99, driven with xdotool) with
# Qwen3-30B-A3B-Q4_K_M loaded, taken with `import -window root`.
set -eu
site=$(cd "$(dirname "$0")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
(cd "$site/../ui" && ../scripts/cap 8G -- go run ./cmd/shots -out "$tmp")
mkdir -p "$site/public/shots"
for f in discover-smartest convert-picked; do
	cp "$tmp/$f.png" "$site/public/shots/$f-light.png"
	cp "$tmp/$f-dark.png" "$site/public/shots/$f-dark.png"
done
