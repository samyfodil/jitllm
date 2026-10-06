#!/bin/sh
# Re-renders the cards at the top of the repository README
# (docs/assets/readme/) from this site: builds it, then runs readme-cards.mjs
# under Playwright's headless Chromium and ffmpeg. Playwright is not one of the
# site's dependencies; set PLAYWRIGHT to a directory whose node_modules holds
# it, or it is installed into a temporary directory for this run.
#
#   website/scripts/readme-cards.sh            # every card
#   ONLY=perf website/scripts/readme-cards.sh  # one card
#
# The perf card's numbers are typed in readme-cards.mjs from the README's
# Performance section; change them there when that section changes.
set -eu
site=$(cd "$(dirname "$0")/.." && pwd)
command -v ffmpeg >/dev/null || { echo "readme-cards: ffmpeg is needed for the GIFs" >&2; exit 1; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cd "$site"
[ -d node_modules ] || npm ci --no-audit --no-fund
npm run build
if [ -z "${PLAYWRIGHT:-}" ]; then
	npm install --prefix "$tmp" --no-audit --no-fund --no-save playwright
	(cd "$tmp" && npx playwright install chromium)
	PLAYWRIGHT=$tmp
fi
PLAYWRIGHT=$PLAYWRIGHT ../scripts/cap 4G -- node scripts/readme-cards.mjs
