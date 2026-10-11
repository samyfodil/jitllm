#!/bin/sh
# Authenticode-signs one Windows executable in place, with osslsigncode.
#
#   WINDOWS_SIGN_PFX=cert.pfx WINDOWS_SIGN_PASSWORD=... packaging/windows/sign.sh jitllm.exe
#
# build.sh calls it for each program and makensis for the uninstaller and the
# setup (!uninstfinalize, !finalize). SHA-256, timestamped, so the signature
# outlives the certificate. WINDOWS_SIGN_TIMESTAMP overrides the timestamp
# server.
set -eu

[ $# -eq 1 ] || { echo "usage: $0 <file.exe>" >&2; exit 2; }
: "${WINDOWS_SIGN_PFX:?set WINDOWS_SIGN_PFX to the certificate (.pfx)}"
: "${WINDOWS_SIGN_PASSWORD:?set WINDOWS_SIGN_PASSWORD}"
ts=${WINDOWS_SIGN_TIMESTAMP:-http://timestamp.digicert.com}

tmp="$1.signed"
osslsigncode sign -pkcs12 "$WINDOWS_SIGN_PFX" -pass "$WINDOWS_SIGN_PASSWORD" \
	-h sha256 -n jitllm -i https://jitllm.org -ts "$ts" \
	-in "$1" -out "$tmp"
mv "$tmp" "$1"
