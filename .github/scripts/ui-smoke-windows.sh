#!/usr/bin/env bash
# Start the packaged jitllm-ui.exe (a GUI program) on a Windows runner, once in
# mock mode and once against the real engine with no model, let each run, and
# fail on a crash: the process gone before its time, or a crash report in the
# app's settings folder (package crash writes %AppData%\jitllm\crash.log and
# crash-report.txt there).
#
#     ui-smoke-windows.sh path/to/jitllm-ui.exe out-dir
#
# A runner has no GPU; gogpu falls back to whatever adapter Windows offers
# (WARP or software). What it proves: the window, the device, the hardware
# probe (CUDA and Vulkan opened through goffi when present) and the engine
# worker start and keep running. It does not drive input.
set -uo pipefail
bin=$1
out=$2
secs=${UI_SMOKE_SECONDS:-30}
mkdir -p "$out"

rc=0
for mode in mock engine; do
  home=$(mktemp -d)
  appdata=$(cygpath -w "$home/appdata" 2>/dev/null || echo "$home/appdata")
  mkdir -p "$home/appdata" "$home/models"
  args=()
  [ "$mode" = mock ] && args=(-mock)
  APPDATA=$appdata LOCALAPPDATA=$appdata JITLLM_MODELS=$home/models \
    "$bin" "${args[@]}" >"$out/$mode.log" 2>&1 &
  pid=$!
  sleep "$secs"
  if ! kill -0 "$pid" 2>/dev/null; then
    wait "$pid"
    echo "::error::jitllm-ui ($mode) exited with status $? within ${secs}s"
    rc=1
  else
    kill "$pid"
    wait "$pid"
  fi
  dir=$home/appdata/jitllm
  for f in crash.log crash-report.txt; do
    if [ -s "$dir/$f" ]; then
      echo "::error::jitllm-ui ($mode) left $f"
      cat "$dir/$f"
      cp "$dir/$f" "$out/$mode-$f"
      rc=1
    fi
  done
  echo "--- jitllm-ui ($mode) log"
  cat "$out/$mode.log"
  [ -f "$dir/jitllm-ui.log" ] && cat "$dir/jitllm-ui.log"
  echo "jitllm-ui ($mode) ran ${secs}s" | tee -a "${GITHUB_STEP_SUMMARY:-/dev/null}"
  rm -rf "$home"
done
exit $rc
