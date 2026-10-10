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
  # Mock mode writes to a pipe, as from a terminal; the engine run has its
  # output on NUL, as from Explorer, so its log and standard error go to the
  # settings folder (jitllm-ui.log, crash.log) the way a person's run does.
  if [ "$mode" = mock ]; then
    APPDATA=$appdata LOCALAPPDATA=$appdata JITLLM_MODELS=$home/models \
      "$bin" -mock >"$out/$mode.log" 2>&1 &
  else
    APPDATA=$appdata LOCALAPPDATA=$appdata JITLLM_MODELS=$home/models \
      "$bin" >/dev/null 2>&1 &
    : >"$out/$mode.log"
  fi
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
    # crash.log also collects what drivers print to standard error; only a Go
    # traceback in it is a crash.
    if [ -s "$dir/$f" ] && grep -qE '^goroutine |fatal error:|panic:|^Exception ' "$dir/$f"; then
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
