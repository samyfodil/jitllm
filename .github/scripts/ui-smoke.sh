#!/usr/bin/env bash
# Open the real jitllm-ui window in mock mode on the X display $DISPLAY names,
# let it run, and take a picture of it.
#
#     ui-smoke.sh path/to/jitllm-ui out-dir
#
# `-mock` is the fixtures-only mode: package mock stands in for the engine,
# the hardware probe and the model files, so the window needs no model. It
# does need a GPU surface: gogpu renders through Vulkan, which on a runner is
# Mesa's software rasterizer (lavapipe) under Xvfb.
#
# What it proves: the binary starts in mock mode -- the X11 window, the Vulkan
# device, the shaders, the mock engine wired to every screen -- and is still
# running after $UI_SMOKE_SECONDS. What it does not: that the window shows
# anything. Under Xvfb with lavapipe the window stays black while vkcube on the
# same display draws, so the app presents no frame there; the screenshot is
# kept and its contrast reported, not gated, until that is understood. The
# frames themselves are stage.TestEveryScenarioRendersClean's, offscreen.
set -uo pipefail
bin=$1
out=$2
secs=${UI_SMOKE_SECONDS:-20}
mkdir -p "$out"

# A private settings and data home, so the run reads no settings and leaves
# none behind.
home=$(mktemp -d)
XDG_CONFIG_HOME=$home/config XDG_DATA_HOME=$home/data "$bin" -mock >"$out/jitllm-ui.log" 2>&1 &
pid=$!
sleep "$secs"

if ! kill -0 "$pid" 2>/dev/null; then
  wait "$pid"
  echo "::error::jitllm-ui -mock exited with status $? within ${secs}s"
  cat "$out/jitllm-ui.log"
  exit 1
fi

rc=0
shot=$out/jitllm-ui-mock.png
if ! import -window root "$shot"; then
  echo "::error::could not take a screenshot of display $DISPLAY"
  rc=1
else
  # A window that never drew leaves the root one flat colour.
  sd=$(identify -format '%[fx:standard_deviation]' "$shot")
  if [ "$(awk -v s="$sd" 'BEGIN { print (s > 0.01) }')" = 1 ]; then
    drew="drew a frame (screenshot standard deviation $sd)"
  else
    drew="presented nothing: the screenshot is one flat colour"
    echo "::warning::jitllm-ui -mock $drew"
  fi
  echo "jitllm-ui -mock ran ${secs}s and $drew" | tee -a "${GITHUB_STEP_SUMMARY:-/dev/null}"
fi

kill "$pid"
wait "$pid"
echo "--- jitllm-ui log"
cat "$out/jitllm-ui.log"
rm -rf "$home"
exit $rc
