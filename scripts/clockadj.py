#!/usr/bin/env python3
"""Read vs-llamacpp.sh's round lines and report the ratio AT A REFERENCE CLOCK.

★ WHY THIS EXISTS, AND WHY IT IS NOT A SECOND SCOREBOARD.

RULE 2's statistic -- the median of per-round ratios with its IQR -- assumes the
ratio is one quantity being sampled with noise. On a box whose clock wanders,
that assumption can be false: this file's own evidence says the two engines do
not shed rate to a clamp equally (the M4 prefill row, jitllm -7.4% against
llama.cpp -4.2%), so the ratio may be a FUNCTION of frequency rather than a
constant observed at several. When it is, the IQR is measuring the frequency
spread and no number of rounds converges -- which is exactly what a 16-round run
at IQR/median 0.134 looks like.

The physical fix is to pin the clock (min_perf_pct = max_perf_pct) and take an
ordinary RULE 2 row. That needs root. This is the fallback when it is not
available: estimate the ratio at ONE stated frequency by regressing the coherent
pairs on their own clock, and report the residual dispersion after the clock
term is removed.

★ IT IS NOT A LICENCE TO QUOTE A ROW THE GATES REFUSED. Three things must hold
before the output means anything, and each is printed rather than assumed:

  - the trend must be REAL (|r| >= 0.5). Below that the adjustment is fitting
    noise and the plain median is the honest statistic;
  - the reference clock must be INSIDE the measured range, never extrapolated;
  - the RESIDUAL dispersion must clear RULE 2's bar. If removing the clock term
    does not tighten the spread, frequency was not what the spread was, and the
    row still fails -- for some other reason this script cannot see.

Usage:  scripts/clockadj.py [-ref MHZ] < board-output.txt
"""

import re
import sys
import statistics

ROUND = re.compile(
    r"jitllm\s+([\d.]+)\s+([\d.]+)\s+llama\.cpp\s+([\d.]+)\s+([\d.]+)\s+"
    r"\[MHz\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\]"
)


def pairs(text):
    """Each round contributes TWO (jitllm, llama.cpp) pairs with their clocks.

    The ABBA order is a1 b1 b2 a2, and the frequency list is printed as
    [f1 f2 g1 g2] -- jitllm's two, then llama.cpp's two -- so a1 pairs with b1
    at (f1, g1) and a2 with b2 at (f2, g2). Pairing a1 with b2 would compare
    samples taken minutes apart, which is the thing the interleave exists to
    avoid.
    """
    out = []
    for m in ROUND.finditer(text):
        a1, a2, b1, b2, f1, f2, g1, g2 = (float(x) for x in m.groups())
        out.append((a1, b1, f1, g1))
        out.append((a2, b2, f2, g2))
    return out


def main():
    ref = None
    args = sys.argv[1:]
    if len(args) == 2 and args[0] == "-ref":
        ref = float(args[1])
    rows = pairs(sys.stdin.read())
    if not rows:
        print("no round lines on stdin")
        return 1

    freqs = [f for r in rows for f in (r[2], r[3]) if f > 0]
    if not freqs:
        print("no frequencies recorded (darwin?) -- nothing to adjust")
        return 1
    fmed = statistics.median(freqs)
    keep = [
        r
        for r in rows
        if r[2] >= 0.75 * fmed
        and r[3] >= 0.75 * fmed
        and max(r[2], r[3]) / min(r[2], r[3]) <= 1.05
    ]
    print("pairs %d, unclamped and coherent %d, session median %.0f MHz"
          % (len(rows), len(keep), fmed))
    if len(keep) < 6:
        print("  fewer than 6 usable pairs: nothing to fit")
        return 1

    xs = [(r[2] + r[3]) / 2 for r in keep]
    ys = [r[0] / r[1] for r in keep]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        print("  the clock does not vary across the survivors: no fit needed")
        return 0
    r = sxy / (sxx * syy) ** 0.5
    slope = sxy / sxx
    lo, hi = min(xs), max(xs)
    plain = statistics.median(ys)
    pl_srt = sorted(ys)
    pl_iqr = pl_srt[n * 3 // 4] - pl_srt[n // 4]
    print("  plain      ratio %.4f  IQR/median %.3f  (RULE 2's statistic)"
          % (plain, pl_iqr / plain))
    print("  clock      r=%+.2f, %+.4f per 100 MHz over %.0f-%.0f MHz"
          % (r, 100 * slope, lo, hi))

    if abs(r) < 0.5:
        print("  NO TREND: the adjustment would fit noise. The plain median above"
              " is the honest statistic, and whatever blew its IQR is not the clock.")
        return 0

    if ref is None:
        ref = fmed
    if not (lo <= ref <= hi):
        print("  reference %.0f MHz is OUTSIDE the measured %.0f-%.0f: refusing to"
              " extrapolate" % (ref, lo, hi))
        return 1

    # Residuals about the fit are what the row's dispersion becomes once the
    # clock term is gone. If they do not tighten, frequency was not the spread.
    adj = [y - slope * (x - ref) for x, y in zip(xs, ys)]
    a_srt = sorted(adj)
    a_med = statistics.median(adj)
    a_iqr = a_srt[n * 3 // 4] - a_srt[n // 4]
    print("  at %.0f MHz  ratio %.4f  IQR/median %.3f  (n=%d)"
          % (ref, a_med, a_iqr / a_med, n))

    # ★ THE TEST IS WHETHER THE ADJUSTMENT HELPED, NOT WHETHER THE RESULT GATES.
    # A residual that is WIDER than the plain spread means the fitted line was
    # not in the data: subtracting it injected variance rather than removing it,
    # and the correlation that licensed it was a moderate r over few points
    # rather than a mechanism. The first version of this script printed
    # "GATES ... (was 0.013)" over a spread that had gone 0.013 -> 0.064, which
    # reads as a confirmation of exactly the thing it disproved.
    before, after = pl_iqr / plain, a_iqr / a_med
    if after >= before:
        print("  THE ADJUSTMENT MADE IT WORSE: IQR/median %.3f -> %.3f. The clock"
              " term was not\n  the spread -- r=%+.2f over %d points licensed a line"
              " that is not in the data.\n  The plain median %.4f is the statistic;"
              " this row is not clock-limited."
              % (before, after, r, n, plain))
        return 0
    if after <= 0.10:
        print("  GATES once the clock term is removed: IQR/median %.3f -> %.3f"
              % (before, after))
        return 0
    print("  STILL REJECTED at IQR/median %.3f (was %.3f) -- the clock was part of"
          " the spread\n  and not the whole of it." % (after, before))
    return 0


if __name__ == "__main__":
    sys.exit(main())
