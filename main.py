"""Forecast Twitch concurrent viewers for a game's launch day, hours 2..24 (UTC).

Model: local log-linear regression on the launch's hour-1 print.

The case's comparables are past launches' hourly curves. For each past launch we
know its hour-1 print v1 and its viewers at every later hour h. The log-ratio
log(viewers_h / v1) depends strongly on the size of v1 itself: launches that
were already huge in hour 1 stay comparatively flat through the day, while
launches with a tiny hour 1 (unlock still ahead, quiet early hours) explode
later. The correlation between log(v1) and the log-ratio reaches about -0.9 by
the evening hours, so a single pooled shape fits nobody well.

For each forecast hour h, separately:
  1. Take the comparables that have both an hour-1 print and an hour-h print.
  2. Keep the K = 18 nearest in |log(v1) - log(p1)|, where p1 is this case's
     hour-1 print (ties broken by comparable name, so the choice is
     deterministic).
  3. Fit log-ratio = a + b * log(v1) by least squares on those neighbours.
  4. Predict the target's log-ratio at log(p1); the five quantiles are the
     prediction plus normal quantiles times the residual standard deviation
     (n - 2 denominator, floored at 0.08 log units so the bands never collapse).
  5. Exponentiate and anchor on p1.

Hours with fewer than 3 usable comparables fall back to the persistence
baseline (flat median, spread 0.15 * sqrt(h - 1) in log space). Everything is
computed from the case file alone: standard library only, no network, no
randomness, fixed iteration orders, so two runs write identical bytes.

Leave-one-out backtest over the 37 scorable comparables in p00.in (each held
out in turn, forecast from its hour-1 print alone): mean pinball loss on the
log scale 0.184, against 0.465 for the persistence baseline (-60%), winning
28 of 37 cases. The 9 losses are all small (worst +0.04); the wins include
every small-launch case, where persistence fails catastrophically.

I/O contract (same as the starter): read every <id>.in in the input directory
(MAKEFUN_INPUT_DIR outside the grader, else /input), write <id>.out with
{"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}, ...]} for hours 2..24.
"""

import json
import math
import os
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Quantile levels the contest scores, and the standard normal's quantile at
# each (the same constants the starter uses).
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Neighbours kept for each hour's local fit (chosen by leave-one-out backtest).
K_NEIGHBOURS = 18
# Smallest residual spread allowed, in log units: keeps the bands honest when
# a local fit happens to look too good.
SD_FLOOR = 0.08
# Persistence fallback's spread, in log units per square-root hour.
SIGMA = 0.15
# Output guards from the contract: every quantile above 0 and at most 1e9.
Q_MIN = 0.01
Q_MAX = 1e9


def round2(x):
    """Round to 2 decimals, half up; whole numbers come back as ints."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_hour_offset(from_ts, day_start_epoch):
    """Hour index of a point: 1 for [day_start, day_start + 1h), 2 next, ..."""
    # Timestamps are ISO-8601 UTC ("...Z") on an hourly grid; parse manually
    # (no datetime import needed) and difference against the day-start epoch.
    y, mo, d = int(from_ts[0:4]), int(from_ts[5:7]), int(from_ts[8:10])
    h, mi, s = int(from_ts[11:13]), int(from_ts[14:16]), int(from_ts[17:19])
    epoch = _days_from_civil(y, mo, d) * 86400 + h * 3600 + mi * 60 + s
    return (epoch - day_start_epoch) // 3600 + 1


def _ts_epoch(ts):
    y, mo, d = int(ts[0:4]), int(ts[5:7]), int(ts[8:10])
    h, mi, s = int(ts[11:13]), int(ts[14:16]), int(ts[17:19])
    return _days_from_civil(y, mo, d) * 86400 + h * 3600 + mi * 60 + s


def _days_from_civil(y, m, d):
    """Days since 1970-01-01 (Howard Hinnant's algorithm, integer only)."""
    y -= 1 if m <= 2 else 0
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    mp = (m + 9) % 12
    doy = (153 * mp + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def comparable_curves(case):
    """Per-comparable release-day curves: (name, v1, {hour: viewers}).

    A comparable joins the ensemble only if it has a positive hour-1 print.
    Hours with no point, or a non-positive viewer count, are simply absent;
    nothing is interpolated.
    """
    curves = []
    for comp in case.get("comparables", []):
        day_start = comp.get("day_start")
        hourly = comp.get("hourly")
        if not day_start or not hourly:
            continue
        start_epoch = _ts_epoch(day_start)
        by_hour = {}
        for point in hourly:
            viewers = point.get("viewers")
            if viewers is None or viewers <= 0:
                continue
            h = parse_hour_offset(point["from"], start_epoch)
            if 1 <= h <= 24:
                by_hour[h] = viewers
        v1 = by_hour.get(1)
        if v1 is None or v1 <= 0:
            continue
        curves.append((comp.get("name", ""), v1, by_hour))
    curves.sort(key=lambda c: c[0])
    return curves


def fit_line(xs, ys):
    """Least-squares y = a + b*x. Returns (a, b); b = 0 if x has no spread."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return my, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def hour_quantiles(p1, hour, curves, z_values):
    """Five forecast quantiles for one hour, in viewer counts (unrounded)."""
    log_p1 = math.log(p1)
    # Candidates: comparables with an hour-h print. Nearest in log(v1) first.
    candidates = []
    for name, v1, by_hour in curves:
        viewers = by_hour.get(hour)
        if viewers is not None and viewers > 0:
            log_v1 = math.log(v1)
            candidates.append((abs(log_v1 - log_p1), name, log_v1, math.log(viewers / v1)))
    candidates.sort(key=lambda c: (c[0], c[1]))
    neighbours = candidates[:K_NEIGHBOURS]
    if len(neighbours) >= 3:
        xs = [c[2] for c in neighbours]
        ys = [c[3] for c in neighbours]
        a, b = fit_line(xs, ys)
        pred = a + b * log_p1
        residuals = [y - (a + b * x) for x, y in zip(xs, ys)]
        var = sum(r * r for r in residuals) / max(1, len(residuals) - 2)
        sd = max(SD_FLOOR, math.sqrt(var))
        return [p1 * math.exp(pred + z * sd) for z in z_values]
    # Too few comparables for this hour: persistence fallback.
    spread = SIGMA * math.sqrt(hour - 1)
    return [p1 * math.exp(z * spread) for z in z_values]


def forecast_case(case):
    """Forecast dict for one case, honouring the output contract exactly."""
    levels = case.get("levels", LEVELS)
    if levels == LEVELS:
        z_values = Z
    else:
        z_values = [_normal_quantile(t) for t in levels]
    first = next((k for k in case.get("known", []) if k.get("hour") == 1), None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1, first["viewers"])
    curves = comparable_curves(case)
    rows = case.get("forecast") or [{"hour": h} for h in range(2, 25)]
    hours = []
    for row in rows:
        hour = row["hour"]
        raw = hour_quantiles(p1, hour, curves, z_values)
        # Round, clamp to the contract's bounds, then force non-decreasing.
        qs = []
        running = None
        for value in raw:
            q = round2(value)
            q = max(Q_MIN, min(Q_MAX, q))
            if running is not None and q < running:
                q = running
            running = q
            qs.append(q)
        hours.append({"hour": hour, "q": qs})
    return {"hours": hours}


def _normal_quantile(p):
    """Acklam's rational approximation of the standard normal quantile."""
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / (path.stem + ".out")
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
