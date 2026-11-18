# GTA VI Twitch CCU forecaster

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/bagusahmad03/makefun-gta-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/bagusahmad03/makefun-gta-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A program that forecasts concurrent Twitch viewers in the Grand Theft Auto VI
category for hours 2..24 of release day (UTC), as five quantiles (5%, 25%,
50%, 75%, 95%) per hour, from the case's hour-1 print plus the 38 comparable
past launch curves shipped in the case file.

## How to run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The grader runs `python3 -P -s /bot/main.py` with the case in `/input` and
expects `<id>.out` in `/output`. Python 3 standard library only, no network,
deterministic: two runs on the same case write identical bytes.

## Model

Work in log space, anchored on the hour-1 print p1. The key fact in the
comparable data is that the launch-day shape depends on the size of hour 1
itself: the correlation between log(v1) and the log-ratio log(viewers_h / v1)
reaches about -0.9 by the evening hours. Launches that were already huge in
hour 1 stay comparatively flat; launches with a tiny hour 1 explode later as
regions unlock. A single pooled shape fits nobody well.

For each forecast hour, separately:

1. Take the comparables with both an hour-1 print and an hour-h print
   (missing hours are skipped, nothing is interpolated).
2. Keep the 18 nearest in |log(v1) - log(p1)|.
3. Fit log-ratio = a + b * log(v1) by least squares on those neighbours,
   with the slope clamped to <= 0 (a positive fitted slope is window noise;
   extrapolating it at a huge hour-1 print points the wrong way).
4. Predict at log(p1); the five quantiles are the prediction plus standard
   normal quantiles times the residual spread: the larger of the residual
   standard deviation (n - 2 denominator) and 1.4826 times the residuals'
   median absolute deviation, floored at 0.08 log units.
5. Exponentiate and anchor on p1.

Hours with fewer than 3 usable comparables fall back to the persistence
baseline (flat median, spread 0.15 * sqrt(h - 1) in log space).

## Backtest

Leave-one-out over the 37 scorable comparables in `p00.in` (each held out in
turn, forecast from its hour-1 print alone, scored by mean pinball loss on
the log scale over hours 2..24 x 5 levels):

| Model | Mean pinball loss (log) |
|---|---|
| This forecaster (v2: slope clamp + MAD bands) | 0.183 |
| Previous version (v1) | 0.184 |
| Persistence baseline (starter) | 0.465 |

That is 60% lower loss, winning 28 of 37 cases. The 9 losses are all small
(worst +0.04); the wins include every small-launch case, where persistence
fails catastrophically (losses up to 1.97 vs ~0.2 here). Extrapolation was
stress-tested by holding out the 3 largest and 3 smallest launches: the model
beats persistence on both extremes (0.06 vs 0.09 on the largest, 0.24 vs 1.81
on the smallest).

Runtime: ~0.1 s per case on this machine (limit 8 s), output ~1.5 KB per case
(limit 65,536 bytes). Outputs were verified byte-identical across repeated
runs, and the output format was checked exactly: 23 rows, hours 2..24 each
once, five finite non-decreasing quantiles per row within (0, 1e9].

## Data

All data comes from the case files themselves: the case's `known` hour-1
print and its `comparables` (past launches' hourly SullyGnome curves, release
day -14d to +3d). No external data is fetched or embedded, and the program
reads nothing outside the input directory at runtime. Note the practice
cases' hourly figures are SullyGnome hourly averages, while the final case
is scored on the mean of minute reads; the model uses only ratios to the
hour-1 print, so the change of cadence does not change the code path.

## Caveats

- The backtest measures beating the persistence baseline. The contest also
  requires beating a house AI forecast, which cannot be reproduced locally;
  the 60% margin over persistence is the evidence available, not a guarantee.
- The final case is a single launch day. One day of realised viewership can
  always surprise any forecaster; the quantile bands are set honestly wide
  where the comparable ensemble disagrees.

## Files

- `main.py` — the forecaster (stdlib only, Python 3.12/3.13 compatible)
- `README.md` — this file
- `LICENSE` — MIT
