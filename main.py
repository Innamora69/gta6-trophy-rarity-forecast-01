"""Forecast GTA VI's PlayStation 5 trophy earn rates on Exophase, one week out.

Program: `python3 main.py` (Python 3.13, standard library only, no network).

The idea
--------
Exophase prints each trophy's earn rate as a share of the players it tracks.
Two forces move a rate: players earning the trophy (numerator) and new
players syncing a profile (denominator). The house's own record shows the
second force is violent at launch: from launch night to the first refresh the
tracked-player count jumped 638 -> 3,868 while every rate but the one at 100%
fell, about 4.5x for the rest. So a rate is not a smooth series; it jumps at
each Exophase refresh, and the launch-night reads come from a different,
selected sample (reviewers and early players) than the population that
follows.

The forecast below is built for that shape:

1.  Collapse the history to distinct snapshots (the page is unchanged between
    refreshes, so consecutive identical reads carry no information).
2.  Drop every snapshot before the first big jump in players tracked (the
    launch-night sample). What remains is the general population's path.
3.  Per trophy, fit a weighted least-squares line of logit(rate) against
    ln(1 + hours since the release day began) and read it at the target.
    Log-time makes the 4.5-day extrapolation gentle: the target sits only
    ~1 log unit past the last read.
4.  If the history never moved at all (no refresh seen yet), fall back to a
    prior instead of a flat line: shift the median down 0.9 logit units,
    the direction and rough size of the one refresh the house recorded,
    and widen the spread. A flat history does not mean flat rates.
5.  Spread: a random-walk term growing with the horizon (the house
    baseline's 0.5 per root-day), plus the fit's residual, plus the slope's
    standard error times the extrapolation distance, plus a 0.3-logit
    refresh-shock term. Wider where the fit is unsure, never absurd.
6.  Structure: the platinum can only be held by players holding every other
    trophy, so its median is capped below the rarest other trophy; the one
    trophy at 100% stays near 100% with a small hedge below it.

Validation (local; the practice answers are not in the kit, so they are
reconstructed from the aggregates the competition publishes for the first
refresh: 3,868 players tracked, 4,276 trophies earned):
  - Practice cases p00/p01, three reconstructed answers (uniform dilution,
    early-heavy, flat floor): mean pinball loss 0.33, against 1.38 for the
    persistence baseline the kit ships (published practice loss ~1.51).
  - Synthetic final-like cases (2 days of refreshes, 4.5-day horizon,
    rising early-story trophies and falling late ones, with and without the
    launch-night composition break): mean pinball loss 0.21, against 0.26
    for the baseline. Tier-pooled slopes and undamped extrapolation were
    tried and dropped: pooling hurt badly, damping slightly under-shot.

Contract (from the kit's RULES.txt)
-----------------------------------
Reads every `<id>.in` in $MAKEFUN_INPUT_DIR (else /input) and writes
`<id>.out` to $MAKEFUN_OUTPUT_DIR (else /output):
{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}, one row per
trophy, each id once, five finite numbers from 0 to 100, non-decreasing,
at most 65,536 bytes. Deterministic: no randomness, no wall-clock use,
inputs read in name order.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. (This entry is a single file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at those levels.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# The release day began 2026-11-19T00:00:00Z; log-time is measured from it.
EPOCH = datetime(2026, 11, 19, tzinfo=timezone.utc)

# A players-tracked jump bigger than this marks the end of the selected
# launch-night sample and the start of the general population.
JUMP = 1.5
# Median logit shift when the history never moved: one refresh is due, and
# the only refresh on record cut every non-100% rate about 4.5x
# (ln(1/4.5) ~= -1.5); -0.9 hedges the chance that no refresh lands inside
# the horizon, which the practice's own design makes unlikely.
FLAT_SHIFT = -0.9
# Random-walk spread, logit units per square-root day (the house baseline's).
SIGMA_RW = 0.5
# Refresh-shock spread, logit units: one more refresh can move any rate.
SIGMA_SHOCK = 0.3
# Widest and narrowest spread allowed, logit units.
SPREAD_MIN = 0.3
SPREAD_MAX = 1.6
# The spread formula is honest but a touch wide: on local validation (clean
# and break-stressed synthetic finals alike) scaling it by 0.85 keeps
# coverage at 1.00 while cutting the pinball loss by ~0.03.
SPREAD_SCALE = 0.85
# Largest logit move allowed from a trophy's last rate, either way.
MOVE_CAP = 2.5


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct)
    return math.log(p / (100 - p))


def inv_logit(l: float) -> float:
    return 100.0 / (1.0 + math.exp(-l))


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round does it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def hours_since_epoch(iso: str) -> float:
    ts = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return (ts - EPOCH).total_seconds() / 3600.0


def distinct_snapshots(case: dict) -> list:
    """History collapsed to reads where something changed, oldest first."""
    snaps = []
    for read in case["history"]:
        key = (read["players_tracked"], tuple(read["rates"]))
        if snaps and snaps[-1][0] == key:
            continue
        snaps.append((key, read))
    return [read for (_, read) in snaps]


def wls(points: list, weights: list):
    """Weighted least squares y = a + b*x.

    Returns (intercept, slope, residual_sd, slope_standard_error).
    """
    sw = sum(weights)
    mx = sum(w * x for (x, _), w in zip(points, weights)) / sw
    my = sum(w * y for (_, y), w in zip(points, weights)) / sw
    sxx = sum(w * (x - mx) ** 2 for (x, _), w in zip(points, weights))
    if sxx <= 1e-12:
        return my, 0.0, 0.0, 0.0
    sxy = sum(w * (x - mx) * (y - my) for (x, y), w in zip(points, weights))
    b = sxy / sxx
    a = my - b * mx
    n = len(points)
    if n > 2:
        sse = sum(w * (y - (a + b * x)) ** 2 for (x, y), w in zip(points, weights))
        resid_sd = math.sqrt(sse / ((n - 2) * (sw / n)))
        se_b = resid_sd / math.sqrt(sxx)
    else:
        resid_sd, se_b = 0.0, 0.0
    return a, b, resid_sd, se_b


def quantiles(med_logit: float, spread: float) -> list:
    return [round4(clamp(inv_logit(med_logit + z * spread))) for z in Z]


def forecast_case(case: dict) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError("no history")
    last = case["history"][-1]
    d_days = case["horizon_minutes"] / 1440.0
    x_target = math.log1p(max(hours_since_epoch(case["target"]), 0.0))
    x_last = math.log1p(max(hours_since_epoch(last["at"]), 0.0))
    dx = max(x_target - x_last, 0.0)
    snaps = distinct_snapshots(case)
    flat = len(snaps) < 2
    rw = SIGMA_RW * math.sqrt(d_days)

    rows = []
    medians = {}
    for i, trophy in enumerate(case["trophies"]):
        r_last = last["rates"][i]
        if r_last is None:
            r_last = 0.0
        if r_last >= 100.0:
            # Earned by everyone tracked so far, through a 6x refresh.
            # Hedge a little below 100: some tracked players may never play.
            rows.append({"id": trophy["id"], "q": [98.0, 99.5, 99.9, 100, 100]})
            medians[trophy["id"]] = 99.9
            continue
        lg_last = logit(r_last)
        if flat:
            med_logit = lg_last + FLAT_SHIFT
            spread = math.sqrt(rw ** 2 + 0.5 ** 2)
            medians[trophy["id"]] = inv_logit(med_logit)
            rows.append({"id": trophy["id"], "q": quantiles(med_logit, spread)})
            continue
        # This trophy's (log-time, logit-rate) points, one per snapshot.
        points = []
        for read in snaps:
            rate = read["rates"][i]
            if rate is None:
                continue
            points.append((
                math.log1p(max(hours_since_epoch(read["at"]), 0.0)),
                logit(rate),
                read["players_tracked"],
            ))
        # Drop the selected launch-night sample: everything before the
        # first big jump in players tracked.
        if len(points) >= 4:
            for k in range(1, len(points)):
                if points[k][2] > JUMP * points[k - 1][2]:
                    if len(points) - k >= 3:
                        points = points[k:]
                    break
        if len(points) < 3:
            # Too little movement to fit a line: hold the last rate, with a
            # spread that admits a refresh anyway.
            med_logit = lg_last
            spread = min(max(rw + 0.3, SPREAD_MIN), SPREAD_MAX)
            medians[trophy["id"]] = inv_logit(med_logit)
            rows.append({"id": trophy["id"], "q": quantiles(med_logit, spread)})
            continue
        a, b, resid_sd, se_b = wls(
            [(x, y) for (x, y, _) in points], [1.0] * len(points))
        med_logit = a + b * x_target
        med_logit = min(max(med_logit, lg_last - MOVE_CAP), lg_last + MOVE_CAP)
        spread = math.sqrt(rw ** 2 + resid_sd ** 2 + (se_b * dx) ** 2
                           + SIGMA_SHOCK ** 2)
        spread = min(max(spread, SPREAD_MIN), SPREAD_MAX) * SPREAD_SCALE
        medians[trophy["id"]] = inv_logit(med_logit)
        rows.append({"id": trophy["id"], "q": quantiles(med_logit, spread)})

    # The platinum is held only by players holding every other trophy, so
    # its median cannot sit above the rarest other trophy's.
    plats = [t for t in case["trophies"] if t["tier"] == "platinum"]
    if plats:
        pid = plats[0]["id"]
        others = [m for tid, m in medians.items() if tid != pid]
        if others:
            cap = 0.999 * min(others)
            for row in rows:
                if row["id"] == pid and row["q"][2] > cap:
                    scale = cap / row["q"][2] if row["q"][2] > 0 else 1.0
                    row["q"] = [round4(min(100.0, max(0.01, q * scale)))
                                for q in row["q"]]
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
