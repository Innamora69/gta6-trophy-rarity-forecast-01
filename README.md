# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Innamora69/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Innamora69/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts each Grand Theft Auto VI PlayStation 5 trophy's earn rate on Exophase a week after launch, as five quantiles per trophy (5%, 25%, 50%, 75%, 95%), scored by pinball loss on the logit scale.

## How it works

Exophase prints each trophy's earn rate as a share of the players it tracks. Two forces move a rate: players earning the trophy, and new players syncing a profile. The second force is violent at launch: from launch night to the first refresh the tracked-player count jumped 638 → 3,868 while every rate but the one at 100% fell about 4.5x. So a rate jumps at each Exophase refresh, and the launch-night reads come from a different, selected sample than the population that follows.

`main.py` (Python 3.13, standard library only, no network) forecasts from one case file:

1. Collapse the history to distinct snapshots — the page is unchanged between refreshes, so identical reads carry no information.
2. Drop every snapshot before the first big jump in players tracked (the launch-night sample).
3. Per trophy, fit a least-squares line of logit(rate) against ln(1 + hours since the release day began) and read it at the target. Log-time makes the 4.5-day extrapolation gentle.
4. If the history never moved at all, fall back to a prior instead of a flat line: shift the median down 0.9 logit units — the direction and rough size of the one refresh on record — and widen the spread.
5. Spread: a random-walk term growing with the horizon, plus the fit's residual, plus the slope's standard error times the extrapolation distance, plus a refresh-shock term.
6. Structure: the platinum is capped below the rarest other trophy (it is held only by players holding every trophy); the one trophy at 100% stays near 100% with a small hedge below it.

Local validation (practice answers reconstructed from the aggregates the competition publishes for the first refresh — 3,868 players tracked, 4,276 trophies earned): mean pinball loss 0.33 on the practice cases, against 1.38 for the persistence baseline the kit ships; 0.21 against 0.26 on synthetic final-like cases.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes `<id>.out` to `$MAKEFUN_OUTPUT_DIR` (else `/output`): `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`. Deterministic: no randomness, no wall-clock use.

---

<sub>Entered on [make.fun](https://make.fun/?ref=w36tq042&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
