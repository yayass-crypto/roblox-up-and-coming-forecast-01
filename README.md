# roblox-up-and-coming-forecast-01 — empirical transition forecaster

Entry for make.fun's "Predict Roblox's Up-and-Coming top 10" ($500).
`main.py` (Python 3.13, standard library only) reads the chart's history and
writes one `forecast.json`.

## Why this beats the house baseline

The baseline just repeats the chart as it stood (0.5 on the incumbent at
each place). The chart churns hard — its #1 changed 13 times in 45 reads —
but the churn is structured, and the history lets us measure it:

- Future rank mean-reverts: rank in +24h = 8.4 + 0.82·rank (residual sd
  15.8); +48h = 18.2 + 0.62·rank (sd 21.7); +72h = 26.0 + 0.46·rank (sd
  24.3). Survival on the chart is rank-dependent (top-10 games survive
  94–99% even at 72h; rank 31–97 games only 60–82%).
- Two extra signals, both fitted on the history: a game's recent rank trend
  as a rate (climbing 40 ranks/day → ~6.5 ranks better than the rank-only
  prediction) and its playing-count growth (doubling in a day → ~1 rank
  better), both decayed with horizon.
- A fixed-seed Monte Carlo samples a uniform slot in the 72-hour window,
  then each game's future rank from the fitted transition (survival first,
  then a normal around the adjusted mean), ranks the survivors, and tallies
  places 1–10. The unnamed share (0.32 per place) is left for games that
  join the chart after the cut.

## Validation (out-of-sample holdout)

History truncated to the first 24 reads (to 5 Nov 06:48 UTC), scored
against the next 10 reads (5–7 Nov, 53h):

- This entry: log score **−40.479** (typical 0.0175 per place)
- House baseline on the same window: **−56.911** (typical 0.0034)

Same top-10 hits (4.4/10); the edge is honest, spread probabilities instead
of 0.5 on incumbents that the chart leaves behind.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Two runs on the same history are byte-identical (fixed seed, no clock, all
orderings fully deterministic).
