# roblox-up-and-coming-forecast-01 entry: an empirical transition forecaster.
#
# Idea (why this should beat the house baseline, which just repeats the
# chart as it stood with 0.5/0.15/0.15):
#   * The Up-and-Coming chart churns hard: its #1 changed 13 times in the
#     archive's 45 reads, and the top 10 changed in 42 of 44 consecutive
#     read pairs (median 6 places). But the churn is structured, and the
#     history lets us measure it:
#       rank in +24h  =  8.4 + 0.82 * rank,  residual sd 15.8
#       rank in +48h  = 18.2 + 0.62 * rank,  residual sd 21.7
#       rank in +72h  = 26.0 + 0.46 * rank,  residual sd 24.3
#     i.e. ranks mean-revert toward ~45 with noise that grows with horizon.
#     Survival on the chart is rank-dependent (top-10 games survive 94-99%
#     even at 72h; rank 31-97 games only 60-82%).
#   * Two extra signals, both fitted on the history: a game's recent rank
#     trend (climbed 10 ranks recently -> ~2.8 ranks better than the
#     rank-only prediction) and its playing-count growth (doubling in a day
#     -> ~1.3 ranks better), both decayed with horizon.
#   * A fixed-seed Monte Carlo samples a uniform slot in the 72-hour window,
#     then each game's future rank from the fitted transition (survival
#     first, then a normal around the adjusted mean), ranks the survivors,
#     and tallies places 1-10. The unnamed share (0.32 per place) is left
#     for games that join the chart after the cut, which the practice shows
#     do take top-10 places.
#   * Everything is deterministic: two runs on the same history are
#     byte-identical.
#
# Python 3.13, standard library only. Reads $MAKEFUN_INPUT_DIR (else /input),
# writes $MAKEFUN_OUTPUT_DIR/forecast.json (else /output/forecast.json).

import json
import math
import os
import random
from datetime import datetime

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
SEED = 20261113
MC_ITERATIONS = 3000
UNIVERSE_CAP = 220
MAX_NAMED_PER_PLACE = 100
UNNAMED = 0.32

# Fitted transition points (hours -> (intercept, slope, resid sd)).
TRANSITION = [
    (0.0, 0.0, 1.0, 5.0),
    (24.0, 8.4, 0.82, 15.8),
    (48.0, 18.2, 0.62, 21.7),
    (72.0, 26.0, 0.46, 24.3),
]
# Survival on the chart, by (rank bucket, hours): (rank, h) -> prob.
# Buckets: rank 1-10, 11-30, 31-97; points at 0/24/48/72h.
SURVIVAL = {
    5: [(0.0, 1.0), (24.0, 0.98), (48.0, 0.99), (72.0, 0.94)],
    20: [(0.0, 1.0), (24.0, 0.96), (48.0, 0.90), (72.0, 0.86)],
    65: [(0.0, 1.0), (24.0, 0.82), (48.0, 0.69), (72.0, 0.60)],
}
TREND_COEF = 0.164
GROWTH_COEF = 1.0


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def interp(points, x):
    """Linear interpolation over (x, y) points sorted by x."""
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        if x0 <= x <= x1:
            if x1 == x0:
                return y0
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[-1][1]


def transition_params(h):
    a = interp([(p[0], p[1]) for p in TRANSITION], h)
    b = interp([(p[0], p[2]) for p in TRANSITION], h)
    sd = interp([(p[0], p[3]) for p in TRANSITION], h)
    return a, b, sd


def survival_prob(rank, h):
    keys = sorted(SURVIVAL.keys())
    if rank <= keys[0]:
        return interp(SURVIVAL[keys[0]], h)
    if rank >= keys[-1]:
        return interp(SURVIVAL[keys[-1]], h)
    for i in range(len(keys) - 1):
        if keys[i] <= rank <= keys[i + 1]:
            lo = interp(SURVIVAL[keys[i]], h)
            hi = interp(SURVIVAL[keys[i + 1]], h)
            return lo + (hi - lo) * (rank - keys[i]) / (keys[i + 1] - keys[i])
    return interp(SURVIVAL[keys[-1]], h)


def load_history():
    with open(os.path.join(INPUT_DIR, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    reads = []
    with open(os.path.join(INPUT_DIR, "history.ndjson"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") is True and isinstance(m.get("up_and_coming"), list):
                reads.append(m)
    if not reads:
        raise ValueError("the history holds no ok read")
    return task, reads


def build_games(reads):
    """Per-universe series -> (last rank, trend, growth, last-seen hours)."""
    series = {}
    for ri, m in enumerate(reads):
        t = parse_time(m["at"])
        for r in m["up_and_coming"]:
            uid = r["universe_id"]
            obs = series.setdefault(uid, [])
            obs.append((t, r["rank"], r.get("playing") or 0, r.get("name", "")))
    t_last = parse_time(reads[-1]["at"])
    games = {}
    for uid, obs in series.items():
        obs.sort(key=lambda o: o[0])
        t0, r0, p0, name = obs[-1]
        stale_h = (t_last - t0).total_seconds() / 3600.0
        # Recent rank trend as a rate (ranks/day, negative = climbing), so
        # the signal means the same on hourly reads and archive reads.
        trend = 0.0
        if len(obs) >= 3:
            span_days = (obs[-1][0] - obs[-3][0]).total_seconds() / 86400.0
            if span_days > 0:
                trend = (obs[-1][1] - obs[-3][1]) / span_days
        elif len(obs) == 2:
            span_days = (obs[-1][0] - obs[-2][0]).total_seconds() / 86400.0
            if span_days > 0:
                trend = (obs[-1][1] - obs[-2][1]) / span_days
        trend = max(-100.0, min(100.0, trend))
        # Playing growth: log slope per day over the last two steps.
        growth = 0.0
        if len(obs) >= 3 and obs[-3][2] > 0 and obs[-1][2] > 0:
            dt_days = (obs[-1][0] - obs[-3][0]).total_seconds() / 86400.0
            if dt_days > 0:
                growth = math.log(obs[-1][2] / obs[-3][2]) / dt_days
        elif len(obs) == 2 and obs[-2][2] > 0 and obs[-1][2] > 0:
            dt_days = (obs[-1][0] - obs[-2][0]).total_seconds() / 86400.0
            if dt_days > 0:
                growth = math.log(obs[-1][2] / obs[-2][2]) / dt_days
        growth = max(-2.0, min(2.0, growth))
        games[uid] = {
            "rank": r0,
            "trend": trend,
            "growth": growth,
            "stale_h": stale_h,
            "name": name,
            "seen": len(obs),
        }
    return games, t_last


def select_universe(games, reads):
    """Games on the last chart plus recent top-20 games, capped by rank."""
    last_ids = [r["universe_id"] for r in reads[-1]["up_and_coming"]]
    cand = set(last_ids)
    for m in reads[-3:]:
        for r in m["up_and_coming"]:
            if r.get("rank", 999) <= 20:
                cand.add(r["universe_id"])
    cand = [u for u in cand if u in games]
    cand.sort(key=lambda u: (games[u]["rank"], u))
    return cand[:UNIVERSE_CAP]


def main():
    task, reads = load_history()
    places = int(task.get("places", 10))
    window = task.get("window", {})
    slots = int(window.get("slots", 72))
    games, t_last = build_games(reads)
    universe = select_universe(games, reads)
    if len(universe) < places:
        raise ValueError("fewer than %d games in the universe" % places)

    rng = random.Random(SEED)
    place_counts = {k: {} for k in range(1, places + 1)}
    rank_sum = {u: 0.0 for u in universe}
    n_iter = MC_ITERATIONS
    for _ in range(n_iter):
        slot = rng.randrange(slots)
        h = slot + 0.5
        a, b, sd = transition_params(h)
        decay = 1.0 / (1.0 + h / 24.0)
        sampled = []
        for u in universe:
            g = games[u]
            hh = h + g["stale_h"]
            surv = survival_prob(g["rank"], hh)
            if rng.random() > surv:
                continue
            mu = a + b * g["rank"] + decay * (TREND_COEF * g["trend"] - GROWTH_COEF * g["growth"])
            v = rng.gauss(mu, sd)
            if v < 1.0:
                v = 1.0
            elif v > 200.0:
                v = 200.0
            sampled.append((v, u))
        sampled.sort(key=lambda x: (x[0], x[1]))
        for k in range(min(places, len(sampled))):
            v, u = sampled[k]
            place_counts[k + 1][u] = place_counts[k + 1].get(u, 0) + 1
            rank_sum[u] += k + 1
    # Expected rank: counted top-10 ranks plus 150 for the rest.
    exp_rank = {}
    for u in universe:
        hits = sum(place_counts[k].get(u, 0) for k in range(1, places + 1))
        exp_rank[u] = (rank_sum[u] + 150.0 * (n_iter - hits)) / n_iter

    named_mass = 1.0 - UNNAMED
    probabilities = []
    for k in range(1, places + 1):
        counts = place_counts[k]
        named = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:MAX_NAMED_PER_PLACE]
        total = sum(v for _, v in named)
        if total <= 0:
            continue
        for uid, cnt in named:
            p = (cnt / total) * named_mass
            if p >= 0.0005:
                probabilities.append({"universe_id": int(uid), "place": k, "p": round(p, 6)})

    top10 = sorted(universe, key=lambda u: (exp_rank[u], u))[:places]

    forecast = {"probabilities": probabilities, "top10": [int(u) for u in top10]}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, task.get("output", "forecast.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(forecast, f, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
