"""Replay the EPA model in predict.py week by week (only data from before each week) and
score it against the closing lines.

  python tune.py tune         # grid-search settings on 2025 only
  python tune.py compare   # 2025 + 2026 vs the line (and a saved backtest run)

`compare` also shows data/backtest_<season>.json if present (a saved
backtest.py run, e.g. of an older model version) on the same games.
"""
import itertools
import json
import os
import sys

import common
import numpy as np
import polars as pl
import predict


def replay(season: int, **params):
    rows = predict.load_rows([season - 1, season])
    sched = pl.read_csv(os.path.join(common.DATA_DIR, f"schedule_{season}.csv"), infer_schema_length=10000)
    played = sched.filter((pl.col("game_type") == "REG") & pl.col("home_score").is_not_null())
    out = []
    for week in sorted(played["week"].unique().to_list()):
        model = predict.Model(rows, season, week, **params)
        for g in played.filter(pl.col("week") == week).to_dicts():
            h, a = model.project(g)
            out.append({"game_id": g["game_id"], "week": week, "home_projected": h, "away_projected": a,
                        "home_score": g["home_score"], "away_score": g["away_score"],
                        "spread_line": g["spread_line"], "total_line": g["total_line"]})
    return out


def score(games, min_week=1):
    g = [x for x in games if x["week"] >= min_week and x.get("spread_line") is not None
         and x.get("total_line") is not None and x.get("home_projected") is not None]
    hm = np.array([x["home_projected"] - x["away_projected"] for x in g])
    mt = np.array([x["home_projected"] + x["away_projected"] for x in g])
    am = np.array([x["home_score"] - x["away_score"] for x in g], dtype=float)
    at = np.array([x["home_score"] + x["away_score"] for x in g], dtype=float)
    sl = np.array([x["spread_line"] for x in g], dtype=float)
    tl = np.array([x["total_line"] for x in g], dtype=float)

    def record(model, line, actual):
        side = np.sign(model - line) * np.sign(actual - line)
        w, l = int((side > 0).sum()), int((side < 0).sum())
        return w, l

    ats_w, ats_l = record(hm, sl, am)
    tot_w, tot_l = record(mt, tl, at)
    big = np.abs(hm - sl) >= 3
    big_w, big_l = record(hm[big], sl[big], am[big])
    return {
        "games": len(g),
        "margin_mae": float(np.abs(hm - am).mean()), "line_margin_mae": float(np.abs(sl - am).mean()),
        "total_mae": float(np.abs(mt - at).mean()), "line_total_mae": float(np.abs(tl - at).mean()),
        "margin_corr": float(np.corrcoef(hm, am)[0, 1]), "line_corr": float(np.corrcoef(sl, am)[0, 1]),
        "ats": f"{ats_w}-{ats_l}", "ats_pct": ats_w / max(ats_w + ats_l, 1),
        "ats_edge3": f"{big_w}-{big_l}", "totals": f"{tot_w}-{tot_l}", "tot_pct": tot_w / max(tot_w + tot_l, 1),
    }


GRID = {
    "half_life": [4.0, 8.0, 16.0],
    "gap": [4.0, 8.0, 16.0],
    "lam_epa": [2.0, 4.0, 8.0],
    "lam_pts": [4.0, 8.0, 16.0],
    "w_epa": [0.25, 0.5, 0.75, 1.0],
}


def tune():
    results = []
    keys = list(GRID)
    for combo in itertools.product(*GRID.values()):
        params = dict(zip(keys, combo))
        s = score(replay(2025, **params))
        results.append((s["margin_mae"] + s["total_mae"], params, s))
    results.sort(key=lambda r: r[0])
    for err, params, s in results[:10]:
        print(f"{err:6.2f}  margin {s['margin_mae']:.2f}  total {s['total_mae']:.2f}  {params}")


def fmt(name, s):
    return (f"{name:<16} n={s['games']:<4} margin MAE {s['margin_mae']:5.2f}  total MAE {s['total_mae']:5.2f}  "
            f"corr {s['margin_corr']:.2f}  ATS {s['ats']} ({s['ats_pct']:.1%})  edge>=3 {s['ats_edge3']}  "
            f"totals {s['totals']} ({s['tot_pct']:.1%})")


def compare():
    for season in (2025, 2026):
        new = replay(season)
        path = os.path.join(common.DATA_DIR, f"backtest_{season}.json")
        old = json.load(open(path)) if os.path.exists(path) else []
        ids = {g["game_id"] for g in old} if old else {g["game_id"] for g in new}
        new = [g for g in new if g["game_id"] in ids]
        old_ids = {g["game_id"] for g in new}
        old = [g for g in old if g["game_id"] in old_ids]
        s_new = score(new)
        print(f"\n== {season} ==")
        print(f"{'Vegas line':<16} n={s_new['games']:<4} margin MAE {s_new['line_margin_mae']:5.2f}  "
              f"total MAE {s_new['line_total_mae']:5.2f}  corr {s_new['line_corr']:.2f}")
        if old:
            print(fmt("Saved backtest", score(old)))
        print(fmt("Model", s_new))


if __name__ == "__main__":
    {"tune": tune, "compare": compare}[sys.argv[1]]()
