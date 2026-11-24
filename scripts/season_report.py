"""Replay the EPA model in predict.py over several seasons (only data from
before each week) and check whether its picks beat the market.

Prints, per season: average miss on margin and total vs the closing line,
straight-up, ATS and O/U records. Then ATS and O/U records grouped by how far
the model sits from the line (all picks, OVER only, UNDER only), and how well
the model's win % matched reality. 2025 is the season the settings were tuned
on, so the "ALL" column leaves it out.

Each season needs data for itself and the season before; fetch with
`python fetch_data.py <season>` (it pulls both).

Usage: python season_report.py [season ...]   (default 2021-2026)
"""
import os
import sys

import numpy as np
import predict
import tune

SEASONS = [int(s) for s in sys.argv[1:]] or [2021, 2022, 2023, 2024, 2025, 2026]


def rec(x):
    x = x[x != 0]
    return (int((x > 0).sum()), int((x < 0).sum()))


def pct(w, l):
    return f"{w}-{l} ({100 * w / max(w + l, 1):.0f}%)"


all_rows = {}
for s in SEASONS:
    g = [x for x in tune.replay(s) if x["spread_line"] is not None and x["total_line"] is not None]
    m = np.array([x["home_projected"] - x["away_projected"] for x in g])
    t = np.array([x["home_projected"] + x["away_projected"] for x in g])
    am = np.array([x["home_score"] - x["away_score"] for x in g], float)
    at = np.array([x["home_score"] + x["away_score"] for x in g], float)
    sl = np.array([x["spread_line"] for x in g], float)
    tl = np.array([x["total_line"] for x in g], float)
    wk = np.array([x["week"] for x in g])
    all_rows[s] = dict(m=m, t=t, am=am, at=at, sl=sl, tl=tl, wk=wk)

print("Season  n    MarginMiss(model/line)  TotalMiss(model/line)  Winner   ATS        O/U")
for s, r in all_rows.items():
    ats = rec(np.sign(r["m"] - r["sl"]) * np.sign(r["am"] - r["sl"]))
    ou = rec(np.sign(r["t"] - r["tl"]) * np.sign(r["at"] - r["tl"]))
    su = rec(np.sign(r["m"]) * np.sign(r["am"]))
    tag = " (tuned)" if s == 2025 else ""
    print(f"{s}{tag:<8} {len(r['m']):<4} {np.abs(r['m']-r['am']).mean():5.2f} / {np.abs(r['sl']-r['am']).mean():5.2f}"
          f"           {np.abs(r['t']-r['at']).mean():5.2f} / {np.abs(r['tl']-r['at']).mean():5.2f}"
          f"          {100*su[0]/sum(su):.0f}%     {pct(*ats):<12} {pct(*ou)}")


def bucket_table(title, edge_key, res_fn, buckets):
    print(f"\n{title}")
    hdr = "Edge      " + "".join(f"{s:>14}" for s in all_rows) + "        ALL (ex-2025)"
    print(hdr)
    for lo, hi in buckets:
        line = f"{lo:g}-{hi if hi < 99 else '+':<6}".ljust(10)
        tw = tl_ = 0
        for s, r in all_rows.items():
            e = edge_key(r)
            k = (e >= lo) & (e < hi)
            w, l = rec(res_fn(r)[k])
            if s != 2025:
                tw += w; tl_ += l
            line += f"{pct(w, l):>14}"
        line += f"   {pct(tw, tl_):>14}"
        print(line)


B = [(0, 2), (2, 4), (4, 6), (6, 99)]
bucket_table("SPREAD: ATS by |model margin - line|",
             lambda r: np.abs(r["m"] - r["sl"]),
             lambda r: np.sign(r["m"] - r["sl"]) * np.sign(r["am"] - r["sl"]), B)
bucket_table("TOTALS: O/U by |model total - line|",
             lambda r: np.abs(r["t"] - r["tl"]),
             lambda r: np.sign(r["t"] - r["tl"]) * np.sign(r["at"] - r["tl"]), B)
bucket_table("TOTALS: OVER picks only",
             lambda r: np.where(r["t"] > r["tl"], np.abs(r["t"] - r["tl"]), -1),
             lambda r: np.sign(r["t"] - r["tl"]) * np.sign(r["at"] - r["tl"]), B)
bucket_table("TOTALS: UNDER picks only",
             lambda r: np.where(r["t"] < r["tl"], np.abs(r["t"] - r["tl"]), -1),
             lambda r: np.sign(r["t"] - r["tl"]) * np.sign(r["at"] - r["tl"]), B)

# Win probability calibration across all seasons except the tuning one.
p, w = [], []
for s, r in all_rows.items():
    if s == 2025:
        continue
    keep = r["am"] != 0
    p += [predict.home_win_prob(x) for x in r["m"][keep]]
    w += list((r["am"][keep] > 0).astype(float))
p, w = np.array(p), np.array(w)
fav = np.maximum(p, 1 - p)
won = np.where(p >= 0.5, w, 1 - w)
print("\nWIN % CALIBRATION (all seasons except 2025)")
for lo, hi in [(0.5, 0.55), (0.55, 0.65), (0.65, 0.75), (0.75, 1.01)]:
    k = (fav >= lo) & (fav < hi)
    print(f"  model {lo:.0%}-{min(hi,1):.0%}: n={k.sum():4d} said {fav[k].mean():.0%}, won {won[k].mean():.0%}")
print(f"  Brier {np.mean((p - w) ** 2):.3f} (coin flip 0.250)")
