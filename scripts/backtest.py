"""Replay predict.py week by week on a past season, using only data that
existed before each week, and save the projections next to the actual scores.

Use it to test any model change out of sample: compare the saved projections
to the closing lines (spread_line / total_line are included per game) before
trusting a change. Needs data/*_<season>.csv and data/*_<season-1>.csv from
`fetch_data.py <season>` (it fetches both).

Usage: python backtest.py <season>
Writes data/backtest_<season>.json
"""
import json
import os
import shutil
import sys
import tempfile

import common
import polars as pl
import predict


def main():
    season = int(sys.argv[1])
    prev = season - 1
    real_dir = common.DATA_DIR
    out_path = os.path.join(real_dir, f"backtest_{season}.json")

    sched = pl.read_csv(os.path.join(real_dir, f"schedule_{season}.csv"))
    stats = pl.read_csv(os.path.join(real_dir, f"team_week_stats_{season}.csv"))
    drives = pl.read_csv(os.path.join(real_dir, f"drives_{season}.csv"))
    reg = sched.filter(pl.col("game_type") == "REG")
    actual = {r["game_id"]: r for r in reg.to_dicts()}

    games_out = []
    for week in sorted(reg["week"].unique().to_list()):
        tmp = tempfile.mkdtemp(prefix="backtest_")
        try:
            for name in ("schedule", "team_week_stats", "drives"):
                shutil.copy(os.path.join(real_dir, f"{name}_{prev}.csv"), os.path.join(tmp, f"{name}_{prev}.csv"))
            hidden = pl.col("week") >= week
            sched.with_columns([
                pl.when(hidden).then(None).otherwise(pl.col("home_score")).alias("home_score"),
                pl.when(hidden).then(None).otherwise(pl.col("away_score")).alias("away_score"),
            ]).write_csv(os.path.join(tmp, f"schedule_{season}.csv"))
            stats.filter(pl.col("week") < week).write_csv(os.path.join(tmp, f"team_week_stats_{season}.csv"))
            drives.filter(pl.col("week") < week).write_csv(os.path.join(tmp, f"drives_{season}.csv"))

            common.DATA_DIR = tmp
            argv, sys.argv = sys.argv, ["predict.py", str(season)]
            try:
                predict.main()
            finally:
                sys.argv = argv
            with open(os.path.join(tmp, f"predictions_{season}.json")) as f:
                preds = json.load(f)["games"]
        finally:
            common.DATA_DIR = real_dir
            shutil.rmtree(tmp, ignore_errors=True)

        kept = 0
        for g in preds:
            row = actual.get(g["game_id"])
            if g["week"] != week or not row or row["home_score"] is None:
                continue
            games_out.append(dict(g, season=season, home_score=row["home_score"], away_score=row["away_score"]))
            kept += 1
        print(f"week {week}: {kept} games", flush=True)

    with open(out_path, "w") as f:
        json.dump(games_out, f)
    print(f"Saved {len(games_out)} games to {out_path}")


if __name__ == "__main__":
    main()
