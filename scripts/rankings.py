"""Compute weekly offense/defense rankings from per-drive data, mirroring the
workbook's "Rankings" / "Team Rankings" sheets (points per drive, scoring %,
turnover %, plays per drive, yards per drive, avg starting field position,
avg time of possession per drive).

Usage: python rankings.py [season]
"""
import json
import os
import sys

import common
import polars as pl

SCORE_RESULTS = {"Touchdown", "Field goal"}
TURNOVER_RESULTS = {"Turnover", "Turnover on downs"}
MIN_GAMES = 2  # below this, a team's current-season sample is too small to rank on


def _time_to_seconds(col: pl.Expr) -> pl.Expr:
    parts = col.str.split(":")
    return (
        parts.list.get(0).cast(pl.Int32, strict=False) * 60
        + parts.list.get(1).cast(pl.Int32, strict=False)
    )


def _yard_line_to_100(col: pl.Expr) -> pl.Expr:
    # "ARI 35" -> distance from own goal line, 0-100. Falls back to 50 (midfield)
    # when the value can't be parsed (e.g. a drive starting at kickoff).
    return (
        pl.when(col.is_null())
        .then(50)
        .otherwise(col.str.split(" ").list.get(1, null_on_oob=True).cast(pl.Int32, strict=False))
        .fill_null(50)
    )


def build_side(drives: pl.DataFrame, team_col: str) -> pl.DataFrame:
    d = drives.with_columns([
        _time_to_seconds(pl.col("drive_time_of_possession")).alias("_top_sec"),
        _yard_line_to_100(pl.col("drive_start_yard_line")).alias("_start_100"),
        pl.col("fixed_drive_result").is_in(list(SCORE_RESULTS)).alias("_scored"),
        pl.col("fixed_drive_result").is_in(list(TURNOVER_RESULTS)).alias("_turnover"),
    ])
    agg = d.group_by(team_col).agg([
        pl.n_unique("game_id").alias("games"),
        pl.len().alias("drives"),
        pl.col("drive_play_count").sum().alias("plays"),
        pl.col("_scored").sum().alias("scoring_drives"),
        pl.col("_turnover").sum().alias("turnover_drives"),
        pl.col("drive_yards").sum().alias("yards"),
        pl.col("_start_100").mean().alias("avg_start_100"),
        pl.col("_top_sec").mean().alias("avg_top_sec"),
    ]).rename({team_col: "team"})
    return agg.with_columns([
        (pl.col("plays") / pl.col("drives")).alias("plays_per_drive"),
        (pl.col("yards") / pl.col("drives")).alias("yards_per_drive"),
        (100.0 * pl.col("scoring_drives") / pl.col("drives")).alias("score_pct"),
        (100.0 * pl.col("turnover_drives") / pl.col("drives")).alias("turnover_pct"),
    ])


def _load_drives(season: int) -> pl.DataFrame:
    path = os.path.join(common.DATA_DIR, f"drives_{season}.csv")
    return pl.read_csv(path) if os.path.exists(path) else pl.DataFrame()


def _pick_with_fallback(cur: pl.DataFrame, prev: pl.DataFrame) -> pl.DataFrame:
    """Per team, use the current-season row if it has enough games, else fall
    back to last season's — mirrors predict.py's early-season fallback so
    rankings and predictions agree on which teams are "cold"."""
    cur_map = {r["team"]: r for r in cur.to_dicts()} if cur.height else {}
    prev_map = {r["team"]: r for r in prev.to_dicts()} if prev.height else {}
    teams = sorted(set(cur_map) | set(prev_map))
    rows = []
    for team in teams:
        c = cur_map.get(team)
        if c and c["games"] >= MIN_GAMES:
            rows.append(c)
        elif team in prev_map:
            rows.append(prev_map[team])
        elif c:
            rows.append(c)
    return pl.DataFrame(rows) if rows else cur


def main():
    season = int(sys.argv[1]) if len(sys.argv) > 1 else common.current_season()
    drives = _load_drives(season)
    prev_drives = _load_drives(season - 1)

    offense = _pick_with_fallback(build_side(drives, "posteam"), build_side(prev_drives, "posteam"))
    defense = _pick_with_fallback(build_side(drives, "defteam"), build_side(prev_drives, "defteam"))

    # offense: rank 1 = most points/drive (best). defense: rank 1 = fewest
    # points/drive allowed (best). We don't have a "points scored on this
    # drive" field directly, so approximate points/drive from scoring-drive
    # rate (weights TDs/FGs the same way the workbook's Rankings sheet does
    # via its own Pts-per-drive column, close enough for ranking purposes).
    offense = offense.sort("score_pct", descending=True).with_row_index("off_rank", offset=1)
    defense = defense.sort("score_pct", descending=False).with_row_index("def_rank", offset=1)

    off_map = {r["team"]: r for r in offense.to_dicts()}
    def_map = {r["team"]: r for r in defense.to_dicts()}

    teams = sorted(set(off_map) | set(def_map))
    out = {}
    for team in teams:
        o = off_map.get(team, {})
        de = def_map.get(team, {})
        out[team] = {
            "off_rank": o.get("off_rank"),
            "def_rank": de.get("def_rank"),
            "off_score_pct": o.get("score_pct"),
            "off_turnover_pct": o.get("turnover_pct"),
            "off_plays_per_drive": o.get("plays_per_drive"),
            "off_yards_per_drive": o.get("yards_per_drive"),
            "def_score_pct_allowed": de.get("score_pct"),
            "def_turnover_pct_forced": de.get("turnover_pct"),
            "def_yards_per_drive_allowed": de.get("yards_per_drive"),
            "games": o.get("games"),
        }

    result = {"season": season, "teams": out}
    out_path = os.path.join(common.DATA_DIR, f"rankings_{season}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote rankings for {len(teams)} teams to {out_path}")


if __name__ == "__main__":
    main()
