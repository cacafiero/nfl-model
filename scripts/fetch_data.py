"""Pull schedule, team-week box score stats, and play-by-play drive data from
nflverse (via nflreadpy), replacing the manual weekly copy-paste from
pro-football-reference.com.

Fetches both the target season and the prior season by default, since the
prediction model falls back to last season's full game log early in a new
season (see predict.py).

Usage: python fetch_data.py [season]
"""
import sys
import os

import common  # noqa: E402
import nflreadpy as nfl
import polars as pl


def fetch_one(season: int):
    print(f"Fetching schedule for {season}...")
    schedule = nfl.load_schedules(seasons=[season])
    schedule.write_csv(os.path.join(common.DATA_DIR, f"schedule_{season}.csv"))

    print(f"Fetching team-week stats for {season}...")
    team_stats = nfl.load_team_stats(seasons=[season])
    team_stats.write_csv(os.path.join(common.DATA_DIR, f"team_week_stats_{season}.csv"))

    print(f"Fetching play-by-play for {season} (this can take a minute)...")
    pbp = nfl.load_pbp(seasons=[season])
    pbp = pbp.filter(pbp["posteam"].is_not_null() & pbp["drive"].is_not_null())
    if pbp.height:
        drives = pbp.group_by(["game_id", "posteam", "drive"]).agg([
            pl.first("season"),
            pl.first("week"),
            pl.first("defteam"),
            pl.first("fixed_drive_result"),
            pl.first("drive_play_count"),
            pl.first("drive_time_of_possession"),
            pl.first("drive_start_yard_line"),
            pl.first("drive_ended_with_score"),
            pl.sum("yards_gained").alias("drive_yards"),
        ])
    else:
        drives = pbp
    drives.write_csv(os.path.join(common.DATA_DIR, f"drives_{season}.csv"))

    print(f"Done. season={season} schedule_rows={schedule.height} "
          f"team_stat_rows={team_stats.height} drive_rows={drives.height}")


def fetch_injuries(season: int):
    # Only needed for the current season -- informational display on the
    # site, not a model input (see predict.py's module docstring for why).
    print(f"Fetching injury reports for {season}...")
    injuries = nfl.load_injuries(seasons=[season])
    injuries.write_csv(os.path.join(common.DATA_DIR, f"injuries_{season}.csv"))
    print(f"Done. injury_rows={injuries.height}")


def main():
    season = int(sys.argv[1]) if len(sys.argv) > 1 else common.current_season()
    os.makedirs(common.DATA_DIR, exist_ok=True)
    fetch_one(season)
    fetch_one(season - 1)
    fetch_injuries(season)


if __name__ == "__main__":
    main()
