"""Project scores for upcoming games.

For each team, fits a linear regression of points scored on that game's box
score inputs (pass attempts, pass yards, rush attempts, rush yards,
turnovers), then evaluates it against the upcoming opponent's season-to-date
average allowed inputs to project a score. Mirrors the workbook's per-team
regression + opponent-strength projection, without the tier-bucketing
specifics (see plan notes: this is a clean re-implementation, not a
cell-for-cell port).

Early in a season a team may not yet have enough games to fit a stable
5-variable regression, so this falls back to the most recently completed
season's full game log until the current season has at least MIN_GAMES.

Usage: python predict.py [season]
"""
import json
import os
import sys

import common
import numpy as np
import polars as pl

FEATURES = ["attempts", "passing_yards", "carries", "rushing_yards", "turnovers"]
MIN_GAMES = 4
RIDGE_ALPHA = 1.0  # small L2 penalty for numerical stability, intercept excluded


def load_team_games(season: int) -> pl.DataFrame:
    path = os.path.join(common.DATA_DIR, f"team_week_stats_{season}.csv")
    if not os.path.exists(path):
        return pl.DataFrame()
    df = pl.read_csv(path)
    df = df.with_columns(
        (pl.col("passing_interceptions").fill_null(0) + pl.col("fumbles_lost_total").fill_null(0))
        .alias("turnovers")
    )
    return df


def load_schedule(season: int) -> pl.DataFrame:
    path = os.path.join(common.DATA_DIR, f"schedule_{season}.csv")
    if not os.path.exists(path):
        return pl.DataFrame()
    return pl.read_csv(path)


def points_for(schedule: pl.DataFrame, season: int, week: int, team: str):
    row = schedule.filter(
        (pl.col("week") == week) & ((pl.col("home_team") == team) | (pl.col("away_team") == team))
    )
    if row.height == 0:
        return None
    r = row.row(0, named=True)
    return r["home_score"] if r["home_team"] == team else r["away_score"]


def fit_ridge(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ridge regression with an unpenalized intercept. Returns [intercept, coef...]."""
    n, p = X.shape
    Xa = np.hstack([np.ones((n, 1)), X])
    penalty = np.eye(p + 1) * RIDGE_ALPHA
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(Xa.T @ Xa + penalty, Xa.T @ y)
    return beta


def team_training_rows(games: pl.DataFrame, schedule: pl.DataFrame, team: str, season: int):
    rows = games.filter(pl.col("team") == team)
    X, y = [], []
    for r in rows.to_dicts():
        pts = points_for(schedule, season, r["week"], team)
        if pts is None:
            continue
        X.append([r[f] for f in FEATURES])
        y.append(pts)
    return np.array(X, dtype=float), np.array(y, dtype=float)


def allowed_stats_avg(games: pl.DataFrame, opponent: str):
    rows = games.filter(pl.col("opponent_team") == opponent)
    if rows.height == 0:
        return None
    return np.array([rows[f].mean() for f in FEATURES], dtype=float)


def next_unplayed_game(schedule: pl.DataFrame, team: str):
    rows = schedule.filter(
        ((pl.col("home_team") == team) | (pl.col("away_team") == team))
        & pl.col("home_score").is_null()
    ).sort("week")
    if rows.height == 0:
        return None
    return rows.row(0, named=True)


def main():
    season = int(sys.argv[1]) if len(sys.argv) > 1 else common.current_season()
    prev_season = season - 1

    cur_games = load_team_games(season)
    cur_schedule = load_schedule(season)
    prev_games = load_team_games(prev_season)
    prev_schedule = load_schedule(prev_season)

    teams = sorted(set(cur_schedule["home_team"].to_list()) | set(cur_schedule["away_team"].to_list()))

    models = {}
    allowed = {}
    for team in teams:
        cur_played = cur_games.filter(pl.col("team") == team).height
        if cur_played >= MIN_GAMES:
            X, y = team_training_rows(cur_games, cur_schedule, team, season)
            src_games, src_schedule = cur_games, cur_schedule
        else:
            X, y = team_training_rows(prev_games, prev_schedule, team, prev_season)
            src_games, src_schedule = prev_games, prev_schedule

        if X.shape[0] >= 2:
            beta = fit_ridge(X, y)
        else:
            beta = None  # not enough history anywhere; handled at prediction time

        models[team] = {"beta": beta, "avg_points": float(y.mean()) if len(y) else None}
        allowed[team] = allowed_stats_avg(src_games, team)

    predictions = []
    seen_games = set()
    for team in teams:
        game = next_unplayed_game(cur_schedule, team)
        if game is None or game["game_id"] in seen_games:
            continue
        seen_games.add(game["game_id"])
        home, away = game["home_team"], game["away_team"]

        def project(team_name, opp_name):
            m = models.get(team_name, {})
            opp_allowed = allowed.get(opp_name)
            beta = m.get("beta")
            if beta is None or opp_allowed is None:
                return m.get("avg_points")
            x = np.concatenate([[1.0], opp_allowed])
            return float(beta @ x)

        home_pts = project(home, away)
        away_pts = project(away, home)
        spread_line = game.get("spread_line")  # home-favored margin per the market (positive = home favored)
        total_line = game.get("total_line")

        pick_spread = pick_total = model_margin = model_total = None
        if home_pts is not None and away_pts is not None:
            model_margin = home_pts - away_pts
            model_total = home_pts + away_pts
            if spread_line is not None:
                pick_spread = home if model_margin > spread_line else away
            if total_line is not None:
                pick_total = "OVER" if model_total > total_line else "UNDER"

        predictions.append({
            "game_id": game["game_id"],
            "week": game["week"],
            "gameday": game.get("gameday"),
            "home_team": home,
            "away_team": away,
            "home_projected": home_pts,
            "away_projected": away_pts,
            "spread_line": spread_line,
            "total_line": total_line,
            "home_moneyline": game.get("home_moneyline"),
            "away_moneyline": game.get("away_moneyline"),
            "model_margin": model_margin,
            "model_total": model_total,
            "pick_spread": pick_spread,
            "pick_total": pick_total,
        })

    predictions.sort(key=lambda p: (p["week"], p["gameday"] or ""))
    out_path = os.path.join(common.DATA_DIR, f"predictions_{season}.json")
    with open(out_path, "w") as f:
        json.dump({"season": season, "games": predictions}, f, indent=2)
    print(f"Wrote {len(predictions)} game projections to {out_path}")


if __name__ == "__main__":
    main()
