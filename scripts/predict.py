"""Project scores for upcoming games.

For each team, fits a linear regression of points scored on that game's box
score inputs (pass attempts, pass yards, rush attempts, rush yards,
turnovers), then evaluates it against the upcoming opponent's average
allowed inputs to project a score. Mirrors the workbook's per-team
regression + opponent-strength projection, without the tier-bucketing
specifics (see plan notes: this is a clean re-implementation, not a
cell-for-cell port).

A team's own scoring regression is fit on its full available history (up
to two regular seasons, pulling from the prior season early in a new
one) rather than just the last few games: 5 features plus an intercept
need enough observations to fit stably, and a short window leaves the
regression underdetermined (near-perfect but meaningless fit, wild
coefficients that blow up when evaluated against a different opponent).

An opponent's allowed-stats average (the input evaluated against a team's
fitted regression) is a different, deliberately short WINDOW: a WEIGHTED
average of the opponent's last WINDOW games, weighting each by how good
that opponent's defense ranked *as of that week* (a point-in-time rank
computed from cumulative drive data through that week, not today's
season-to-date rank), so a game from a stretch where they were playing
elite defense counts more than one from a stretch where they weren't.

Usage: python predict.py [season]
"""
import json
import os
import sys

import common
import numpy as np
import polars as pl
import rankings

FEATURES = ["attempts", "passing_yards", "carries", "rushing_yards", "turnovers"]
WINDOW = 6  # recent games used for the opponent's rank-weighted allowed-stats average
TRAIN_WINDOW = 17  # cap at one full regular season's worth of games
RIDGE_ALPHA = 25.0  # L2 penalty; shrinks per-team coefficients so a small, noisy
# training window can't produce swings like -6 points per turnover that blow up
# when evaluated against a different opponent's stat profile
WINSORIZE_K = 2.0  # cap allowed-stats outliers beyond median +/- K * MAD


def load_team_games(season: int) -> pl.DataFrame:
    path = os.path.join(common.DATA_DIR, f"team_week_stats_{season}.csv")
    if not os.path.exists(path):
        return pl.DataFrame()
    df = pl.read_csv(path)
    df = df.filter(pl.col("season_type") == "REG")
    df = df.with_columns(
        (pl.col("passing_interceptions").fill_null(0) + pl.col("fumbles_lost_total").fill_null(0))
        .alias("turnovers")
    )
    return df


def load_schedule(season: int) -> pl.DataFrame:
    path = os.path.join(common.DATA_DIR, f"schedule_{season}.csv")
    if not os.path.exists(path):
        return pl.DataFrame()
    return pl.read_csv(path).filter(pl.col("game_type") == "REG")


def points_for(schedule: pl.DataFrame, season: int, week: int, team: str):
    row = schedule.filter(
        (pl.col("week") == week) & ((pl.col("home_team") == team) | (pl.col("away_team") == team))
    )
    if row.height == 0:
        return None
    r = row.row(0, named=True)
    return r["home_score"] if r["home_team"] == team else r["away_score"]


def fit_ridge(X: np.ndarray, y: np.ndarray, scaler=None, prior_std: np.ndarray = None) -> np.ndarray:
    """Ridge regression with an unpenalized intercept. Returns raw-scale
    [intercept, coef...] (so callers evaluate it against raw allowed-stats
    inputs unchanged), but fits internally on *standardized* features when
    `scaler` (mean, std) is given, shrinking toward `prior_std` (also in
    standardized units) instead of toward zero.

    Why standardize: `attempts`/`carries` (small scale, std ~7-8) and their
    own `passing_yards`/`rushing_yards` (std ~50-70) are highly correlated
    pairs. An unstandardized ridge penalty — a flat sum(beta^2) — falls
    unevenly across differently-scaled features, so with only ~17 games per
    team the fit can land on an arbitrary, extreme split within a
    correlated pair (e.g. all the rushing signal loaded onto "carries" with
    a huge coefficient, sign flipping team to team) that a scale-comparable
    penalty on standardized features prevents. Shrinking toward the
    league-wide fit (instead of zero) then means a team's coefficients can
    only deviate from that sane baseline as far as its own data supports.
    """
    n, p = X.shape
    if scaler is not None:
        mean, std = scaler
        Xs = (X - mean) / std
    else:
        Xs = X
    Xa = np.hstack([np.ones((n, 1)), Xs])
    penalty = np.eye(p + 1) * RIDGE_ALPHA
    penalty[0, 0] = 0.0
    target = np.zeros(p + 1) if prior_std is None else np.asarray(prior_std)
    beta_std = np.linalg.solve(Xa.T @ Xa + penalty, Xa.T @ y + penalty @ target)
    if scaler is None:
        return beta_std
    # Convert standardized-space [intercept, coefs] back to raw-scale.
    coef = beta_std[1:] / std
    intercept = beta_std[0] - float(coef @ mean)
    return np.concatenate([[intercept], coef])


def r_squared(X: np.ndarray, y: np.ndarray, beta: np.ndarray):
    """Fraction of variance in points scored explained by the fitted model
    (the correlation between fitted and actual values, squared)."""
    if len(y) < 2 or np.std(y) == 0:
        return None
    Xa = np.hstack([np.ones((X.shape[0], 1)), X])
    y_hat = Xa @ beta
    if np.std(y_hat) == 0:
        return None
    return float(np.corrcoef(y, y_hat)[0, 1] ** 2)


def recent_rows(pool: pl.DataFrame, team_col: str, team: str, window: int = WINDOW) -> pl.DataFrame:
    """The `window` most recent regular-season rows for `team`, newest first,
    drawn from the combined current+prior season pool."""
    if pool.height == 0:
        return pool
    return pool.filter(pl.col(team_col) == team).sort(["season", "week"], descending=True).head(window)


def team_training_rows(pool: pl.DataFrame, schedules: dict, team: str):
    rows = recent_rows(pool, "team", team, window=TRAIN_WINDOW)
    X, y = [], []
    for r in rows.to_dicts():
        schedule = schedules.get(r["season"])
        if schedule is None:
            continue
        pts = points_for(schedule, r["season"], r["week"], team)
        if pts is None:
            continue
        X.append([r[f] for f in FEATURES])
        y.append(pts)
    return np.array(X, dtype=float), np.array(y, dtype=float)


def weekly_defense_ranks(season: int, schedule: pl.DataFrame) -> dict:
    """{week: {team: def_rank}}, where def_rank for week w is computed from
    cumulative REG-season drives through week w only (point-in-time), not
    the full-season number. def_rank 1 = fewest points allowed per drive."""
    path = os.path.join(common.DATA_DIR, f"drives_{season}.csv")
    if not os.path.exists(path) or schedule.height == 0:
        return {}
    drives = pl.read_csv(path)
    if drives.height == 0:
        return {}
    reg_game_ids = set(schedule["game_id"].to_list())
    drives = drives.filter(pl.col("game_id").is_in(reg_game_ids))
    if drives.height == 0:
        return {}
    out = {}
    for w in sorted(drives["week"].unique().to_list()):
        defense = rankings.build_side(drives.filter(pl.col("week") <= w), "defteam")
        if defense.height == 0:
            continue
        defense = defense.sort("score_pct", descending=False).with_row_index("def_rank", offset=1)
        out[w] = {r["team"]: r["def_rank"] for r in defense.to_dicts()}
    return out


def winsorize(values: np.ndarray, k: float = WINSORIZE_K) -> np.ndarray:
    """Cap values beyond median +/- k*MAD so a single outlier game (e.g. one
    freak 6-turnover performance) can't dominate a small-window average."""
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        return values
    return np.clip(values, median - k * mad, median + k * mad)


def allowed_stats_avg(pool: pl.DataFrame, opponent: str, week_ranks: dict):
    """Weighted average of `opponent`'s last WINDOW allowed-stat games,
    weighting each game by the opponent's own point-in-time defensive rank
    for that week (falling back to a neutral mid-pack rank when unknown).
    def_rank 1 = best defense, so the weight inverts it (33 - rank) to give
    the strongest defensive weeks the most weight. Each feature is
    winsorized first so one outlier game can't dominate the average."""
    rows = recent_rows(pool, "opponent_team", opponent)
    if rows.height == 0:
        return None
    weights = np.array([
        33.0 - float(week_ranks.get(r["season"], {}).get(r["week"], {}).get(opponent, 16.5))
        for r in rows.to_dicts()
    ], dtype=float)
    return np.array(
        [float(np.sum(winsorize(np.array(rows[f].to_list(), dtype=float)) * weights) / weights.sum()) for f in FEATURES],
        dtype=float,
    )


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

    schedules = {season: cur_schedule, prev_season: prev_schedule}
    pool_frames = [g for g in (cur_games, prev_games) if g.height]
    pool = pl.concat(pool_frames, how="diagonal_relaxed") if pool_frames else cur_games

    week_ranks = {
        season: weekly_defense_ranks(season, cur_schedule),
        prev_season: weekly_defense_ranks(prev_season, prev_schedule),
    }

    teams = sorted(set(cur_schedule["home_team"].to_list()) | set(cur_schedule["away_team"].to_list()))

    # League-average regression: same 5 inputs, fit on every team's games
    # pooled together (~500+ rows, well-conditioned), in standardized units
    # so its coefficients are on the same scale as each team's fit below.
    # Used as the shrinkage target instead of zero.
    all_X, all_y = [], []
    for team in teams:
        X, y = team_training_rows(pool, schedules, team)
        if X.shape[0]:
            all_X.append(X)
            all_y.append(y)
    if all_X:
        Xall = np.vstack(all_X)
        scaler = (Xall.mean(axis=0), Xall.std(axis=0))
        Xall_std = (Xall - scaler[0]) / scaler[1]
        n = Xall_std.shape[0]
        Xa_all = np.hstack([np.ones((n, 1)), Xall_std])
        # Plain (unshrunk) standardized fit — this becomes the shrinkage target.
        league_prior_std = np.linalg.solve(Xa_all.T @ Xa_all, Xa_all.T @ np.concatenate(all_y))
    else:
        scaler = None
        league_prior_std = None

    models = {}
    allowed = {}
    for team in teams:
        X, y = team_training_rows(pool, schedules, team)

        if X.shape[0] >= 2:
            beta = fit_ridge(X, y, scaler=scaler, prior_std=league_prior_std)
            r2 = r_squared(X, y, beta)
        else:
            beta = None  # not enough history anywhere; handled at prediction time
            r2 = None

        models[team] = {"beta": beta, "avg_points": float(y.mean()) if len(y) else None, "r_squared": r2}
        allowed[team] = allowed_stats_avg(pool, team, week_ranks)

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
            "home_r_squared": models.get(home, {}).get("r_squared"),
            "away_r_squared": models.get(away, {}).get("r_squared"),
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
