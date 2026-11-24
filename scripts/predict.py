"""Project scores for each team's next unplayed game.

Replaced the per-team box-score regression on 2026-11-23 (Week 11 MNF). Replayed week
by week, the old model missed the final margin by 10.75 / 10.53 points on
average (2025 / 2026) and this one by 10.28 / 9.81 (Vegas line: 9.72 / 9.43).

Instead of regressing points on a game's own box-score totals, this rates
every team's offense and defense *jointly*, adjusting for who they played:

    stat(team offense in a game) = league avg + off[team] + def[opponent]
                                   + home edge + noise

It solves that once for EPA per play (the most stable public predictor of
future scoring) and once for points scored, with:
  * recency weights (a game's weight halves every HALF_LIFE weeks, and last
    season's games get an extra OFFSEASON_GAP weeks of decay), and
  * ridge shrinkage toward average, so a team with few games or a weird
    stretch is pulled toward the middle instead of trusting the noise.

A team's projected points =
    league avg points + home edge
    + W_EPA     * K * (off_epa[team] + def_epa[opp])      # EPA, in points
    + (1-W_EPA) *     (off_pts[team] + def_pts[opp])      # raw points
where K converts EPA/play into points (fit from the data: about the number
of plays a team runs per game).

The defaults were picked by replaying 2025 week by week (`tune.py tune`)
and then checked on 2026, which the tuning never saw. Neither this model
nor the old one picked against the spread better than a coin flip in those
replays: it is more accurate on scores, not a proven betting edge.

No level adjustment is needed: the replayed totals averaged +0.26 (2025)
and -0.34 (2026) points from actual.

Usage: python predict.py [season]  -> data/predictions_<season>.json
"""
import json
import math
import os
import sys

import common
import numpy as np
import polars as pl

HALF_LIFE = 8.0       # weeks for a game's weight to halve
OFFSEASON_GAP = 8.0   # extra weeks of decay applied to last season's games
LAMBDA_EPA = 2.0      # ridge penalty (in weighted-game units) for EPA ratings
LAMBDA_PTS = 4.0      # ridge penalty for points ratings (points are noisier)
W_EPA = 0.5           # share of the projection that comes from EPA ratings
# Spread of actual margins around the model's margin (2025 replay: 13.0 pts).
# Turns a projected margin into a win chance: P(home wins) = Phi(margin / SD).
# Checked on 2026 games it wasn't fit on: 55-65% picks won 62%, 65-75% won
# 80%, 75%+ won 83%; Brier 0.218 vs 0.250 for a coin flip.
MARGIN_SD = 13.0
# "Under watch": games where the model's total is 2-4 points BELOW the posted
# total. Replaying 2021-2026 these UNDER picks went 102-77 (57%), but the
# pattern was found by slicing results many ways, so it is tracked live from
# 2026 Week 12 on (results.py) before anyone treats it as an edge.
UNDER_WATCH_MIN, UNDER_WATCH_MAX = 2.0, 4.0


def home_win_prob(margin: float) -> float:
    return 0.5 * (1.0 + math.erf(margin / (MARGIN_SD * math.sqrt(2.0))))


def load_rows(seasons) -> pl.DataFrame:
    """One row per team per regular-season game: offensive EPA/play, points
    scored, opponent, and home/neutral context."""
    frames = []
    for s in seasons:
        stats_path = os.path.join(common.DATA_DIR, f"team_week_stats_{s}.csv")
        sched_path = os.path.join(common.DATA_DIR, f"schedule_{s}.csv")
        if not (os.path.exists(stats_path) and os.path.exists(sched_path)):
            continue
        st = pl.read_csv(stats_path, infer_schema_length=10000).filter(pl.col("season_type") == "REG")
        sc = pl.read_csv(sched_path, infer_schema_length=10000).filter(pl.col("game_type") == "REG")
        sc = sc.filter(pl.col("home_score").is_not_null())
        home = sc.select([
            "season", "week", "game_id", pl.col("home_team").alias("team"), pl.col("away_team").alias("opp"),
            pl.col("home_score").alias("points"),
            pl.when(pl.col("location") == "Neutral").then(0.0).otherwise(1.0).alias("home"),
        ])
        away = sc.select([
            "season", "week", "game_id", pl.col("away_team").alias("team"), pl.col("home_team").alias("opp"),
            pl.col("away_score").alias("points"),
            pl.when(pl.col("location") == "Neutral").then(0.0).otherwise(-1.0).alias("home"),
        ])
        games = pl.concat([home, away])
        st = st.with_columns(
            ((pl.col("passing_epa").fill_null(0) + pl.col("rushing_epa").fill_null(0))
             / (pl.col("attempts").fill_null(0) + pl.col("sacks_suffered").fill_null(0)
                + pl.col("carries").fill_null(0))).alias("epa_play")
        ).select(["season", "week", "team", "epa_play"])
        frames.append(games.join(st, on=["season", "week", "team"], how="inner"))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames).with_columns(pl.col("points").cast(pl.Float64))


def fit_ratings(rows: pl.DataFrame, target: str, weights: np.ndarray, teams: list, lam: float):
    """Weighted ridge: target = mu + off[team] + def[opp] + h*home.
    `home` is +1 home / -1 away / 0 neutral, so h is half the home edge.
    Returns (mu, h, off dict, def dict)."""
    idx = {t: i for i, t in enumerate(teams)}
    n, T = rows.height, len(teams)
    X = np.zeros((n, 2 + 2 * T))
    X[:, 0] = 1.0
    X[:, 1] = rows["home"].to_numpy()
    for r, (t, o) in enumerate(zip(rows["team"].to_list(), rows["opp"].to_list())):
        X[r, 2 + idx[t]] = 1.0
        X[r, 2 + T + idx[o]] = 1.0
    y = rows[target].to_numpy().astype(float)
    W = weights
    penalty = np.full(2 + 2 * T, lam)
    penalty[:2] = 0.0
    A = X.T @ (X * W[:, None]) + np.diag(penalty)
    b = X.T @ (y * W)
    beta = np.linalg.solve(A, b)
    off = {t: beta[2 + i] for t, i in idx.items()}
    dfn = {t: beta[2 + T + i] for t, i in idx.items()}
    return beta[0], beta[1], off, dfn


def game_weights(rows: pl.DataFrame, season: int, week: int, half_life: float, gap: float) -> np.ndarray:
    weeks_ago = (week - rows["week"].to_numpy()).astype(float)
    prior = rows["season"].to_numpy() < season
    # A prior-season game: weeks remaining in that season + this season's weeks + the offseason gap.
    weeks_ago = np.where(prior, (19 - rows["week"].to_numpy()) + (week - 1) + gap, weeks_ago)
    return 0.5 ** (weeks_ago / half_life)


class Model:
    """Ratings fit on every completed game before (season, week)."""

    def __init__(self, rows: pl.DataFrame, season: int, week: int, half_life=HALF_LIFE,
                 gap=OFFSEASON_GAP, lam_epa=LAMBDA_EPA, lam_pts=LAMBDA_PTS, w_epa=W_EPA):
        train = rows.filter((pl.col("season") < season)
                            | ((pl.col("season") == season) & (pl.col("week") < week)))
        train = train.filter(pl.col("season") >= season - 1)
        self.teams = sorted(set(train["team"].to_list()) | set(train["opp"].to_list()))
        w = game_weights(train, season, week, half_life, gap)
        self.mu_e, self.h_e, self.off_e, self.def_e = fit_ratings(train, "epa_play", w, self.teams, lam_epa)
        self.mu_p, self.h_p, self.off_p, self.def_p = fit_ratings(train, "points", w, self.teams, lam_pts)
        # Points per unit of EPA/play, from game-level data (~plays per game).
        e = train["epa_play"].to_numpy()
        p = train["points"].to_numpy()
        self.k = float(np.cov(e, p, aweights=w)[0, 1] / np.cov(e, aweights=w))
        self.w_epa = w_epa

    def points(self, team: str, opp: str, home: float) -> float:
        epa_part = self.k * (self.off_e.get(team, 0.0) + self.def_e.get(opp, 0.0))
        pts_part = self.off_p.get(team, 0.0) + self.def_p.get(opp, 0.0)
        return self.mu_p + self.h_p * home + self.w_epa * epa_part + (1 - self.w_epa) * pts_part

    def project(self, game: dict):
        neutral = game.get("location") == "Neutral"
        hs = 0.0 if neutral else 1.0
        home_pts = self.points(game["home_team"], game["away_team"], hs)
        away_pts = self.points(game["away_team"], game["home_team"], -hs)
        return home_pts, away_pts


def main():
    season = int(sys.argv[1]) if len(sys.argv) > 1 else common.current_season()
    rows = load_rows([season - 1, season])
    sched = pl.read_csv(os.path.join(common.DATA_DIR, f"schedule_{season}.csv"), infer_schema_length=10000)
    upcoming = sched.filter((pl.col("game_type") == "REG") & pl.col("home_score").is_null()).sort("week")
    out = []
    if upcoming.height:
        # Each team's next unplayed game (so a team on bye still shows its
        # next one). Ratings use every completed game, which is all known
        # data as of the earliest unplayed week.
        model = Model(rows, season, upcoming["week"].min())
        seen = set()
        for g in upcoming.to_dicts():
            if g["home_team"] in seen or g["away_team"] in seen:
                seen.update((g["home_team"], g["away_team"]))
                continue
            seen.update((g["home_team"], g["away_team"]))
            h, a = model.project(g)
            spread_line, total_line = g.get("spread_line"), g.get("total_line")
            out.append({
                "game_id": g["game_id"], "week": g["week"], "gameday": g.get("gameday"),
                "home_team": g["home_team"], "away_team": g["away_team"],
                "home_projected": h, "away_projected": a,
                "spread_line": spread_line, "total_line": total_line,
                "home_moneyline": g.get("home_moneyline"), "away_moneyline": g.get("away_moneyline"),
                "model_margin": h - a, "model_total": h + a,
                "pick_spread": None if spread_line is None else (g["home_team"] if h - a > spread_line else g["away_team"]),
                "pick_total": None if total_line is None else ("OVER" if h + a > total_line else "UNDER"),
                "home_win_prob": home_win_prob(h - a),
                "under_watch": bool(total_line is not None
                                    and UNDER_WATCH_MIN <= total_line - (h + a) < UNDER_WATCH_MAX),
                "model": "epa",
            })
    out.sort(key=lambda p: (p["week"], p["gameday"] or ""))
    path = os.path.join(common.DATA_DIR, f"predictions_{season}.json")
    with open(path, "w") as f:
        json.dump({"season": season, "games": out}, f, indent=2)
    for p in out:
        line = p["spread_line"]
        print(f"wk{p['week']:>2} {p['away_team']:>3} {p['away_projected']:5.1f} @ {p['home_team']:<3} "
              f"{p['home_projected']:5.1f}   model margin {p['model_margin']:+5.1f}  line {line if line is not None else '-'}")
    print(f"Wrote {len(out)} game projections to {path}")


if __name__ == "__main__":
    main()
