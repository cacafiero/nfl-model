"""Grade past predictions against actual final scores.

predict.py only ever writes the single upcoming week's games to
predictions_<season>.json, and data/ is gitignored, so nothing keeps a
history of what the model actually projected in past weeks. But every
weekly refresh commits docs/index.html to git, and that file embeds the
week's predictions (projected scores, spread/total lines, picks) as a
SITE_DATA blob -- so git history doubles as a prediction archive.

For each completed game, this script walks the git history of
docs/index.html, finds the most recent snapshot that included a
prediction for that game AND was committed before the game's kickoff
(so later same-week model tweaks don't leak in a change made after the
result was known), and grades that prediction against the actual score:
against-the-spread, over/under, and straight-up (who actually won).

Usage: python results.py [season]
"""
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import common
import polars as pl

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_PATH = "docs/index.html"
SITE_DATA_RE = re.compile(r"const SITE_DATA = (\{.*?\});\s*\n\s*function fmtScore", re.DOTALL)
KICKOFF_TZ = ZoneInfo("America/New_York")  # NFL schedule times are published in ET


def git_history(path: str):
    """[(sha, commit_datetime)], oldest first."""
    out = subprocess.run(
        ["git", "log", "--format=%H|%cI", "--", path],
        cwd=ROOT_DIR, capture_output=True, text=True, check=True,
    ).stdout.strip()
    commits = []
    for line in out.splitlines():
        sha, iso_date = line.split("|", 1)
        commits.append((sha, datetime.fromisoformat(iso_date)))
    commits.reverse()
    return commits


def site_data_at(sha: str, path: str):
    proc = subprocess.run(["git", "show", f"{sha}:{path}"], cwd=ROOT_DIR, capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    m = SITE_DATA_RE.search(proc.stdout)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def collect_predictions_by_game():
    """{game_id: [(commit_datetime, game_dict), ...]} across all of git history,
    sorted oldest-first."""
    by_game: dict = {}
    for sha, commit_dt in git_history(DOCS_PATH):
        data = site_data_at(sha, DOCS_PATH)
        if not data:
            continue
        for game in data.get("games", []):
            by_game.setdefault(game["game_id"], []).append((commit_dt, game))
    return by_game


def kickoff_datetime(gameday: str, gametime: str):
    if not gameday or not gametime:
        return None
    date_part = datetime.fromisoformat(gameday).date()
    hour, minute = (int(x) for x in gametime.split(":"))
    return datetime(date_part.year, date_part.month, date_part.day, hour, minute, tzinfo=KICKOFF_TZ)


def prediction_for(candidates, kickoff):
    """Most recent (commit_dt, game_dict) strictly before kickoff, or None."""
    if kickoff is None:
        return candidates[-1][1] if candidates else None
    before = [c for c in candidates if c[0] < kickoff]
    if not before:
        return None
    return max(before, key=lambda c: c[0])[1]


def grade_side(pick, favored_if_pick_a, actual_diff, line):
    """Generic ATS/O-U grader: `pick` is the side the model chose; returns
    WIN/LOSS/PUSH depending on whether `actual_diff` cleared `line` in the
    direction implied by `favored_if_pick_a` (True if picking side "a" means
    betting the actual_diff will exceed the line)."""
    if pick is None or line is None:
        return None
    cover = (actual_diff - line) if favored_if_pick_a else (line - actual_diff)
    if cover == 0:
        return "PUSH"
    return "WIN" if cover > 0 else "LOSS"


def grade_game(pred: dict, actual_row: dict):
    home, away = pred["home_team"], pred["away_team"]
    home_score, away_score = actual_row["home_score"], actual_row["away_score"]
    actual_margin = home_score - away_score  # positive = home won
    actual_total = home_score + away_score

    pick_spread = pred.get("pick_spread")
    spread_line = pred.get("spread_line")
    ats_result = grade_side(pick_spread, pick_spread == home, actual_margin, spread_line)

    pick_total = pred.get("pick_total")
    total_line = pred.get("total_line")
    total_result = None
    if pick_total is not None and total_line is not None:
        if actual_total == total_line:
            total_result = "PUSH"
        else:
            over_hit = actual_total > total_line
            total_result = "WIN" if (pick_total == "OVER") == over_hit else "LOSS"

    home_proj, away_proj = pred.get("home_projected"), pred.get("away_projected")
    straight_up_result = None
    if home_proj is not None and away_proj is not None and actual_margin != 0:
        projected_winner = home if home_proj >= away_proj else away
        actual_winner = home if actual_margin > 0 else away
        straight_up_result = "WIN" if projected_winner == actual_winner else "LOSS"

    model_margin, model_total = pred.get("model_margin"), pred.get("model_total")

    return {
        "game_id": pred["game_id"],
        "week": pred["week"],
        "gameday": pred.get("gameday"),
        "home_team": home,
        "away_team": away,
        "home_score": home_score,
        "away_score": away_score,
        "home_projected": home_proj,
        "away_projected": away_proj,
        "spread_line": spread_line,
        "total_line": total_line,
        "pick_spread": pick_spread,
        "pick_total": pick_total,
        "ats_result": ats_result,
        "total_result": total_result,
        "straight_up_result": straight_up_result,
        "margin_abs_error": abs(model_margin - actual_margin) if model_margin is not None else None,
        "total_abs_error": abs(model_total - actual_total) if model_total is not None else None,
    }


def record(games: list, key: str):
    wins = sum(1 for g in games if g[key] == "WIN")
    losses = sum(1 for g in games if g[key] == "LOSS")
    pushes = sum(1 for g in games if g[key] == "PUSH")
    decided = wins + losses
    return {
        "wins": wins, "losses": losses, "pushes": pushes,
        "win_pct": round(wins / decided, 3) if decided else None,
    }


def summarize(games: list):
    margin_errors = [g["margin_abs_error"] for g in games if g["margin_abs_error"] is not None]
    total_errors = [g["total_abs_error"] for g in games if g["total_abs_error"] is not None]

    by_week: dict = {}
    for g in games:
        by_week.setdefault(g["week"], []).append(g)

    return {
        "games_graded": len(games),
        "ats": record(games, "ats_result"),
        "total": record(games, "total_result"),
        "straight_up": record(games, "straight_up_result"),
        "avg_margin_abs_error": round(sum(margin_errors) / len(margin_errors), 2) if margin_errors else None,
        "avg_total_abs_error": round(sum(total_errors) / len(total_errors), 2) if total_errors else None,
        "by_week": [
            {
                "week": week,
                "games": len(wgames),
                "ats": record(wgames, "ats_result"),
                "total": record(wgames, "total_result"),
                "straight_up": record(wgames, "straight_up_result"),
            }
            for week, wgames in sorted(by_week.items())
        ],
    }


def main():
    season = int(sys.argv[1]) if len(sys.argv) > 1 else common.current_season()

    schedule_path = os.path.join(common.DATA_DIR, f"schedule_{season}.csv")
    schedule = pl.read_csv(schedule_path).filter(
        (pl.col("game_type") == "REG") & pl.col("home_score").is_not_null()
    )

    predictions_by_game = collect_predictions_by_game()

    graded = []
    for row in schedule.to_dicts():
        candidates = predictions_by_game.get(row["game_id"])
        if not candidates:
            continue
        kickoff = kickoff_datetime(row["gameday"], row["gametime"])
        pred = prediction_for(candidates, kickoff)
        if pred is None:
            continue
        graded.append(grade_game(pred, row))

    graded.sort(key=lambda g: (g["week"], g["gameday"] or ""))

    out = {
        "season": season,
        "games": graded,
        "summary": summarize(graded),
    }
    out_path = os.path.join(common.DATA_DIR, f"results_{season}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Graded {len(graded)} completed games to {out_path}")


if __name__ == "__main__":
    main()
