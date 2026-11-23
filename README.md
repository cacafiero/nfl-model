# Fourth & Model

Weekly NFL score projections and power rankings, published to a Claude Artifact.

Replaces a manual weekly workflow of pasting pro-football-reference.com box
scores into a spreadsheet. Data comes from [nflverse](https://github.com/nflverse)
(`nflreadpy`) instead of scraping PFR directly.

## Pipeline

```
python scripts/fetch_data.py [season]   # pulls schedule, team-week stats, drive data
python scripts/rankings.py [season]     # computes offense/defense ranks
python scripts/predict.py [season]      # projects scores for the next unplayed game per team
python scripts/build_site.py [season]   # renders site/dist.html from site/template.html
```

Each defaults to the current NFL season if no argument is given (see
`scripts/common.py::current_season`). `site/dist.html` is then published as a
Claude Artifact.

## Model notes

- Rankings: per-team average-drive stats (scoring %, turnover %, yards/drive)
  from play-by-play, ranked 1–32.
- Predictions (since 2026-11-23): every offense and defense is rated jointly,
  adjusted for opponents, from EPA per play and points scored (weighted
  ridge regression; recent games count more, last season counts less, and
  teams with little data are pulled toward average). Projected points =
  league average + home edge + offense rating + opponent's defense rating.
  See the docstring in `scripts/predict.py` for the backtest numbers.
- Tuning: `python scripts/tune.py tune` grid-searches the settings on 2025;
  `python scripts/tune.py compare` replays 2025 and 2026 against the closing
  lines. `scripts/backtest.py` still replays the full `predict.py` pipeline.
- Before 2026-11-23, predictions came from a per-team regression of points on
  box-score inputs (see git history of `scripts/predict.py`).
- Projections are compared to the market's closing spread and total.
