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
- Predictions: a ridge-regularized regression per team of points scored on
  that game's box score inputs (pass/rush attempts and yards, turnovers),
  evaluated against the upcoming opponent's season-to-date average allowed
  inputs. Early in a season (before a team has played a handful of games) it
  falls back to the most recently completed season's full game log.
- No betting lines/odds yet — projections only.
