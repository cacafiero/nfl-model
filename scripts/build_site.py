"""Combine rankings.json + predictions.json into the site's embedded data
blob and write the final static page.

Usage: python build_site.py [season]
"""
import datetime
import json
import os
import sys

import common
from teams import TEAM_NAMES

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE_DIR = os.path.join(ROOT_DIR, "site")
DOCS_DIR = os.path.join(ROOT_DIR, "docs")


def main():
    season = int(sys.argv[1]) if len(sys.argv) > 1 else common.current_season()

    with open(os.path.join(common.DATA_DIR, f"rankings_{season}.json")) as f:
        rankings = json.load(f)
    with open(os.path.join(common.DATA_DIR, f"predictions_{season}.json")) as f:
        predictions = json.load(f)

    teams = {}
    for abbr, r in rankings["teams"].items():
        teams[abbr] = {"name": TEAM_NAMES.get(abbr, abbr), **r}

    payload = {
        "season": season,
        "updated": datetime.date.today().isoformat(),
        "teams": teams,
        "games": predictions["games"],
    }

    with open(os.path.join(SITE_DIR, "template.html"), encoding="utf-8") as f:
        template = f.read()

    out = template.replace("__SITE_DATA__", json.dumps(payload))

    out_path = os.path.join(SITE_DIR, "dist.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"Wrote {out_path} ({len(out)} bytes)")

    # Also written for GitHub Pages (served from the /docs folder on `main`).
    os.makedirs(DOCS_DIR, exist_ok=True)
    pages_path = os.path.join(DOCS_DIR, "index.html")
    with open(pages_path, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"Wrote {pages_path} ({len(out)} bytes)")


if __name__ == "__main__":
    main()
