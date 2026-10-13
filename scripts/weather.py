"""Fetch kickoff-window weather forecasts for upcoming games (display only).

Not a model input: the site shows the forecast next to each matchup so a
reader can weigh wind/cold/rain themselves. Uses Open-Meteo (no API key;
free tier is for non-commercial use). Dome and closed-roof games are marked
indoors without a forecast call; neutral-site games use the venue coordinates
below when the stadium name is recognized, otherwise they're skipped.

A failed or missing forecast never blocks the pipeline: that game simply
shows no weather.

Usage: python weather.py [season]
"""
import datetime
import json
import os
import sys

import common
import polars as pl
import requests

API_URL = "https://api.open-meteo.com/v1/forecast"
MAX_DAYS_AHEAD = 15  # Open-Meteo forecasts reach 16 days
GAME_HOURS = 3  # window length covered after kickoff

STADIUMS = {
    "ARI": (33.5276, -112.2626), "ATL": (33.7554, -84.4010), "BAL": (39.2780, -76.6227),
    "BUF": (42.7738, -78.7870), "CAR": (35.2258, -80.8528), "CHI": (41.8623, -87.6167),
    "CIN": (39.0955, -84.5161), "CLE": (41.5061, -81.6995), "DAL": (32.7473, -97.0945),
    "DEN": (39.7439, -105.0201), "DET": (42.3400, -83.0456), "GB": (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107), "IND": (39.7601, -86.1639), "JAX": (30.3239, -81.6373),
    "KC": (39.0489, -94.4839), "LA": (33.9535, -118.3392), "LAC": (33.9535, -118.3392),
    "LV": (36.0909, -115.1833), "MIA": (25.9580, -80.2389), "MIN": (44.9738, -93.2575),
    "NE": (42.0909, -71.2643), "NO": (29.9511, -90.0812), "NYG": (40.8135, -74.0745),
    "NYJ": (40.8135, -74.0745), "PHI": (39.9008, -75.1675), "PIT": (40.4468, -80.0158),
    "SEA": (47.5952, -122.3316), "SF": (37.4030, -121.9700), "TB": (27.9759, -82.5033),
    "TEN": (36.1665, -86.7713), "WAS": (38.9076, -76.8645),
}

# Matched as lowercase substrings of the schedule's stadium name.
NEUTRAL_VENUES = {
    "wembley": (51.5560, -0.2796),
    "tottenham": (51.6043, -0.0662),
    "allianz": (48.2188, 11.6247),
    "bernab": (40.4531, -3.6883),
    "azteca": (19.3029, -99.1505),
    "banorte": (19.3029, -99.1505),
}


def venue_coords(row: dict):
    if row.get("location") == "Neutral":
        name = (row.get("stadium") or "").lower()
        for key, coords in NEUTRAL_VENUES.items():
            if key in name:
                return coords
        return None
    return STADIUMS.get(row["home_team"])


def fetch_forecast(lat: float, lon: float, gameday: str, gametime: str):
    """Kickoff temperature plus worst wind/rain over the game window, or None."""
    day = datetime.date.fromisoformat(gameday)
    resp = requests.get(API_URL, params={
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,wind_speed_10m,wind_gusts_10m,precipitation_probability",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        # schedule kickoff times are Eastern, so ask for Eastern timestamps
        "timezone": "America/New_York",
        "start_date": day.isoformat(),
        "end_date": (day + datetime.timedelta(days=1)).isoformat(),
    }, timeout=20)
    resp.raise_for_status()
    hourly = resp.json()["hourly"]

    kickoff = f"{gameday}T{gametime[:2]}:00"
    if kickoff not in hourly["time"]:
        return None
    i = hourly["time"].index(kickoff)
    window = range(i, min(i + GAME_HOURS + 1, len(hourly["time"])))

    def worst(key):
        vals = [hourly[key][j] for j in window if hourly[key][j] is not None]
        return max(vals) if vals else None

    temp = hourly["temperature_2m"][i]
    if temp is None:
        return None
    return {
        "temp_f": round(temp),
        "wind_mph": round(worst("wind_speed_10m")) if worst("wind_speed_10m") is not None else None,
        "gust_mph": round(worst("wind_gusts_10m")) if worst("wind_gusts_10m") is not None else None,
        "precip_pct": round(worst("precipitation_probability")) if worst("precipitation_probability") is not None else None,
    }


def main():
    season = int(sys.argv[1]) if len(sys.argv) > 1 else common.current_season()
    with open(os.path.join(common.DATA_DIR, f"predictions_{season}.json")) as f:
        game_ids = [g["game_id"] for g in json.load(f)["games"]]
    schedule = pl.read_csv(os.path.join(common.DATA_DIR, f"schedule_{season}.csv"))
    rows = {r["game_id"]: r for r in schedule.filter(pl.col("game_id").is_in(game_ids)).to_dicts()}

    today = datetime.date.today()
    out = {}
    for game_id in game_ids:
        row = rows.get(game_id)
        if not row:
            continue
        roof = row.get("roof")
        if roof in ("dome", "closed"):
            out[game_id] = {"kind": "indoor"}
            continue
        days_ahead = (datetime.date.fromisoformat(row["gameday"]) - today).days
        coords = venue_coords(row)
        if coords is None or not row.get("gametime") or days_ahead < 0 or days_ahead > MAX_DAYS_AHEAD:
            continue
        try:
            forecast = fetch_forecast(*coords, row["gameday"], row["gametime"])
        except Exception as exc:
            print(f"  no forecast for {game_id}: {exc}")
            continue
        if forecast:
            out[game_id] = {"kind": "forecast", "roof_tbd": roof is None, **forecast}

    path = os.path.join(common.DATA_DIR, f"weather_{season}.json")
    with open(path, "w") as f:
        json.dump({"season": season, "fetched": today.isoformat(), "games": out}, f, indent=2)
    n_fc = sum(1 for v in out.values() if v["kind"] == "forecast")
    print(f"Wrote weather for {len(out)} games ({n_fc} forecasts) to {path}")


if __name__ == "__main__":
    main()
