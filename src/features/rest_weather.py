"""
Rest-differential adjustment: days of rest before a game (already present in
nflverse schedule data as home_rest/away_rest -- no new sourcing needed).
Short week (Thursday game, ~4 days rest) is a modest disadvantage; extra rest
(bye week, ~13-14 days) is a modest advantage. This is a well-known public
signal, so the honest expectation going in is that the market already prices
most of it -- this layer exists to test that empirically rather than assume it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REST_COEF = 0.20        # points of spread shift per day of rest advantage -- modest prior,
                          # deliberately conservative given this is a well-publicized signal
MAX_REST_ADJUSTMENT = 3.0  # cap -- even a bye-week vs short-week extreme (10-day gap) should not
                            # imply more than a few points


def build_rest_adjustment(schedules: pd.DataFrame) -> pd.DataFrame:
    """
    schedules must have home_rest, away_rest already (present in nflverse
    schedule data as fetched by run_full_backtest.fetch_schedules).

    Returns one row per game with rest_adjustment_pts: positive = home team's
    rest advantage should shift the spread MORE toward home (more negative,
    per our standard convention).
    """
    out = schedules[["season", "week", "home_team", "away_team", "home_rest", "away_rest"]].copy()
    out["rest_diff"] = out["home_rest"] - out["away_rest"]
    out["rest_adjustment_pts"] = np.clip(out["rest_diff"] * REST_COEF, -MAX_REST_ADJUSTMENT, MAX_REST_ADJUSTMENT)
    return out


if __name__ == "__main__":
    print("Import build_rest_adjustment(schedules) -- see src/backtest/rest_weather_backtest.py for usage.")
