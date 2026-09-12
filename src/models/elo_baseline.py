"""
Elo baseline model — margin-of-victory-adjusted Elo, updated game by game
in chronological order. This is intentionally simple: it's the sanity-check
baseline that any more complex model (EPA blend, QB-adjusted) must beat in
backtest before it's trusted.

Standard NFL Elo conventions (based on FiveThirtyEight's published methodology,
adapted):
  - K-factor controls how much one game moves a rating
  - MOV multiplier dampens blowout inflation
  - Ratings regress toward league mean between seasons
"""
from __future__ import annotations

import numpy as np
import pandas as pd

INITIAL_ELO = 1500.0
K_FACTOR = 20.0
HOME_FIELD_ELO_PTS = 65.0  # ~ home field advantage expressed in Elo points
SEASON_REGRESSION = 0.33   # fraction reverted toward mean at season start


def elo_win_prob(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def mov_multiplier(point_diff: float, elo_diff: float) -> float:
    """Margin-of-victory multiplier, FiveThirtyEight-style, to prevent
    blowouts from over-inflating rating changes."""
    return np.log(abs(point_diff) + 1) * (2.2 / (elo_diff * 0.001 + 2.2))


def run_elo_backtest(schedules: pd.DataFrame) -> pd.DataFrame:
    """
    schedules must have: season, week, gameday, home_team, away_team,
    home_score, away_score (actual results, for updating ratings after the fact).

    Returns a row per game with pre-game elo ratings for both teams and the
    resulting elo-implied spread — computed BEFORE that game's result is folded in.
    """
    elo = {}  # team -> current rating
    rows = []

    sched = schedules.sort_values(["season", "week", "gameday"]).reset_index(drop=True)

    current_season = None
    for _, g in sched.iterrows():
        season = g["season"]
        home, away = g["home_team"], g["away_team"]

        # regress ratings toward mean at the start of a new season
        if current_season is not None and season != current_season:
            for team in elo:
                elo[team] = INITIAL_ELO + (1 - SEASON_REGRESSION) * (elo[team] - INITIAL_ELO)
        current_season = season

        elo.setdefault(home, INITIAL_ELO)
        elo.setdefault(away, INITIAL_ELO)

        home_elo_adj = elo[home] + HOME_FIELD_ELO_PTS
        away_elo = elo[away]

        # pre-game implied win prob and spread (rough conversion: 25 elo pts ~ 1 point spread)
        win_prob_home = elo_win_prob(home_elo_adj, away_elo)
        elo_diff = home_elo_adj - away_elo
        implied_spread = -elo_diff / 25.0  # negative = home favored, matches standard spread convention

        rows.append(
            {
                "season": season,
                "week": g["week"],
                "home_team": home,
                "away_team": away,
                "pre_game_home_elo": elo[home],
                "pre_game_away_elo": elo[away],
                "elo_implied_spread": implied_spread,
                "elo_win_prob_home": win_prob_home,
            }
        )

        # update ratings post-game if result is known (skip for future/unplayed games)
        if pd.notna(g.get("home_score")) and pd.notna(g.get("away_score")):
            point_diff = g["home_score"] - g["away_score"]
            actual_home = 1.0 if point_diff > 0 else (0.0 if point_diff < 0 else 0.5)
            mult = mov_multiplier(point_diff, elo_diff)
            shift = K_FACTOR * mult * (actual_home - win_prob_home)
            elo[home] += shift
            elo[away] -= shift

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from src.ingestion.nflverse_loader import load_schedules

    sched = load_schedules([2021, 2022, 2023, 2024, 2025])
    result = run_elo_backtest(sched)
    result.to_parquet("data/processed/elo_ratings.parquet")
    print(result.tail(10))
