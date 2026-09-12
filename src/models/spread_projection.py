"""
Merges independent rating sources (Elo, EPA, DVOA) into game-level rows and
produces a final projected spread. Also exposes each source's STANDALONE
implied spread so you can backtest them separately before deciding how (or
whether) to blend them.

Point conversion conventions used throughout (all "home perspective", negative = home favored,
matching standard American spread notation, e.g. -3.5 means home favored by 3.5):

  Elo:   implied_spread = -(home_elo_adj - away_elo) / 25       (~25 elo pts per point, standard convention)
  EPA:   implied_spread = -((home_off - away_def) - (away_off - home_def)) * PLAYS_PER_GAME * EPA_TO_POINTS
         i.e. net expected-points edge per play, scaled by a typical game's play volume
  DVOA:  implied_spread = -(home_total_dvoa - away_total_dvoa) * DVOA_TO_POINTS
         DVOA is expressed in % above/below average; needs its own empirical calibration
         constant since DVOA points-per-percent isn't a settled public number — the harness
         will help you fit DVOA_TO_POINTS from your own backtest residuals rather than guessing.

All scaling constants below are STARTING PRIORS. The whole point of the backtest
is to refit them (or at minimum sanity-check them) against 5 seasons of actual
closing lines and results — don't treat these as final.
"""
from __future__ import annotations

import pandas as pd

HOME_FIELD_PTS = 1.8          # starting prior, recalibrate from backtest residuals
PLAYS_PER_GAME = 65.0         # rough league-average offensive plays/game, used to scale EPA/play to points
EPA_TO_POINTS = 1.0           # EPA is already in units of points; this is a no-op placeholder for tuning
DVOA_TO_POINTS = 0.35         # STARTING GUESS — e.g. a 20% DVOA edge -> ~7 pts. MUST be refit empirically.


def build_elo_spread(elo_df: pd.DataFrame) -> pd.DataFrame:
    """elo_df comes straight from src/models/elo_baseline.py output."""
    out = elo_df.copy()
    out["elo_spread"] = out["elo_implied_spread"]
    return out[["season", "week", "home_team", "away_team", "elo_spread"]]


def build_epa_spread(games: pd.DataFrame, epa_ratings: pd.DataFrame) -> pd.DataFrame:
    """
    games: season, week, home_team, away_team
    epa_ratings: output of src/features/epa_ratings.py build_rolling_ratings()
                 (season, week, team, off_epa_rating, def_epa_rating)
    """
    out = games.copy()

    home_r = epa_ratings.rename(
        columns={"off_epa_rating": "home_off_epa", "def_epa_rating": "home_def_epa", "team": "home_team"}
    )[["season", "week", "home_team", "home_off_epa", "home_def_epa"]]
    away_r = epa_ratings.rename(
        columns={"off_epa_rating": "away_off_epa", "def_epa_rating": "away_def_epa", "team": "away_team"}
    )[["season", "week", "away_team", "away_off_epa", "away_def_epa"]]

    out = out.merge(home_r, on=["season", "week", "home_team"], how="left")
    out = out.merge(away_r, on=["season", "week", "away_team"], how="left")

    # net edge: home offense vs away defense, minus away offense vs home defense
    home_net = out["home_off_epa"] - out["away_def_epa"]
    away_net = out["away_off_epa"] - out["home_def_epa"]
    out["epa_spread"] = -(home_net - away_net) * PLAYS_PER_GAME * EPA_TO_POINTS / 2.0
    # divided by 2 because both teams' plays count toward the same game; keeps scale sane
    # -> THIS CONSTANT IS A GUESS. Backtest MAE/RMSE and adjust PLAYS_PER_GAME/EPA_TO_POINTS accordingly.
    return out


def build_dvoa_spread(games_with_dvoa: pd.DataFrame) -> pd.DataFrame:
    """games_with_dvoa: output of dvoa_loader.attach_dvoa_to_games()."""
    out = games_with_dvoa.copy()
    dvoa_diff = out["home_total_dvoa_lagged"] - out["away_total_dvoa_lagged"]
    out["dvoa_spread"] = -dvoa_diff * DVOA_TO_POINTS
    return out


def combine_spreads(
    games: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    games must already have elo_spread, epa_spread, dvoa_spread columns (some may
    be null for rows where a source doesn't have coverage, e.g. DVOA not yet loaded).

    weights: e.g. {"elo_spread": 0.3, "epa_spread": 0.4, "dvoa_spread": 0.3}
             Defaults to equal-weighting only the non-null sources per row.
    Also applies home field advantage ONCE at the end (don't double-count it if
    any individual source already bakes in home field).
    """
    out = games.copy()
    source_cols = [c for c in ["elo_spread", "epa_spread", "dvoa_spread"] if c in out.columns]

    if weights is None:
        # equal weight across whichever sources are non-null for that row
        out["combined_spread"] = out[source_cols].mean(axis=1, skipna=True)
    else:
        w = pd.Series({c: weights.get(c, 0.0) for c in source_cols})
        weighted = out[source_cols].mul(w, axis=1)
        row_weight_sum = out[source_cols].notna().mul(w, axis=1).sum(axis=1)
        out["combined_spread"] = weighted.sum(axis=1, skipna=True) / row_weight_sum

    out["combined_spread"] = out["combined_spread"] - HOME_FIELD_PTS
    # subtracting because more-negative = more home-favored in our convention
    return out


if __name__ == "__main__":
    print(
        "This module expects pre-built elo_df, epa_ratings, and games_with_dvoa tables. "
        "See docs/CHANGELOG.md v1/v2 sections for the intended build order, and use "
        "src/backtest/harness.py to score elo_spread, epa_spread, dvoa_spread, and "
        "combined_spread separately against the closing line."
    )
