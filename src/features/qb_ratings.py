"""
QB efficiency ratings and QB-change detection, built directly from play-by-play
dropback data (no paid injury feed required for backtesting -- we use who
ACTUALLY took the snaps as the historical record, which is the correct thing
to backtest against; live weekly use would instead need a depth-chart/injury
feed to know who's expected to start BEFORE the game, which is a separate,
later problem from proving the adjustment has value at all).

Core idea:
  1. Build a per-passer trailing EPA/dropback rating (walk-forward safe --
     only prior games count).
  2. For each team-week, identify the team's "established" QB -- the passer
     with the most dropbacks over the trailing window -- separately from
     whoever actually played that specific week.
  3. Flag "QB change" games: where the actual starter that week differs from
     the team's established QB.
  4. Compute an adjustment: the gap between the actual starter's trailing
     rating and the established starter's trailing rating, capped to avoid
     wild swings from tiny samples (a backup with 10 career dropbacks
     shouldn't produce a 15-point swing).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SHRINKAGE_K = 150                # shrinkage strength in dropbacks -- roughly 4-5 starts' worth.
                                  # A passer's rating is a weighted blend of his own observed trailing
                                  # rating and REPLACEMENT_LEVEL_EPA, weighted by sample/(sample+K).
                                  # This replaces a hard "trust/don't-trust" cutoff with a continuous
                                  # one: a 67-dropback sample (barely past an old 50-dropback cutoff)
                                  # should NOT be trusted at full strength, and shrinkage handles that
                                  # automatically rather than needing a second manual threshold.
MAX_ADJUSTMENT_PTS = 7.0          # cap, matches the config.yaml qb_adjustment.max_shift_pts prior
ROLLING_GAMES = 16                # trailing window for passer rating (roughly a season)
REPLACEMENT_LEVEL_EPA = -0.10     # thin-sample passers shrink toward BELOW-average, not average --
                                   # a QB without an established track record is disproportionately likely
                                   # to be a backup for a reason. Starting prior, not a precisely derived
                                   # constant -- worth refining against real backup performance data.


def build_passer_game_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, team, passer): dropbacks and EPA/dropback that game."""
    df = pbp[(pbp["qb_dropback"] == 1) & pbp["passer_player_id"].notna() & pbp["epa"].notna()].copy()
    grouped = (
        df.groupby(["season", "week", "posteam", "passer_player_id", "passer_player_name"])
        .agg(dropbacks=("epa", "size"), epa_sum=("epa", "sum"))
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    grouped["epa_per_dropback"] = grouped["epa_sum"] / grouped["dropbacks"]
    return grouped


def identify_starters(passer_game_stats: pd.DataFrame) -> pd.DataFrame:
    """For each (season, week, team), the passer with the most dropbacks = that game's starter."""
    idx = passer_game_stats.groupby(["season", "week", "team"])["dropbacks"].idxmax()
    starters = passer_game_stats.loc[idx].reset_index(drop=True)
    return starters.rename(columns={"passer_player_id": "starter_id", "passer_player_name": "starter_name"})


def build_walk_forward_passer_ratings(passer_game_stats: pd.DataFrame, rolling_games: int = ROLLING_GAMES) -> pd.DataFrame:
    """
    Trailing EPA/dropback rating per passer, using only STRICTLY PRIOR games
    (across season boundaries -- a QB's track record doesn't reset each year).
    Weighted by dropback volume within the window, not a simple mean of
    per-game averages (avoids letting a 3-dropback relief appearance count
    the same as a 35-dropback start).

    IMPORTANT: this produces a row only for weeks where the passer actually
    had a dropback that week. A passer's rating must still be look-up-able
    for weeks he DIDN'T play (e.g. injured, benched) -- callers needing "this
    passer's most recent known rating as of week W" should forward-fill this
    output across a full (season, week) grid; see fill_ratings_forward().
    """
    df = passer_game_stats.sort_values(["passer_player_id", "season", "week"]).copy()
    out_rows = []
    for passer_id, grp in df.groupby("passer_player_id"):
        grp = grp.reset_index(drop=True)
        for i in range(len(grp)):
            history = grp.iloc[:i].tail(rolling_games)
            season, week = grp.loc[i, "season"], grp.loc[i, "week"]
            total_dropbacks = history["dropbacks"].sum()
            if total_dropbacks == 0:
                rating = REPLACEMENT_LEVEL_EPA  # no history at all -- full weight on the prior
            else:
                observed_rating = history["epa_sum"].sum() / total_dropbacks
                shrink_weight = total_dropbacks / (total_dropbacks + SHRINKAGE_K)
                rating = shrink_weight * observed_rating + (1 - shrink_weight) * REPLACEMENT_LEVEL_EPA
            sample = total_dropbacks
            out_rows.append(
                {
                    "season": season,
                    "week": week,
                    "passer_id": passer_id,
                    "passer_name": grp.loc[i, "passer_player_name"],
                    "trailing_epa_per_dropback": rating,
                    "trailing_dropback_sample": sample,
                }
            )
    return pd.DataFrame(out_rows)


def fill_ratings_forward(passer_ratings: pd.DataFrame, all_weeks: pd.DataFrame) -> pd.DataFrame:
    """
    Expands passer_ratings onto every (season, week) in all_weeks (typically
    every week any game was played league-wide) so a lookup for a passer who
    didn't play a given week (injured, benched, on another team's bench)
    still returns his most recently known trailing rating, forward-filled --
    NOT a silent fallback to replacement level, which would incorrectly
    treat "didn't play this week" the same as "has no track record at all."

    all_weeks: dataframe with columns season, week (one row per week that
    exists in the schedule, deduplicated).
    """
    passers = passer_ratings["passer_id"].unique()
    grid = all_weeks.assign(key=1).merge(pd.DataFrame({"passer_id": passers, "key": 1}), on="key").drop(columns="key")
    merged = grid.merge(passer_ratings, on=["season", "week", "passer_id"], how="left")
    merged = merged.sort_values(["passer_id", "season", "week"])
    merged["passer_name"] = merged.groupby("passer_id")["passer_name"].ffill()
    merged["trailing_epa_per_dropback"] = merged.groupby("passer_id")["trailing_epa_per_dropback"].ffill()
    merged["trailing_dropback_sample"] = merged.groupby("passer_id")["trailing_dropback_sample"].ffill()
    return merged


def build_established_starter(starters: pd.DataFrame, rolling_games: int = ROLLING_GAMES) -> pd.DataFrame:
    """
    For each (season, week, team), who is the team's ESTABLISHED starter --
    the passer with the most cumulative dropbacks over the trailing window,
    evaluated using only games BEFORE this week (walk-forward safe). This is
    what "this team's normal QB" means, independent of who actually played
    this specific week.
    """
    df = starters.sort_values(["team", "season", "week"]).copy()
    out_rows = []
    for team, grp in df.groupby("team"):
        grp = grp.reset_index(drop=True)
        for i in range(len(grp)):
            history = grp.iloc[:i].tail(rolling_games)
            season, week = grp.loc[i, "season"], grp.loc[i, "week"]
            if len(history) == 0:
                established_id, established_name = None, None
            else:
                # whoever has the most total dropbacks as starter in the trailing window
                totals = history.groupby(["starter_id", "starter_name"])["dropbacks"].sum()
                top = totals.idxmax()
                established_id, established_name = top
            out_rows.append(
                {
                    "season": season,
                    "week": week,
                    "team": team,
                    "established_qb_id": established_id,
                    "established_qb_name": established_name,
                }
            )
    return pd.DataFrame(out_rows)


def build_qb_adjustment(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Full pipeline: returns one row per (season, week, team) with:
      - actual_starter_id / name (who actually played)
      - established_qb_id / name (team's normal starter, walk-forward safe)
      - qb_change (bool)
      - qb_adjustment_pts (capped point adjustment to apply to that team's
        offensive projection: negative = weaker QB playing than established,
        positive = stronger QB playing than established)
    """
    passer_stats = build_passer_game_stats(pbp)
    starters = identify_starters(passer_stats)
    passer_ratings = build_walk_forward_passer_ratings(passer_stats)
    established = build_established_starter(starters)

    all_weeks = starters[["season", "week"]].drop_duplicates()
    passer_ratings_filled = fill_ratings_forward(passer_ratings, all_weeks)

    out = starters.merge(established, on=["season", "week", "team"], how="left")
    out = out.merge(
        passer_ratings_filled.rename(columns={"passer_id": "starter_id", "trailing_epa_per_dropback": "actual_starter_rating", "trailing_dropback_sample": "actual_starter_sample"})[
            ["season", "week", "starter_id", "actual_starter_rating", "actual_starter_sample"]
        ],
        on=["season", "week", "starter_id"], how="left",
    )
    out = out.merge(
        passer_ratings_filled.rename(columns={"passer_id": "established_qb_id", "trailing_epa_per_dropback": "established_qb_rating", "trailing_dropback_sample": "established_qb_sample"})[
            ["season", "week", "established_qb_id", "established_qb_rating", "established_qb_sample"]
        ],
        on=["season", "week", "established_qb_id"], how="left",
    )

    out["qb_change"] = (out["starter_id"] != out["established_qb_id"]) & out["established_qb_id"].notna()

    # ratings are already shrinkage-adjusted in build_walk_forward_passer_ratings;
    # any remaining nulls here mean a passer genuinely has zero prior history at all
    out["actual_starter_rating"] = out["actual_starter_rating"].fillna(REPLACEMENT_LEVEL_EPA)
    out["established_qb_rating"] = out["established_qb_rating"].fillna(REPLACEMENT_LEVEL_EPA)

    raw_gap = (out["actual_starter_rating"] - out["established_qb_rating"]).fillna(0.0)
    # EPA/dropback gap -> points: scale by a typical team's ~35 dropbacks/game, then cap
    out["qb_adjustment_pts"] = np.clip(raw_gap * 35, -MAX_ADJUSTMENT_PTS, MAX_ADJUSTMENT_PTS)
    out.loc[~out["qb_change"], "qb_adjustment_pts"] = 0.0  # no adjustment when the normal starter played

    return out[["season", "week", "team", "starter_name", "established_qb_name", "qb_change", "qb_adjustment_pts"]]


if __name__ == "__main__":
    print("Import build_qb_adjustment(pbp) -- see src/backtest/qb_adjustment_backtest.py for usage.")
