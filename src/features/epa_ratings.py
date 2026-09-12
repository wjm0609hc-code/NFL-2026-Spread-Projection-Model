"""
Builds team-week power ratings from play-by-play EPA data.

CRITICAL DESIGN RULE: every rating for team T going into week W of season S
is computed using ONLY plays from before that game's kickoff. This file
produces a long-format table: (season, week, team, off_epa_play, def_epa_play,
success_rate, ...) that downstream model code can safely join into game rows
without leaking future information.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _clean_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    """Filter to meaningful plays (exclude no-plays, kneels, spikes, garbage time optional)."""
    df = pbp.copy()
    df = df[df["play_type"].isin(["pass", "run"])]
    df = df[df["epa"].notna()]
    # Optional garbage-time filter: exclude plays where win prob is extreme and late
    # Keeping this OFF by default — garbage time still contains signal, and filtering
    # it changes sample size unevenly across teams. Toggle on for a v2 experiment.
    return df


def compute_play_level_epa(pbp: pd.DataFrame) -> pd.DataFrame:
    """Reduce play-by-play to one row per (season, week, posteam) offense and
    (season, week, defteam) defense, with EPA/play and success rate."""
    df = _clean_pbp(pbp)
    df["success"] = (df["epa"] > 0).astype(int)

    off = (
        df.groupby(["season", "week", "posteam"])
        .agg(off_epa_play=("epa", "mean"), off_success_rate=("success", "mean"), off_plays=("epa", "size"))
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    defn = (
        df.groupby(["season", "week", "defteam"])
        .agg(def_epa_play=("epa", "mean"), def_success_rate=("success", "mean"), def_plays=("epa", "size"))
        .reset_index()
        .rename(columns={"defteam": "team"})
    )
    return off.merge(defn, on=["season", "week", "team"], how="outer")


def build_rolling_ratings(
    team_week_epa: pd.DataFrame,
    rolling_window: int = 8,
    min_games: int = 4,
) -> pd.DataFrame:
    """
    For each team, compute a rolling (trailing, excludes current week) EPA rating
    using the last `rolling_window` games, falling back to season-to-date if fewer
    games are available. This is what makes the feature safe for walk-forward use:
    the rating attached to week W was computed entirely from weeks < W.

    Returns columns: season, week, team, off_epa_rating, def_epa_rating, games_sample
    """
    df = team_week_epa.sort_values(["team", "season", "week"]).copy()

    out_rows = []
    for team, grp in df.groupby("team"):
        grp = grp.reset_index(drop=True)
        for i in range(len(grp)):
            history = grp.iloc[:i]  # strictly prior games — no leakage
            season, week = grp.loc[i, "season"], grp.loc[i, "week"]

            if len(history) == 0:
                # no prior data at all — league-average prior, flagged low-confidence
                out_rows.append(
                    {
                        "season": season,
                        "week": week,
                        "team": team,
                        "off_epa_rating": 0.0,
                        "def_epa_rating": 0.0,
                        "games_sample": 0,
                    }
                )
                continue

            trailing = history.tail(rolling_window)
            out_rows.append(
                {
                    "season": season,
                    "week": week,
                    "team": team,
                    "off_epa_rating": trailing["off_epa_play"].mean(),
                    "def_epa_rating": trailing["def_epa_play"].mean(),
                    "games_sample": len(trailing),
                }
            )

    ratings = pd.DataFrame(out_rows)
    ratings["low_confidence"] = ratings["games_sample"] < min_games
    return ratings


if __name__ == "__main__":
    from src.ingestion.nflverse_loader import load_pbp

    pbp = load_pbp([2021, 2022, 2023, 2024, 2025])
    team_week = compute_play_level_epa(pbp)
    ratings = build_rolling_ratings(team_week)
    ratings.to_parquet("data/processed/epa_ratings.parquet")
    print(ratings.tail(10))
