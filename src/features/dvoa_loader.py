"""
Loader for manually-sourced DVOA data.

DVOA (Football Outsiders / FTN) is proprietary and not available via API,
so this module expects you to have already exported/scraped weekly team DVOA
tables into CSV files, one per season, at:

    data/raw/dvoa/dvoa_{season}.csv

Expected columns (rename your source columns to match before loading):
    season      int     e.g. 2024
    week        int     week number as of which this DVOA figure applies
                        (i.e. DVOA rating "entering week W" or "through week W-1" —
                        BE CONSISTENT, see note below on leakage)
    team        str     standard team abbreviation (matches nflverse convention, e.g. 'KC', 'SF')
    total_dvoa      float   overall DVOA (%)
    off_dvoa        float   offensive DVOA (%)
    def_dvoa        float   defensive DVOA (%)  -- NOTE: FO convention is often that
                            NEGATIVE def_dvoa is GOOD (fewer points allowed than expected).
                            Confirm your source's sign convention and normalize here.
    st_dvoa         float   special teams DVOA (%), optional

CRITICAL LEAKAGE WARNING:
FO's weekly DVOA snapshots are typically published as "DVOA through week W" —
meaning the week W figure already includes week W's game. If you feed that
directly into a projection for week W's game, you're leaking that game's own
result into its own prediction. The loader below shifts every team's DVOA
forward by one week (lag=1) by default so that the value used to predict
week W's game reflects only weeks < W. Set `lag_weeks=0` only if you've
already pre-shifted your source data and are certain of it.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DVOA_DIR = Path("data/raw/dvoa")

REQUIRED_COLS = {"season", "week", "team", "total_dvoa", "off_dvoa", "def_dvoa"}


def load_dvoa_season(season: int) -> pd.DataFrame:
    path = DVOA_DIR / f"dvoa_{season}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No DVOA file found at {path}. Export/scrape your source and save it there "
            f"with columns: {sorted(REQUIRED_COLS)}"
        )
    df = pd.read_csv(path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def load_dvoa(seasons: list[int]) -> pd.DataFrame:
    frames = [load_dvoa_season(s) for s in seasons]
    return pd.concat(frames, ignore_index=True)


def build_walk_forward_dvoa(
    dvoa_raw: pd.DataFrame,
    lag_weeks: int = 1,
) -> pd.DataFrame:
    """
    Shifts each team's DVOA figures forward by `lag_weeks` so that the value
    attached to week W was actually published as of week W - lag_weeks —
    i.e. safe to use as a pre-game feature for week W's matchup.

    Within a season only — DVOA does not carry across the offseason into week 1
    (week 1 will be null pre-lag; that's expected and should be backfilled with
    a preseason projection if your source has one, or left null and handled by
    the model as "low confidence" like the EPA rating cold-start).
    """
    df = dvoa_raw.sort_values(["team", "season", "week"]).copy()
    dvoa_cols = ["total_dvoa", "off_dvoa", "def_dvoa"]
    if "st_dvoa" in df.columns:
        dvoa_cols.append("st_dvoa")

    for col in dvoa_cols:
        df[f"{col}_lagged"] = df.groupby(["team", "season"])[col].shift(lag_weeks)

    keep_cols = ["season", "week", "team"] + [f"{c}_lagged" for c in dvoa_cols]
    return df[keep_cols]


def attach_dvoa_to_games(
    games: pd.DataFrame,
    dvoa_walk_forward: pd.DataFrame,
) -> pd.DataFrame:
    """
    Joins lagged DVOA onto a games table (one row per game, home_team/away_team,
    season, week) to produce home_total_dvoa_lagged, away_total_dvoa_lagged, etc.
    """
    out = games.copy()
    home = dvoa_walk_forward.rename(columns=lambda c: f"home_{c}" if c not in ("season", "week") else c)
    away = dvoa_walk_forward.rename(columns=lambda c: f"away_{c}" if c not in ("season", "week") else c)

    out = out.merge(home, left_on=["season", "week", "home_team"], right_on=["season", "week", "home_team"], how="left")
    out = out.merge(away, left_on=["season", "week", "away_team"], right_on=["season", "week", "away_team"], how="left")
    return out


if __name__ == "__main__":
    print(
        "Place manually-sourced CSVs at data/raw/dvoa/dvoa_{season}.csv, "
        "then run load_dvoa() -> build_walk_forward_dvoa() -> attach_dvoa_to_games()."
    )
