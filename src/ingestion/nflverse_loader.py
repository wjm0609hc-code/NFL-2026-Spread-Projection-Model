"""
Loads play-by-play and schedule data from nflverse via the nfl_data_py package.
This is the backtest backbone: free, well-validated, goes back to 1999,
and is the standard source for EPA-based analysis in the public sabermetrics community.

Usage:
    from src.ingestion.nflverse_loader import load_pbp, load_schedules
    pbp = load_pbp([2021, 2022, 2023, 2024, 2025])
"""
from __future__ import annotations

from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

RAW_DIR = Path("data/raw/nflverse")


def load_pbp(seasons: list[int], force_refresh: bool = False) -> pd.DataFrame:
    """Load play-by-play for given seasons, caching to parquet per season."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for season in seasons:
        cache_path = RAW_DIR / f"pbp_{season}.parquet"
        if cache_path.exists() and not force_refresh:
            frames.append(pd.read_parquet(cache_path))
        else:
            df = nfl.import_pbp_data([season])
            df.to_parquet(cache_path)
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_schedules(seasons: list[int], force_refresh: bool = False) -> pd.DataFrame:
    """Load game schedules/results (includes closing spread, total, actual score)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_DIR / f"schedules_{min(seasons)}_{max(seasons)}.parquet"
    if cache_path.exists() and not force_refresh:
        return pd.read_parquet(cache_path)
    df = nfl.import_schedules(seasons)
    df.to_parquet(cache_path)
    return df


def load_rosters(seasons: list[int], force_refresh: bool = False) -> pd.DataFrame:
    """Weekly rosters — useful for identifying starting QB by week."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_DIR / f"rosters_{min(seasons)}_{max(seasons)}.parquet"
    if cache_path.exists() and not force_refresh:
        return pd.read_parquet(cache_path)
    df = nfl.import_weekly_rosters(seasons)
    df.to_parquet(cache_path)
    return df


if __name__ == "__main__":
    seasons = [2021, 2022, 2023, 2024, 2025]
    pbp = load_pbp(seasons)
    sched = load_schedules(seasons)
    print(f"Loaded {len(pbp):,} plays and {len(sched):,} scheduled games across {seasons}")
