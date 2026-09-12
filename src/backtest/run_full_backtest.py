"""
Runs the full Elo / EPA / DVOA standalone backtest using nflverse data fetched
directly from GitHub release URLs (bypassing the nfl_data_py package, which
pins pandas<2.0 and fails to build on newer Python/pandas environments).

This produces identical data to what nfl_data_py would give you — same
source, same files — just fetched with plain pandas.read_parquet() instead
of going through the package. If nfl_data_py installs cleanly in your local
environment, you can use src/ingestion/nflverse_loader.py instead; this
script is the fallback that's known to work.

Usage:
    python -m src.backtest.run_full_backtest
"""
from __future__ import annotations

import pandas as pd

from src.backtest.harness import score_model
from src.features.dvoa_loader import attach_dvoa_to_games, build_walk_forward_dvoa, load_dvoa
from src.features.epa_ratings import build_rolling_ratings, compute_play_level_epa
from src.models.elo_baseline import run_elo_backtest
from src.models.spread_projection import build_dvoa_spread, build_elo_spread, build_epa_spread

SEASONS = [2021, 2022, 2023, 2024, 2025]
PBP_URL_TMPL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
SCHEDULES_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.parquet"


def fetch_pbp(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        df = pd.read_parquet(PBP_URL_TMPL.format(season=s))
        frames.append(df)
    pbp = pd.concat(frames, ignore_index=True)
    return pbp[pbp["season_type"] == "REG"].copy()


def fetch_schedules(seasons: list[int]) -> pd.DataFrame:
    sched = pd.read_parquet(SCHEDULES_URL)
    sched = sched[(sched["season"].isin(seasons)) & (sched["game_type"] == "REG")].copy()
    sched["actual_margin"] = sched["home_score"] - sched["away_score"]
    # nflverse spread_line convention: POSITIVE = home favored.
    # Our internal convention (harness.py, spread_projection.py): NEGATIVE = home favored.
    # Flip sign here, once, at the source, so every downstream comparison is apples-to-apples.
    sched["spread_line_standard"] = -sched["spread_line"]
    return sched


def main():
    print(f"Fetching play-by-play for {SEASONS} directly from nflverse-data releases...")
    pbp = fetch_pbp(SEASONS)
    print(f"  {len(pbp):,} regular-season plays loaded.")

    print("Fetching schedules (includes closing spread_line)...")
    schedules = fetch_schedules(SEASONS)
    print(f"  {len(schedules):,} regular-season games loaded.")

    print("\nBuilding Elo ratings (walk-forward, game-by-game)...")
    elo_df = run_elo_backtest(schedules)
    elo_spread_df = build_elo_spread(elo_df)

    print("Building EPA rolling ratings (8-game trailing window, walk-forward safe)...")
    team_week_epa = compute_play_level_epa(pbp)
    epa_ratings = build_rolling_ratings(team_week_epa)
    epa_spread_df = build_epa_spread(schedules[["season", "week", "home_team", "away_team"]], epa_ratings)

    print("Loading sourced DVOA data (lagged 1 week to prevent leakage)...")
    dvoa_raw = load_dvoa(SEASONS)
    dvoa_wf = build_walk_forward_dvoa(dvoa_raw, lag_weeks=1)
    games_with_dvoa = attach_dvoa_to_games(schedules[["season", "week", "home_team", "away_team"]], dvoa_wf)
    dvoa_spread_df = build_dvoa_spread(games_with_dvoa)

    print("\nMerging all sources onto the schedule for scoring...")
    scored = schedules.merge(
        elo_spread_df, on=["season", "week", "home_team", "away_team"], how="left"
    ).merge(
        epa_spread_df[["season", "week", "home_team", "away_team", "epa_spread"]],
        on=["season", "week", "home_team", "away_team"], how="left",
    ).merge(
        dvoa_spread_df[["season", "week", "home_team", "away_team", "dvoa_spread"]],
        on=["season", "week", "home_team", "away_team"], how="left",
    )

    print("\n" + "=" * 70)
    print("BACKTEST RESULTS: standalone performance vs. closing line")
    print("2021-2025 regular season, n =", len(scored), "games")
    print("=" * 70)

    results_table = []
    for label, col in [("Elo", "elo_spread"), ("EPA", "epa_spread"), ("DVOA", "dvoa_spread")]:
        metrics = score_model(
            scored, model_spread_col=col, closing_spread_col="spread_line_standard", actual_margin_col="actual_margin"
        )
        metrics["model"] = label
        results_table.append(metrics)
        print(f"\n{label}:")
        for k, v in metrics.items():
            if k != "model":
                print(f"  {k}: {v}")

    results_df = pd.DataFrame(results_table).set_index("model")
    results_df.to_csv("data/processed/backtest_summary.csv")
    scored.to_parquet("data/processed/full_backtest_scored_games.parquet")

    print("\n" + "=" * 70)
    print("Summary table (also saved to data/processed/backtest_summary.csv):")
    print("=" * 70)
    print(results_df.to_string())

    print("\nFull scored game-level dataset saved to data/processed/full_backtest_scored_games.parquet")


if __name__ == "__main__":
    main()
