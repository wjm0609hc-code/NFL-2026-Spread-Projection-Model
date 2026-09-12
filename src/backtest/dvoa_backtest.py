"""
Standalone backtest: how predictive is DVOA on its own?

Answers three questions:
  1. How well does DVOA-implied spread track actual game margins (MAE/RMSE)?
  2. Does DVOA beat the closing line (ATS win rate, CLV) — i.e. is there
     anything in DVOA the market hasn't already priced in?
  3. How does DVOA's standalone performance compare to Elo and EPA standalone?

Run this AFTER you've:
  - populated data/raw/dvoa/dvoa_{season}.csv for each season in scope
  - built elo_df via src/models/elo_baseline.py
  - built epa_ratings via src/features/epa_ratings.py

Usage:
    python -m src.backtest.dvoa_backtest
"""
from __future__ import annotations

import pandas as pd

from src.backtest.harness import score_model, segment_backtest
from src.features.dvoa_loader import attach_dvoa_to_games, build_walk_forward_dvoa, load_dvoa
from src.features.epa_ratings import build_rolling_ratings, compute_play_level_epa
from src.ingestion.nflverse_loader import load_pbp, load_schedules
from src.models.elo_baseline import run_elo_backtest
from src.models.spread_projection import build_dvoa_spread, build_elo_spread, build_epa_spread

SEASONS = [2021, 2022, 2023, 2024, 2025]


def build_actual_margin(schedules: pd.DataFrame) -> pd.DataFrame:
    df = schedules.copy()
    df["actual_margin"] = df["home_score"] - df["away_score"]
    return df


def main():
    print(f"Loading schedules and play-by-play for {SEASONS} ...")
    schedules = load_schedules(SEASONS)
    schedules = build_actual_margin(schedules)
    pbp = load_pbp(SEASONS)

    # --- Elo standalone ---
    print("Running Elo baseline ...")
    elo_df = run_elo_backtest(schedules)
    elo_spread_df = build_elo_spread(elo_df)

    # --- EPA standalone ---
    print("Building EPA rolling ratings ...")
    team_week_epa = compute_play_level_epa(pbp)
    epa_ratings = build_rolling_ratings(team_week_epa)
    epa_spread_df = build_epa_spread(
        schedules[["season", "week", "home_team", "away_team"]], epa_ratings
    )

    # --- DVOA standalone ---
    print("Loading manually-sourced DVOA ...")
    try:
        dvoa_raw = load_dvoa(SEASONS)
    except FileNotFoundError as e:
        print(f"\n[STOPPED] {e}\n")
        print("Add your sourced DVOA CSVs, then re-run this script.")
        return

    dvoa_wf = build_walk_forward_dvoa(dvoa_raw, lag_weeks=1)
    games_with_dvoa = attach_dvoa_to_games(
        schedules[["season", "week", "home_team", "away_team"]], dvoa_wf
    )
    dvoa_spread_df = build_dvoa_spread(games_with_dvoa)

    # --- Merge everything onto the schedule for scoring ---
    scored = schedules.merge(
        elo_spread_df, on=["season", "week", "home_team", "away_team"], how="left"
    ).merge(
        epa_spread_df[["season", "week", "home_team", "away_team", "epa_spread"]],
        on=["season", "week", "home_team", "away_team"],
        how="left",
    ).merge(
        dvoa_spread_df[["season", "week", "home_team", "away_team", "dvoa_spread"]],
        on=["season", "week", "home_team", "away_team"],
        how="left",
    )

    # nflverse schedules include closing line as 'spread_line' (home perspective, negative = home favored)
    if "spread_line" not in scored.columns:
        raise KeyError(
            "Expected 'spread_line' column from nflverse schedules for closing-line comparison. "
            "Check nfl_data_py schedule columns — this may need remapping if the source changed."
        )

    print("\n=== STANDALONE BACKTEST RESULTS (vs. closing line) ===\n")
    for label, col in [("Elo", "elo_spread"), ("EPA", "epa_spread"), ("DVOA", "dvoa_spread")]:
        metrics = score_model(scored, model_spread_col=col)
        print(f"{label}: {metrics}")

    print("\n=== DVOA SEGMENT BREAKDOWN ===\n")
    if "div_game" in scored.columns:
        seg = segment_backtest(scored, model_spread_col="dvoa_spread", segment_col="div_game")
        print(seg)
    else:
        print("No 'div_game' column found in schedule — skipping segment breakdown (optional).")

    out_path = "data/processed/dvoa_backtest_results.parquet"
    scored.to_parquet(out_path)
    print(f"\nFull scored dataset written to {out_path}")


if __name__ == "__main__":
    main()
