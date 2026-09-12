"""
Tests whether the turnover-luck adjustment improves on EPA-alone -- overall,
and on the subset of games where at least one team carries a meaningful
turnover-luck signal (not near-zero), which is where this layer should
matter most.

Usage:
    python -m src.backtest.turnover_adjustment_backtest
"""
from __future__ import annotations

import pandas as pd

from src.backtest.harness import score_model
from src.backtest.run_full_backtest import fetch_pbp, fetch_schedules
from src.features.epa_ratings import build_rolling_ratings, compute_play_level_epa
from src.features.turnover_luck import build_turnover_adjustment
from src.models.spread_projection import build_epa_spread

SEASONS = [2021, 2022, 2023, 2024, 2025]
TRAIN_SEASONS = [2021, 2022, 2023]
TEST_SEASONS = [2024, 2025]
MEANINGFUL_LUCK_THRESHOLD = 2.0  # |adjustment| >= this to count as a "meaningful" signal game


def build_scored_dataset() -> pd.DataFrame:
    print(f"Fetching play-by-play and schedules for {SEASONS}...")
    pbp = fetch_pbp(SEASONS)
    schedules = fetch_schedules(SEASONS)

    print("Building EPA ratings and standalone spread...")
    team_week_epa = compute_play_level_epa(pbp)
    epa_ratings = build_rolling_ratings(team_week_epa)
    epa_spread_df = build_epa_spread(schedules[["season", "week", "home_team", "away_team"]], epa_ratings)

    print("Building turnover luck adjustment layer...")
    to_adj = build_turnover_adjustment(pbp)

    print("Merging...")
    scored = schedules.merge(
        epa_spread_df[["season", "week", "home_team", "away_team", "epa_spread"]],
        on=["season", "week", "home_team", "away_team"], how="left",
    )
    home_adj = to_adj.rename(columns={"team": "home_team", "turnover_adjustment_pts": "home_to_adj"})[["season", "week", "home_team", "home_to_adj"]]
    away_adj = to_adj.rename(columns={"team": "away_team", "turnover_adjustment_pts": "away_to_adj"})[["season", "week", "away_team", "away_to_adj"]]
    scored = scored.merge(home_adj, on=["season", "week", "home_team"], how="left")
    scored = scored.merge(away_adj, on=["season", "week", "away_team"], how="left")
    scored["home_to_adj"] = scored["home_to_adj"].fillna(0.0)
    scored["away_to_adj"] = scored["away_to_adj"].fillna(0.0)

    # positive home_to_adj = home team due for POSITIVE regression (was unlucky) -> spread more negative (more home-favored)
    scored["epa_spread_to_adjusted"] = scored["epa_spread"] - scored["home_to_adj"] + scored["away_to_adj"]
    scored["meaningful_luck_game"] = (scored["home_to_adj"].abs() >= MEANINGFUL_LUCK_THRESHOLD) | (scored["away_to_adj"].abs() >= MEANINGFUL_LUCK_THRESHOLD)

    return scored


def main():
    scored = build_scored_dataset()
    print(f"\nGames with meaningful turnover-luck signal (|adj| >= {MEANINGFUL_LUCK_THRESHOLD}): {scored['meaningful_luck_game'].sum()} of {len(scored)}")

    for segment_name, seasons in [("ALL", SEASONS), ("TRAIN (2021-23)", TRAIN_SEASONS), ("HOLDOUT (2024-25)", TEST_SEASONS)]:
        seg = scored[scored["season"].isin(seasons)]
        print(f"\n{'=' * 70}")
        print(f"{segment_name} -- overall (n={len(seg)})")
        print(f"{'=' * 70}")
        for label, col in [("EPA baseline", "epa_spread"), ("EPA + turnover luck adj", "epa_spread_to_adjusted")]:
            m = score_model(seg, model_spread_col=col, closing_spread_col="spread_line_standard", actual_margin_col="actual_margin")
            print(f"  {label}: MAE={m['mae']}, RMSE={m['rmse']}, ATS={m['ats_win_rate']}, n={m['n_games']}")

        meaningful_seg = seg[seg["meaningful_luck_game"]]
        print(f"\n{segment_name} -- MEANINGFUL LUCK GAMES ONLY (n={len(meaningful_seg)})")
        for label, col in [("EPA baseline", "epa_spread"), ("EPA + turnover luck adj", "epa_spread_to_adjusted")]:
            m = score_model(meaningful_seg, model_spread_col=col, closing_spread_col="spread_line_standard", actual_margin_col="actual_margin")
            print(f"  {label}: MAE={m['mae']}, RMSE={m['rmse']}, ATS={m['ats_win_rate']}, n={m['n_games']}")

    scored.to_parquet("data/processed/turnover_adjustment_backtest_scored.parquet")
    print("\nSaved to data/processed/turnover_adjustment_backtest_scored.parquet")


if __name__ == "__main__":
    main()
