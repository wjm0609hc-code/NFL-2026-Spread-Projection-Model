"""
Tests the rest-differential adjustment layered on top of the EPA + turnover
luck core (the established main drivers), with train/holdout split. Also
runs a diagnostic (not a full adjustment layer) on wind speed, since wind's
well-established effect is on the TOTAL (both teams score less), not
necessarily the spread -- this checks honestly whether that holds here
before deciding whether a wind-based spread adjustment is worth building.

Usage:
    python -m src.backtest.rest_weather_backtest
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.harness import score_model
from src.backtest.run_full_backtest import fetch_pbp, fetch_schedules
from src.features.epa_ratings import build_rolling_ratings, compute_play_level_epa
from src.features.rest_weather import build_rest_adjustment
from src.features.turnover_luck import build_turnover_adjustment
from src.models.spread_projection import build_epa_spread

SEASONS = [2021, 2022, 2023, 2024, 2025]
TRAIN_SEASONS = [2021, 2022, 2023]
TEST_SEASONS = [2024, 2025]


def build_scored_dataset() -> pd.DataFrame:
    print(f"Fetching play-by-play and schedules for {SEASONS}...")
    pbp = fetch_pbp(SEASONS)
    schedules = fetch_schedules(SEASONS)

    print("Building EPA core spread...")
    team_week_epa = compute_play_level_epa(pbp)
    epa_ratings = build_rolling_ratings(team_week_epa)
    epa_spread_df = build_epa_spread(schedules[["season", "week", "home_team", "away_team"]], epa_ratings)

    print("Building turnover luck adjustment...")
    to_adj = build_turnover_adjustment(pbp)

    print("Building rest adjustment...")
    rest_adj = build_rest_adjustment(schedules)

    print("Merging...")
    scored = schedules.merge(
        epa_spread_df[["season", "week", "home_team", "away_team", "epa_spread"]],
        on=["season", "week", "home_team", "away_team"], how="left",
    )
    home_to = to_adj.rename(columns={"team": "home_team", "turnover_adjustment_pts": "home_to_adj"})[["season", "week", "home_team", "home_to_adj"]]
    away_to = to_adj.rename(columns={"team": "away_team", "turnover_adjustment_pts": "away_to_adj"})[["season", "week", "away_team", "away_to_adj"]]
    scored = scored.merge(home_to, on=["season", "week", "home_team"], how="left")
    scored = scored.merge(away_to, on=["season", "week", "away_team"], how="left")
    scored["home_to_adj"] = scored["home_to_adj"].fillna(0.0)
    scored["away_to_adj"] = scored["away_to_adj"].fillna(0.0)

    scored = scored.merge(rest_adj[["season", "week", "home_team", "away_team", "rest_adjustment_pts"]], on=["season", "week", "home_team", "away_team"], how="left")

    # core: EPA + turnover luck (established main drivers)
    scored["core_spread"] = scored["epa_spread"] - scored["home_to_adj"] + scored["away_to_adj"]
    # core + rest
    scored["core_plus_rest_spread"] = scored["core_spread"] - scored["rest_adjustment_pts"]

    return scored


def main():
    scored = build_scored_dataset()

    for segment_name, seasons in [("ALL", SEASONS), ("TRAIN (2021-23)", TRAIN_SEASONS), ("HOLDOUT (2024-25)", TEST_SEASONS)]:
        seg = scored[scored["season"].isin(seasons)]
        print(f"\n{'=' * 70}")
        print(f"{segment_name} (n={len(seg)})")
        print(f"{'=' * 70}")
        for label, col in [("Core (EPA + turnover)", "core_spread"), ("Core + rest", "core_plus_rest_spread")]:
            m = score_model(seg, model_spread_col=col, closing_spread_col="spread_line_standard", actual_margin_col="actual_margin")
            print(f"  {label}: MAE={m['mae']}, RMSE={m['rmse']}, ATS={m['ats_win_rate']}, n={m['n_games']}")

    # subset: meaningful rest mismatches (short week, bye week, etc.)
    big_rest_diff = scored[scored["rest_adjustment_pts"].abs() >= 1.5]
    print(f"\n{'=' * 70}")
    print(f"MEANINGFUL REST MISMATCH GAMES ONLY (|adj| >= 1.5, n={len(big_rest_diff)})")
    print(f"{'=' * 70}")
    for label, col in [("Core (EPA + turnover)", "core_spread"), ("Core + rest", "core_plus_rest_spread")]:
        m = score_model(big_rest_diff, model_spread_col=col, closing_spread_col="spread_line_standard", actual_margin_col="actual_margin")
        print(f"  {label}: MAE={m['mae']}, RMSE={m['rmse']}, ATS={m['ats_win_rate']}, n={m['n_games']}")

    # --- Wind diagnostic (not a spread adjustment -- checking whether it's worth building one) ---
    print(f"\n{'=' * 70}")
    print("WIND DIAGNOSTIC -- checking if high wind correlates with core-spread error direction")
    print(f"{'=' * 70}")
    wind_games = scored.dropna(subset=["wind"]).copy()
    wind_games["error"] = wind_games["core_spread"] - wind_games["actual_margin"]
    high_wind = wind_games[wind_games["wind"] >= 15]
    low_wind = wind_games[wind_games["wind"] < 15]
    print(f"High wind (>=15mph, n={len(high_wind)}): mean error={high_wind['error'].mean():.3f}, mean |actual_margin|={high_wind['actual_margin'].abs().mean():.3f}")
    print(f"Low/no wind (<15mph, n={len(low_wind)}): mean error={low_wind['error'].mean():.3f}, mean |actual_margin|={low_wind['actual_margin'].abs().mean():.3f}")
    print("(No systematic spread bias expected here if wind mainly affects total scoring, not margin -- checking honestly.)")

    scored.to_parquet("data/processed/rest_weather_backtest_scored.parquet")
    print("\nSaved to data/processed/rest_weather_backtest_scored.parquet")


if __name__ == "__main__":
    main()
