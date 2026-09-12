"""
Tests whether the QB adjustment layer improves on EPA-alone -- overall, and
critically, on the SUBSET of games where a QB change actually occurred (that's
where this layer should matter; diluting into the full sample where ~70% of
games have no QB change at all will mute any real effect).

Same train/holdout discipline as weight_sweep.py: nothing here is "tuned" on
the holdout, since qb_ratings.py's constants (shrinkage K, replacement level,
adjustment cap) were set from first principles / sanity checks, not fit to
backtest performance -- but we still report both splits separately to see if
the effect (if any) is consistent across them, which is the honest way to
gut-check a layer built from domain reasoning rather than optimization.

Usage:
    python -m src.backtest.qb_adjustment_backtest
"""
from __future__ import annotations

import pandas as pd

from src.backtest.harness import score_model
from src.backtest.run_full_backtest import fetch_pbp, fetch_schedules
from src.features.epa_ratings import build_rolling_ratings, compute_play_level_epa
from src.features.qb_ratings import build_qb_adjustment
from src.models.spread_projection import build_epa_spread

SEASONS = [2021, 2022, 2023, 2024, 2025]
TRAIN_SEASONS = [2021, 2022, 2023]
TEST_SEASONS = [2024, 2025]


def build_scored_dataset() -> pd.DataFrame:
    print(f"Fetching play-by-play and schedules for {SEASONS}...")
    pbp = fetch_pbp(SEASONS)
    schedules = fetch_schedules(SEASONS)

    print("Building EPA ratings and standalone spread...")
    team_week_epa = compute_play_level_epa(pbp)
    epa_ratings = build_rolling_ratings(team_week_epa)
    epa_spread_df = build_epa_spread(schedules[["season", "week", "home_team", "away_team"]], epa_ratings)

    print("Building QB adjustment layer...")
    qb_adj = build_qb_adjustment(pbp)

    print("Merging...")
    scored = schedules.merge(
        epa_spread_df[["season", "week", "home_team", "away_team", "epa_spread"]],
        on=["season", "week", "home_team", "away_team"], how="left",
    )
    home_adj = qb_adj.rename(columns={"team": "home_team", "qb_adjustment_pts": "home_qb_adj", "qb_change": "home_qb_change"})[
        ["season", "week", "home_team", "home_qb_adj", "home_qb_change"]
    ]
    away_adj = qb_adj.rename(columns={"team": "away_team", "qb_adjustment_pts": "away_qb_adj", "qb_change": "away_qb_change"})[
        ["season", "week", "away_team", "away_qb_adj", "away_qb_change"]
    ]
    scored = scored.merge(home_adj, on=["season", "week", "home_team"], how="left")
    scored = scored.merge(away_adj, on=["season", "week", "away_team"], how="left")
    scored["home_qb_adj"] = scored["home_qb_adj"].fillna(0.0)
    scored["away_qb_adj"] = scored["away_qb_adj"].fillna(0.0)
    scored["home_qb_change"] = scored["home_qb_change"].fillna(False)
    scored["away_qb_change"] = scored["away_qb_change"].fillna(False)
    scored["any_qb_change"] = scored["home_qb_change"] | scored["away_qb_change"]

    # positive home_qb_adj = home QB better than usual -> spread should move MORE negative (home more favored)
    # positive away_qb_adj = away QB better than usual -> spread should move LESS negative / more positive
    scored["epa_spread_qb_adjusted"] = scored["epa_spread"] - scored["home_qb_adj"] + scored["away_qb_adj"]

    return scored


def main():
    scored = build_scored_dataset()

    print(f"\nGames with any QB change: {scored['any_qb_change'].sum()} of {len(scored)}")

    for segment_name, seasons in [("ALL", SEASONS), ("TRAIN (2021-23)", TRAIN_SEASONS), ("HOLDOUT (2024-25)", TEST_SEASONS)]:
        seg = scored[scored["season"].isin(seasons)]
        print(f"\n{'=' * 70}")
        print(f"{segment_name} -- overall (n={len(seg)})")
        print(f"{'=' * 70}")
        for label, col in [("EPA baseline", "epa_spread"), ("EPA + QB adjustment", "epa_spread_qb_adjusted")]:
            m = score_model(seg, model_spread_col=col, closing_spread_col="spread_line_standard", actual_margin_col="actual_margin")
            print(f"  {label}: MAE={m['mae']}, RMSE={m['rmse']}, ATS={m['ats_win_rate']}, n={m['n_games']}")

        qb_change_seg = seg[seg["any_qb_change"]]
        print(f"\n{segment_name} -- QB-CHANGE GAMES ONLY (n={len(qb_change_seg)}) -- where this layer should matter most")
        for label, col in [("EPA baseline", "epa_spread"), ("EPA + QB adjustment", "epa_spread_qb_adjusted")]:
            m = score_model(qb_change_seg, model_spread_col=col, closing_spread_col="spread_line_standard", actual_margin_col="actual_margin")
            print(f"  {label}: MAE={m['mae']}, RMSE={m['rmse']}, ATS={m['ats_win_rate']}, n={m['n_games']}")

    scored.to_parquet("data/processed/qb_adjustment_backtest_scored.parquet")
    print("\nFull scored dataset saved to data/processed/qb_adjustment_backtest_scored.parquet")


if __name__ == "__main__":
    main()
