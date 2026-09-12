"""
Sweeps weight combinations for the Elo/EPA/DVOA blend to find whether any
weighting beats EPA standalone -- and critically, validates on a genuine
out-of-sample holdout rather than reporting in-sample "best" weights.

Why the train/test split matters: with ~1,359 games and ATS win rates all
sitting in a 48-52% band, the gap between weight combinations is close to
the noise floor. Grid-searching weights and reporting the best combo's
performance on the SAME data it was chosen from is a classic overfitting
trap -- you're partly just fitting noise. Splitting by season (train on
2021-2023, evaluate the winning weights on 2024-2025 untouched) is a much
more honest test of whether a weighting scheme actually generalizes.

Usage:
    python -m src.backtest.weight_sweep
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.blended_backtest import build_scored_dataset
from src.backtest.harness import score_model

TRAIN_SEASONS = [2021, 2022, 2023]
TEST_SEASONS = [2024, 2025]
HOME_FIELD_PTS = 1.8
WEIGHT_STEP = 0.1  # grid granularity


def weighted_blend(df: pd.DataFrame, w_elo: float, w_epa: float, w_dvoa: float) -> pd.Series:
    """
    Row-wise weighted average of elo_spread_no_hfa, epa_spread, dvoa_spread,
    renormalizing weights per row for whichever components are non-null
    (matters for Week 1 games where dvoa_spread is always null).
    """
    cols = {"elo_spread_no_hfa": w_elo, "epa_spread": w_epa, "dvoa_spread": w_dvoa}
    weighted_sum = pd.Series(0.0, index=df.index)
    weight_total = pd.Series(0.0, index=df.index)
    for col, w in cols.items():
        present = df[col].notna()
        weighted_sum = weighted_sum + df[col].fillna(0) * w * present
        weight_total = weight_total + (w * present)
    blend = weighted_sum / weight_total.replace(0, np.nan)
    return blend - HOME_FIELD_PTS


def generate_weight_grid(step: float = WEIGHT_STEP) -> list[tuple[float, float, float]]:
    """All (w_elo, w_epa, w_dvoa) combos on a grid that sum to 1.0."""
    n = round(1 / step)
    combos = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            combos.append((round(i * step, 2), round(j * step, 2), round(k * step, 2)))
    return combos


def main():
    scored = build_scored_dataset()

    train = scored[scored["season"].isin(TRAIN_SEASONS)].copy()
    test = scored[scored["season"].isin(TEST_SEASONS)].copy()
    print(f"\nTrain seasons {TRAIN_SEASONS}: {len(train)} games")
    print(f"Test seasons {TEST_SEASONS} (untouched holdout): {len(test)} games\n")

    print("Sweeping weight grid on TRAIN data only...")
    grid = generate_weight_grid()
    results = []
    for w_elo, w_epa, w_dvoa in grid:
        train_blend = weighted_blend(train, w_elo, w_epa, w_dvoa)
        train_scored = train.assign(blend=train_blend)
        m = score_model(train_scored, model_spread_col="blend", closing_spread_col="spread_line_standard", actual_margin_col="actual_margin")
        m.update({"w_elo": w_elo, "w_epa": w_epa, "w_dvoa": w_dvoa})
        results.append(m)

    grid_df = pd.DataFrame(results).sort_values("ats_win_rate", ascending=False)
    grid_df.to_csv("data/processed/weight_sweep_train_results.csv", index=False)

    print("\nTop 10 weight combos on TRAIN data (by ATS win rate):")
    print(grid_df[["w_elo", "w_epa", "w_dvoa", "mae", "ats_win_rate", "n_games"]].head(10).to_string(index=False))

    # Fair EPA-alone baseline: same home-field treatment as every grid combo
    # (w_elo=0, w_epa=1, w_dvoa=0 is already IN the grid -- pull it out directly
    # rather than scoring the raw epa_spread column, which has no HFA applied
    # and would make this an apples-to-oranges comparison).
    epa_alone_row = grid_df[(grid_df["w_elo"] == 0.0) & (grid_df["w_epa"] == 1.0) & (grid_df["w_dvoa"] == 0.0)].iloc[0]
    print(f"\nEPA-alone baseline on TRAIN (same HFA treatment as grid): MAE={epa_alone_row['mae']}, ATS={epa_alone_row['ats_win_rate']}")

    best = grid_df.iloc[0]
    print(f"\n{'=' * 70}")
    print(f"Best train combo: elo={best['w_elo']}, epa={best['w_epa']}, dvoa={best['w_dvoa']}")
    print(f"{'=' * 70}")

    print("\nEvaluating that SAME combo on the TEST holdout (2024-2025, never touched during search)...")
    test_blend = weighted_blend(test, best["w_elo"], best["w_epa"], best["w_dvoa"])
    test_scored = test.assign(blend=test_blend)
    test_result = score_model(test_scored, model_spread_col="blend", closing_spread_col="spread_line_standard", actual_margin_col="actual_margin")
    print(f"Holdout result: MAE={test_result['mae']}, RMSE={test_result['rmse']}, ATS={test_result['ats_win_rate']}, n={test_result['n_games']}")

    # Same fair comparison on holdout: EPA-alone WITH the same HFA treatment (w=1,0,0 blend, not raw epa_spread)
    epa_alone_test_blend = weighted_blend(test, 0.0, 1.0, 0.0)
    epa_alone_test_scored = test.assign(blend=epa_alone_test_blend)
    epa_alone_test = score_model(epa_alone_test_scored, model_spread_col="blend", closing_spread_col="spread_line_standard", actual_margin_col="actual_margin")
    print(f"EPA-alone on same holdout (same HFA treatment): MAE={epa_alone_test['mae']}, ATS={epa_alone_test['ats_win_rate']}")

    print(f"\n{'=' * 70}")
    print("VERDICT")
    print(f"{'=' * 70}")
    if test_result["ats_win_rate"] > epa_alone_test["ats_win_rate"]:
        print("The best train-set weighting DID generalize -- it beat EPA-alone on the untouched holdout too.")
    else:
        print("The best train-set weighting did NOT generalize -- it beat EPA-alone on train data")
        print("(where weights were fitted) but NOT on the holdout. This is the overfitting signature:")
        print("the 'best' weights were partly fitting noise in the train seasons, not a real, stable edge.")

    summary = pd.DataFrame([
        {"segment": "train (weights fit here)", "model": "EPA alone", **epa_alone_row.to_dict()},
        {"segment": "train (weights fit here)", "model": "Blended (best train weights)", **best.to_dict()},
        {"segment": "test (untouched holdout)", "model": "EPA alone", **epa_alone_test},
        {"segment": "test (untouched holdout)", "model": "Blended (same weights, out-of-sample)", **test_result},
    ])
    summary.to_csv("data/processed/weight_sweep_train_vs_holdout_summary.csv", index=False)
    print("\nFull comparison saved to data/processed/weight_sweep_train_vs_holdout_summary.csv")


if __name__ == "__main__":
    main()
