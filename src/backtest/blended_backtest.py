"""
Blended (Elo + EPA + DVOA) spread backtest, with segmentation by week bucket
to test whether the model is more effective later in the season (once EPA's
rolling window and DVOA both have more games of current-season signal to
work with) vs. early weeks where both are running on thin/preseason-adjacent
data.

Home-field handling: Elo's standalone implied spread already bakes in ~2.6 pts
of home-field advantage (via HOME_FIELD_ELO_PTS in elo_baseline.py). EPA and
DVOA standalone spreads do NOT include any home-field adjustment. Naively
averaging all three would double-count home-field for Elo's share of the
blend. Fixed here by computing a home-field-NEUTRAL Elo spread (straight
rating difference, no home bonus) before blending, then applying a single
HOME_FIELD_PTS constant once, at the end, to the blended result.

Usage:
    python -m src.backtest.blended_backtest
"""
from __future__ import annotations

import pandas as pd

from src.backtest.harness import score_model
from src.backtest.run_full_backtest import fetch_pbp, fetch_schedules
from src.features.dvoa_loader import attach_dvoa_to_games, build_walk_forward_dvoa, load_dvoa
from src.features.epa_ratings import build_rolling_ratings, compute_play_level_epa
from src.models.elo_baseline import run_elo_backtest
from src.models.spread_projection import build_dvoa_spread, build_epa_spread

SEASONS = [2021, 2022, 2023, 2024, 2025]
HOME_FIELD_PTS = 1.8  # applied once, here, to the blended spread only


def build_elo_spread_no_hfa(elo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Home-field-neutral Elo spread: uses the raw pre-game ratings with no
    home-field bonus added, so it can be safely averaged with EPA/DVOA
    (which also carry no home-field adjustment) without double-counting.
    """
    out = elo_df.copy()
    out["elo_spread_no_hfa"] = -(out["pre_game_home_elo"] - out["pre_game_away_elo"]) / 25.0
    return out[["season", "week", "home_team", "away_team", "elo_spread_no_hfa"]]


def build_scored_dataset() -> pd.DataFrame:
    print(f"Fetching play-by-play and schedules for {SEASONS}...")
    pbp = fetch_pbp(SEASONS)
    schedules = fetch_schedules(SEASONS)

    print("Building Elo (home-field-neutral for blending)...")
    elo_df = run_elo_backtest(schedules)
    elo_spread_df = build_elo_spread_no_hfa(elo_df)

    print("Building EPA ratings...")
    team_week_epa = compute_play_level_epa(pbp)
    epa_ratings = build_rolling_ratings(team_week_epa)
    epa_spread_df = build_epa_spread(schedules[["season", "week", "home_team", "away_team"]], epa_ratings)

    print("Loading DVOA (lagged)...")
    dvoa_raw = load_dvoa(SEASONS)
    dvoa_wf = build_walk_forward_dvoa(dvoa_raw, lag_weeks=1)
    games_with_dvoa = attach_dvoa_to_games(schedules[["season", "week", "home_team", "away_team"]], dvoa_wf)
    dvoa_spread_df = build_dvoa_spread(games_with_dvoa)

    print("Merging and blending...")
    scored = schedules.merge(
        elo_spread_df, on=["season", "week", "home_team", "away_team"], how="left"
    ).merge(
        epa_spread_df[["season", "week", "home_team", "away_team", "epa_spread"]],
        on=["season", "week", "home_team", "away_team"], how="left",
    ).merge(
        dvoa_spread_df[["season", "week", "home_team", "away_team", "dvoa_spread"]],
        on=["season", "week", "home_team", "away_team"], how="left",
    )

    source_cols = ["elo_spread_no_hfa", "epa_spread", "dvoa_spread"]
    scored["blended_spread"] = scored[source_cols].mean(axis=1, skipna=True) - HOME_FIELD_PTS

    # also carry the original (HFA-inclusive) elo spread and standalone epa/dvoa
    # for the head-to-head comparison table
    scored["elo_spread"] = scored["elo_spread_no_hfa"] - HOME_FIELD_PTS  # apply HFA once here too, for fair comparison

    scored["week_bucket"] = scored["week"].apply(lambda w: "Wk 1-4 (early)" if w <= 4 else "Wk 5+ (established)")
    return scored


def main():
    scored = build_scored_dataset()

    print("\n" + "=" * 70)
    print("OVERALL: blended vs. each standalone component")
    print("=" * 70)
    overall_rows = []
    for label, col in [("Elo (HFA-adj)", "elo_spread"), ("EPA", "epa_spread"), ("DVOA", "dvoa_spread"), ("BLENDED", "blended_spread")]:
        m = score_model(scored, model_spread_col=col, closing_spread_col="spread_line_standard", actual_margin_col="actual_margin")
        m["model"] = label
        overall_rows.append(m)
        print(f"\n{label}: MAE={m['mae']}, RMSE={m['rmse']}, ATS={m['ats_win_rate']}, n={m['n_games']}")

    overall_df = pd.DataFrame(overall_rows).set_index("model")

    print("\n" + "=" * 70)
    print("SEGMENTED: blended spread, early weeks (1-4) vs. established (5+)")
    print("=" * 70)
    segment_rows = []
    for bucket, grp in scored.groupby("week_bucket"):
        m = score_model(grp, model_spread_col="blended_spread", closing_spread_col="spread_line_standard", actual_margin_col="actual_margin")
        m["week_bucket"] = bucket
        segment_rows.append(m)
        print(f"\n{bucket}: MAE={m['mae']}, RMSE={m['rmse']}, ATS={m['ats_win_rate']}, n={m['n_games']}")

    segment_df = pd.DataFrame(segment_rows).set_index("week_bucket")

    print("\n" + "=" * 70)
    print("PER-WEEK breakdown (blended spread) — finer resolution")
    print("=" * 70)
    per_week_rows = []
    for wk, grp in scored.groupby("week"):
        m = score_model(grp, model_spread_col="blended_spread", closing_spread_col="spread_line_standard", actual_margin_col="actual_margin")
        m["week"] = wk
        per_week_rows.append(m)
    per_week_df = pd.DataFrame(per_week_rows).set_index("week").sort_index()
    print(per_week_df[["n_games", "mae", "rmse", "ats_win_rate"]].to_string())

    overall_df.to_csv("data/processed/blended_backtest_overall.csv")
    segment_df.to_csv("data/processed/blended_backtest_by_week_bucket.csv")
    per_week_df.to_csv("data/processed/blended_backtest_by_week.csv")
    scored.to_parquet("data/processed/blended_backtest_scored_games.parquet")

    print("\nSaved: blended_backtest_overall.csv, blended_backtest_by_week_bucket.csv, blended_backtest_by_week.csv")


if __name__ == "__main__":
    main()
