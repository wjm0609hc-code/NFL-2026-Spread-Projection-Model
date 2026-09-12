"""
Backtest harness: scores a model's projected spreads against actual results
AND against the market closing line. Beating the closing line consistently
(positive CLV) is the real bar — a model can "predict winners" reasonably
well and still have zero betting edge if it's just re-deriving the market.

Metrics reported:
  - MAE / RMSE of projected margin vs. actual margin
  - ATS win rate (using the model's own side vs. the closing spread)
  - CLV: average (model_spread - closing_spread) signed toward the side the
    model favors, i.e. does the model find value the market didn't have yet
  - Segment breakdowns: divisional games, primetime, rest mismatch, QB change
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def score_model(
    games: pd.DataFrame,
    model_spread_col: str,
    closing_spread_col: str = "spread_line",
    actual_margin_col: str = "actual_margin",
) -> dict:
    """
    games: one row per game, must contain model_spread_col, closing_spread_col,
    actual_margin_col (home_score - away_score), all from the home team's perspective,
    using standard convention: negative = home favored.

    Returns a dict of summary metrics.
    """
    df = games.dropna(subset=[model_spread_col, closing_spread_col, actual_margin_col]).copy()

    df["error"] = df[model_spread_col] - df[actual_margin_col]
    mae = df["error"].abs().mean()
    rmse = np.sqrt((df["error"] ** 2).mean())

    # ATS: did the side the model favored (relative to closing line) cover?
    # model_edge > 0 means model likes home more than market does (model spread is more negative / home more favored)
    df["model_edge"] = df[closing_spread_col] - df[model_spread_col]
    df["model_favors_home_more"] = df["model_edge"] > 0

    # cover margin relative to closing spread: actual_margin + closing_spread > 0 means home covered
    df["home_covered"] = (df[actual_margin_col] + df[closing_spread_col]) > 0

    # model "wins" the ATS bet if it favored home-more and home covered, or favored away-more and home didn't cover
    df["model_ats_win"] = np.where(
        df["model_favors_home_more"], df["home_covered"], ~df["home_covered"]
    )
    # exclude pushes (edge == 0) from ATS win rate denominator
    ats_eligible = df[df["model_edge"] != 0]
    ats_win_rate = ats_eligible["model_ats_win"].mean() if len(ats_eligible) else np.nan

    return {
        "n_games": len(df),
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "ats_win_rate": round(ats_win_rate, 4) if pd.notna(ats_win_rate) else None,
        "ats_eligible_games": len(ats_eligible),
        "mean_model_edge_vs_close": round(df["model_edge"].mean(), 3),
    }


def segment_backtest(
    games: pd.DataFrame,
    model_spread_col: str,
    segment_col: str,
    closing_spread_col: str = "spread_line",
    actual_margin_col: str = "actual_margin",
) -> pd.DataFrame:
    """Run score_model separately for each value of segment_col (e.g. 'divisional',
    'primetime', 'rest_mismatch_flag') to find where the model has real edge vs. noise."""
    results = []
    for segment_val, grp in games.groupby(segment_col):
        metrics = score_model(grp, model_spread_col, closing_spread_col, actual_margin_col)
        metrics[segment_col] = segment_val
        results.append(metrics)
    return pd.DataFrame(results).set_index(segment_col)


def walk_forward_check(games: pd.DataFrame, feature_cols: list[str], game_date_col: str = "gameday") -> None:
    """
    Sanity check to catch leakage: asserts that no feature column contains
    information that couldn't have existed before game_date_col. This doesn't
    catch everything automatically — it's a placeholder for spot-checks you
    should run per feature (e.g., confirm epa_ratings for week W only used
    plays from week < W). Raises if any feature col is entirely null (likely
    a join bug) or if games are not sorted chronologically.
    """
    if not games[game_date_col].is_monotonic_increasing:
        raise ValueError("Games are not sorted chronologically — walk-forward backtest requires sorted input.")
    for col in feature_cols:
        if games[col].isna().all():
            raise ValueError(f"Feature column '{col}' is entirely null — check upstream join.")
    print("Walk-forward sanity check passed (sorted order + no fully-null feature columns).")


if __name__ == "__main__":
    print("Import score_model / segment_backtest / walk_forward_check into a backtest notebook or script.")
