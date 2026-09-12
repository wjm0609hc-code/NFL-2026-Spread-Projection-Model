"""
Produces the actual weekly output view: for each game this week, combines
your model's simulated projection with live odds to surface probability and
edge %, per side, per book — the FTN-style deliverable.

Usage (once live odds + a fitted projection are available):
    python -m src.reports.weekly_model_output --season 2026 --week 2
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.ingestion.live_odds import best_price_per_side, get_live_odds
from src.ingestion.sportsdataio_client import SportsDataIOClient
from src.models.monte_carlo import classify_edge, edge_pct, prob_covers_spread, prob_over_total, simulate_game


def build_weekly_output(
    projections: pd.DataFrame,
    season: int,
    week: int,
) -> pd.DataFrame:
    """
    projections: one row per game this week, must have home_team, away_team,
    projected_margin (positive = home favored), projected_total.

    Pulls live odds, simulates each game, and returns one row per game with
    median projected score, cover probability, edge %, and best book per side.

    NOTE on injuries (per FTN's own caveat): this pipeline runs projections
    assuming currently-listed starters play. If a key player is questionable,
    manually haircut projected_margin/projected_total before calling this, or
    the model will look artificially confident on that side — same failure
    mode FTN explicitly warns about.
    """
    client = SportsDataIOClient()
    odds_raw = get_live_odds(client, season, week)
    best_odds = best_price_per_side(odds_raw)

    merged = projections.merge(best_odds, on=["home_team", "away_team"], how="left")

    output_rows = []
    for _, row in merged.iterrows():
        sim = simulate_game(
            projected_margin=row["projected_margin"],
            projected_total=row["projected_total"],
        )

        home_spread_line = row.get("best_home_spread")
        total_line = row.get("consensus_total")

        home_cover_prob = prob_covers_spread(sim, home_spread_line) if pd.notna(home_spread_line) else None
        over_prob = prob_over_total(sim, total_line) if pd.notna(total_line) else None

        home_spread_edge = (
            edge_pct(home_cover_prob, -110) if home_cover_prob is not None else None
        )  # assumes -110 vig unless best_home_spread_odds is populated; refine if you want exact odds per book
        over_edge = edge_pct(over_prob, -110) if over_prob is not None else None

        output_rows.append(
            {
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "median_home_score": sim["median_home_score"],
                "median_away_score": sim["median_away_score"],
                "market_home_spread": home_spread_line,
                "home_cover_prob": home_cover_prob,
                "home_spread_edge_pct": home_spread_edge,
                "home_spread_edge_class": classify_edge(home_spread_edge) if home_spread_edge is not None else None,
                "best_home_spread_book": row.get("best_home_spread_book"),
                "market_total": total_line,
                "over_prob": over_prob,
                "over_edge_pct": over_edge,
                "over_edge_class": classify_edge(over_edge) if over_edge is not None else None,
                "best_over_book": row.get("best_over_book"),
            }
        )

    return pd.DataFrame(output_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()

    print(
        "This script expects a `projections` DataFrame (home_team, away_team, "
        "projected_margin, projected_total) built from your spread_projection.py "
        "output for the given week — wire that up, then call build_weekly_output()."
    )
