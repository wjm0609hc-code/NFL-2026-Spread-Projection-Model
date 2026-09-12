"""
Live odds ingestion — pulls real-time spread/total lines across sportsbooks
via SportsDataIO, normalizes into a flat table, and identifies the best
available price per side (for the "which book gives you the best number"
feature FTN's tool has).

Unlike the historical odds pulls used for backtesting (which cache aggressively
since the past doesn't change), this module is meant to be re-run close to
game time and does NOT cache — lines move.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.ingestion.sportsdataio_client import SportsDataIOClient


def get_live_odds(client: SportsDataIOClient, season: int, week: int) -> pd.DataFrame:
    """
    Pulls current-week odds across all available sportsbooks and flattens
    SportsDataIO's nested per-book structure into one row per (game, sportsbook).

    Expected raw shape (SportsDataIO GameOddsByWeek): each game object contains
    a 'PregameOdds' list, one entry per sportsbook, with fields like
    SportsbookId, HomePointSpread, HomePointSpreadPayout, AwayPointSpread,
    AwayPointSpreadPayout, OverUnder, OverPayout, UnderPayout.
    Field names can drift slightly by API version — verify against a live pull
    and adjust the parsing below if SportsDataIO's schema differs.
    """
    raw = client._get(f"/odds/json/GameOddsByWeek/{season}/{week}", use_cache=False)

    rows = []
    for game in raw:
        game_id = game.get("GameId")
        home = game.get("HomeTeam")
        away = game.get("AwayTeam")
        for book_line in game.get("PregameOdds", []):
            rows.append(
                {
                    "game_id": game_id,
                    "home_team": home,
                    "away_team": away,
                    "sportsbook": book_line.get("Sportsbook"),
                    "home_spread": book_line.get("HomePointSpread"),
                    "home_spread_odds": book_line.get("HomePointSpreadPayout"),
                    "away_spread": book_line.get("AwayPointSpread"),
                    "away_spread_odds": book_line.get("AwayPointSpreadPayout"),
                    "total": book_line.get("OverUnder"),
                    "over_odds": book_line.get("OverPayout"),
                    "under_odds": book_line.get("UnderPayout"),
                    "pulled_at": datetime.now(timezone.utc).isoformat(),
                }
            )
    return pd.DataFrame(rows)


def best_price_per_side(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each game, finds the sportsbook offering the most favorable number for
    each side of the spread and each side of the total — i.e. the best number
    to bet at, mirroring FTN's "which sportsbook offers the most favorable odds."

    'Most favorable' for a spread bettor = most points (least negative spread
    if favored, most positive if underdog) combined with best odds; here we
    keep it simple and surface both the best line (points) and best odds
    separately so you can judge line vs. price tradeoffs yourself.
    """
    results = []
    for game_id, grp in odds_df.groupby("game_id"):
        home_team = grp["home_team"].iloc[0]
        away_team = grp["away_team"].iloc[0]

        best_home_spread_row = grp.loc[grp["home_spread"].idxmax()] if grp["home_spread"].notna().any() else None
        best_away_spread_row = grp.loc[grp["away_spread"].idxmax()] if grp["away_spread"].notna().any() else None
        best_over_odds_row = grp.loc[grp["over_odds"].idxmax()] if grp["over_odds"].notna().any() else None
        best_under_odds_row = grp.loc[grp["under_odds"].idxmax()] if grp["under_odds"].notna().any() else None

        results.append(
            {
                "game_id": game_id,
                "home_team": home_team,
                "away_team": away_team,
                "best_home_spread": best_home_spread_row["home_spread"] if best_home_spread_row is not None else None,
                "best_home_spread_book": best_home_spread_row["sportsbook"] if best_home_spread_row is not None else None,
                "best_away_spread": best_away_spread_row["away_spread"] if best_away_spread_row is not None else None,
                "best_away_spread_book": best_away_spread_row["sportsbook"] if best_away_spread_row is not None else None,
                "best_over_odds": best_over_odds_row["over_odds"] if best_over_odds_row is not None else None,
                "best_over_book": best_over_odds_row["sportsbook"] if best_over_odds_row is not None else None,
                "best_under_odds": best_under_odds_row["under_odds"] if best_under_odds_row is not None else None,
                "best_under_book": best_under_odds_row["sportsbook"] if best_under_odds_row is not None else None,
                "consensus_home_spread": grp["home_spread"].median(),
                "consensus_total": grp["total"].median(),
                "n_books": grp["sportsbook"].nunique(),
            }
        )
    return pd.DataFrame(results)


if __name__ == "__main__":
    client = SportsDataIOClient()
    # Example — set to the current season/week before running live
    # odds = get_live_odds(client, season=2026, week=2)
    # best = best_price_per_side(odds)
    # print(best)
    print("Set season/week and uncomment the example block to pull live odds.")
