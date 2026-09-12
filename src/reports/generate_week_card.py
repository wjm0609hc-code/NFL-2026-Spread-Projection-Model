"""
Generates the forward-looking "week card": for each not-yet-played game in a
season/week, projects a spread using this repo's current adopted model (EPA +
turnover-luck + rest -- see docs/CHANGELOG.md's "Current model" section) and
prices it against a market line to surface cover probability and edge. This
is the forward-looking counterpart to weekly_model_output.py, which is a stub
waiting on a `projections` DataFrame to be wired in -- this script builds that
DataFrame from scratch for a given week and prints the resulting card.

No totals model exists in this repo yet, so this card is spread-only (no
over/under row).

Ratings are computed the same walk-forward-safe way as the backtest (see
src/features/epa_ratings.py, src/features/turnover_luck.py) but "as of right
now": each team's rating uses its trailing games up to and including its most
recently PLAYED game, so it's a valid input to that team's NEXT game
regardless of season boundary (mirrors the walk-forward logic, just evaluated
one step past the end of each team's history instead of at a historical row).

Live odds are attempted via SportsDataIO (src/ingestion/live_odds.py). If
that call fails (e.g. network policy blocks the host, or the week has no
odds posted yet), this falls back to the market spread already bundled in
the nflverse schedule data, clearly labeled as a fallback -- never fabricated.

Usage:
    python -m src.reports.generate_week_card --season 2026 --week 1
"""
from __future__ import annotations

import argparse

import pandas as pd
from scipy import stats

from src.backtest.run_full_backtest import fetch_pbp, fetch_schedules
from src.features.epa_ratings import compute_play_level_epa
from src.features.rest_weather import build_rest_adjustment
from src.features.turnover_luck import (
    LEAGUE_RECOVERY_RATE,
    MAX_ADJUSTMENT_PTS,
    POINTS_PER_TURNOVER,
    SHRINKAGE_K,
    build_fumble_events,
    build_team_week_fumble_stats,
)
from src.ingestion.live_odds import best_price_per_side, get_live_odds
from src.ingestion.sportsdataio_client import SportsDataIOClient
from src.models.monte_carlo import DEFAULT_MARGIN_SIGMA, T_DIST_DOF, classify_edge, edge_pct
from src.models.spread_projection import build_epa_spread

LOOKBACK_SEASONS = [2021, 2022, 2023, 2024, 2025]
EPA_ROLLING_WINDOW = 8
EPA_MIN_GAMES = 4
TURNOVER_ROLLING_GAMES = 8


def latest_epa_ratings(
    team_week_epa: pd.DataFrame, rolling_window: int = EPA_ROLLING_WINDOW, min_games: int = EPA_MIN_GAMES
) -> pd.DataFrame:
    """
    Each team's EPA rating entering its NEXT game: mean of its trailing
    `rolling_window` games INCLUDING its most recently played game. This
    differs from epa_ratings.build_rolling_ratings(), which excludes the
    current row at each step to stay walk-forward-safe for HISTORICAL games;
    here there's no current game yet to exclude, so the trailing window
    includes everything played so far.
    """
    rows = []
    for team, grp in team_week_epa.sort_values(["season", "week"]).groupby("team"):
        trailing = grp.tail(rolling_window)
        rows.append(
            {
                "team": team,
                "off_epa_rating": trailing["off_epa_play"].mean(),
                "def_epa_rating": trailing["def_epa_play"].mean(),
                "games_sample": len(trailing),
                "last_played_season": grp["season"].iloc[-1],
                "last_played_week": grp["week"].iloc[-1],
            }
        )
    ratings = pd.DataFrame(rows)
    ratings["low_confidence"] = ratings["games_sample"] < min_games
    return ratings


def latest_turnover_adjustment(team_week_fumbles: pd.DataFrame, rolling_games: int = TURNOVER_ROLLING_GAMES) -> pd.DataFrame:
    """
    Same shrinkage/cap logic as turnover_luck.build_walk_forward_turnover_luck()
    + build_turnover_adjustment(), but evaluated "as of now" (trailing window
    including each team's most recently played game) rather than at a
    historical row -- see latest_epa_ratings() for why.
    """
    rows = []
    for team, grp in team_week_fumbles.sort_values(["season", "week"]).groupby("team"):
        trailing = grp.tail(rolling_games)
        total_fumbles = trailing["fumbles_total"].sum()
        if total_fumbles == 0:
            luck_index = 0.0
        else:
            actual_recovered = trailing["fumbles_recovered_self"].sum()
            expected_recovered = total_fumbles * LEAGUE_RECOVERY_RATE
            raw_luck = actual_recovered - expected_recovered
            shrink_weight = total_fumbles / (total_fumbles + SHRINKAGE_K)
            luck_index = raw_luck * shrink_weight
        adj_pts = float(max(-MAX_ADJUSTMENT_PTS, min(MAX_ADJUSTMENT_PTS, -luck_index * POINTS_PER_TURNOVER)))
        rows.append({"team": team, "turnover_adjustment_pts": adj_pts, "fumble_sample": int(total_fumbles)})
    return pd.DataFrame(rows)


def prob_home_covers(
    projected_margin_home: float, market_home_spread_standard: float, sigma: float = DEFAULT_MARGIN_SIGMA, dof: int = T_DIST_DOF
) -> float:
    """
    Analytic equivalent of monte_carlo.prob_covers_spread(), for use without a
    projected_total (this card is spread-only -- see module docstring). Models
    margin directly as Student's-t around the projected margin, same
    sigma/dof monte_carlo.py uses, so results are consistent with the rest of
    the pipeline without needing to simulate/decompose a total.

    market_home_spread_standard: standard convention, negative = home favored.
    projected_margin_home: positive = home favored (monte_carlo.py convention).
    """
    threshold = -market_home_spread_standard
    z = (threshold - projected_margin_home) / sigma
    return float(stats.t.sf(z, dof))


def describe_favorite(spread_standard: float, home_team: str, away_team: str) -> str:
    """spread_standard: negative = home favored (spread_projection.py convention)."""
    if pd.isna(spread_standard):
        return "n/a"
    if spread_standard < -0.05:
        return f"{home_team} favored by {-spread_standard:.1f}"
    elif spread_standard > 0.05:
        return f"{away_team} favored by {spread_standard:.1f}"
    return "pick'em"


def build_week_card(season: int, week: int) -> None:
    seasons_needed = sorted(set(LOOKBACK_SEASONS) | {season})

    print(f"Fetching play-by-play for rating lookback ({seasons_needed})...")
    pbp = fetch_pbp(seasons_needed)

    print("Fetching schedules...")
    sched = fetch_schedules(seasons_needed)

    target_games = sched[(sched["season"] == season) & (sched["week"] == week)].copy()
    if target_games.empty:
        print(f"No games found for season {season} week {week}.")
        return

    played = target_games[target_games["actual_margin"].notna()]
    upcoming = target_games[target_games["actual_margin"].isna()].copy()

    if len(played):
        print(f"\n{len(played)} game(s) in week {week} already played (excluded from the card):")
        for _, g in played.sort_values("gameday").iterrows():
            print(f"  {g['away_team']} {g['away_score']:.0f} @ {g['home_team']} {g['home_score']:.0f} ({g['gameday']})")

    if upcoming.empty:
        print(f"\nAll week {week} games have been played -- nothing left to project.")
        return

    print("\nBuilding EPA ratings (as of now, trailing games through each team's most recent played game)...")
    team_week_epa = compute_play_level_epa(pbp)
    epa_ratings_now = latest_epa_ratings(team_week_epa)

    print("Building turnover-luck adjustment (as of now)...")
    fumble_events = build_fumble_events(pbp)
    team_week_fumbles = build_team_week_fumble_stats(fumble_events)
    turnover_now = latest_turnover_adjustment(team_week_fumbles)

    print("Building rest adjustment...")
    rest_adj = build_rest_adjustment(upcoming)

    # build_epa_spread() expects an epa_ratings table keyed by (season, week,
    # team); stamp our "as of now" ratings with the target season/week so the
    # merge lines up against this week's games.
    epa_ratings_stamped = epa_ratings_now.assign(season=season, week=week)
    games_for_epa = upcoming[["season", "week", "home_team", "away_team"]]
    epa_spread_df = build_epa_spread(games_for_epa, epa_ratings_stamped)

    result = upcoming.merge(
        epa_spread_df[["season", "week", "home_team", "away_team", "epa_spread"]],
        on=["season", "week", "home_team", "away_team"],
        how="left",
    )
    result = result.merge(
        epa_ratings_now[["team", "games_sample", "low_confidence"]].rename(
            columns={"team": "home_team", "games_sample": "home_games_sample", "low_confidence": "home_low_confidence"}
        ),
        on="home_team",
        how="left",
    )
    result = result.merge(
        epa_ratings_now[["team", "games_sample", "low_confidence"]].rename(
            columns={"team": "away_team", "games_sample": "away_games_sample", "low_confidence": "away_low_confidence"}
        ),
        on="away_team",
        how="left",
    )
    result = result.merge(
        turnover_now.rename(columns={"team": "home_team", "turnover_adjustment_pts": "home_to_adj"})[["home_team", "home_to_adj"]],
        on="home_team",
        how="left",
    )
    result = result.merge(
        turnover_now.rename(columns={"team": "away_team", "turnover_adjustment_pts": "away_to_adj"})[["away_team", "away_to_adj"]],
        on="away_team",
        how="left",
    )
    result = result.merge(
        rest_adj[["season", "week", "home_team", "away_team", "rest_adjustment_pts"]],
        on=["season", "week", "home_team", "away_team"],
        how="left",
    )
    result["home_to_adj"] = result["home_to_adj"].fillna(0.0)
    result["away_to_adj"] = result["away_to_adj"].fillna(0.0)
    result["rest_adjustment_pts"] = result["rest_adjustment_pts"].fillna(0.0)

    # current adopted model, per docs/CHANGELOG.md "Current model": EPA + turnover luck + rest
    result["model_spread"] = result["epa_spread"] - result["home_to_adj"] + result["away_to_adj"] - result["rest_adjustment_pts"]
    # monte_carlo.py convention: projected_margin positive = home favored (opposite of model_spread's sign)
    result["projected_margin_home"] = -result["model_spread"]

    print("\nAttempting live odds from SportsDataIO...")
    live_odds_df = None
    odds_failure = None
    try:
        client = SportsDataIOClient()
        odds_raw = get_live_odds(client, season, week)
        if odds_raw.empty:
            odds_failure = "SportsDataIO returned no odds rows for this season/week."
        else:
            live_odds_df = best_price_per_side(odds_raw)
    except Exception as e:
        odds_failure = f"{type(e).__name__}: {e}"

    if live_odds_df is not None:
        result = result.merge(live_odds_df, on=["home_team", "away_team"], how="left")
        # SportsDataIO's HomePointSpread already follows standard book convention (negative = home favored)
        result["market_spread_standard"] = result["consensus_home_spread"]
        market_source = "SportsDataIO live consensus"
    else:
        print(f"  Live odds unavailable ({odds_failure}).")
        print("  Falling back to the market spread already bundled in the nflverse schedule data")
        print("  (real market data, just not a live SportsDataIO pull) -- labeled below.")
        result["market_spread_standard"] = result["spread_line_standard"]
        market_source = "nflverse bundled spread_line (FALLBACK -- not a live SportsDataIO pull; see note above)"

    print("\n" + "=" * 72)
    print(f"WEEK CARD -- Season {season}, Week {week}")
    print(f"Market source: {market_source}")
    print("Model: EPA + turnover-luck + rest (current adopted model per docs/CHANGELOG.md)")
    print("No totals model exists yet in this repo -- spread only, no over/under.")
    print("=" * 72)

    for _, g in result.sort_values("gameday").iterrows():
        home, away = g["home_team"], g["away_team"]
        model_spread = g["model_spread"]
        market_spread = g["market_spread_standard"]

        cover_prob = None
        edge = None
        if pd.notna(market_spread):
            cover_prob = prob_home_covers(g["projected_margin_home"], market_spread)
            edge = edge_pct(cover_prob, -110)

        print(f"\n{away} @ {home}  ({g['gameday']})")
        print(f"  Model:  {describe_favorite(model_spread, home, away):<28} (home-perspective spread: {model_spread:+.1f})")
        if pd.notna(market_spread):
            print(f"  Market: {describe_favorite(market_spread, home, away):<28} (home-perspective spread: {market_spread:+.1f})")
            print(f"  P({home} covers market line) = {cover_prob:.1%}   Edge vs -110 juice: {edge:+.1%} ({classify_edge(edge)})")
        else:
            print("  Market: no line available")

        home_conf = " (LOW CONFIDENCE)" if g.get("home_low_confidence") else ""
        away_conf = " (LOW CONFIDENCE)" if g.get("away_low_confidence") else ""
        home_gs = int(g["home_games_sample"]) if pd.notna(g.get("home_games_sample")) else 0
        away_gs = int(g["away_games_sample"]) if pd.notna(g.get("away_games_sample")) else 0
        print(f"  Rating sample: {home} {home_gs} games{home_conf} | {away} {away_gs} games{away_conf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()
    build_week_card(args.season, args.week)
