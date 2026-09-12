"""
Turnover luck adjustment: fumble recovery rate is close to a coin flip
league-wide (confirmed empirically at 52.1% in this dataset, matching the
well-established finding that recovering your own fumble is not a repeatable
skill). Because a recovered fumble costs almost no EPA while a lost fumble
costs a lot, recovery luck gets baked directly into a team's trailing EPA
rating AS IF it were skill. This layer estimates how much of a team's recent
EPA rating is attributable to fumble-recovery luck (good or bad) and produces
a point adjustment that pulls the team's projection back toward what its
underlying performance would suggest absent that luck.

This is NOT the same kind of adjustment as the QB layer: it isn't
information the market has to react to (there's no headline event), it's a
statistical correction to a rating that's silently absorbing noise as signal.
That's exactly the kind of edge that's plausible to still exist.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LEAGUE_RECOVERY_RATE = 0.50   # a team recovering its own fumble is close to a coin flip;
                               # 0.50 is the standard assumption in the public research this is based on
                               # (measured 52.1% in this dataset -- close enough that 0.50 is the more
                               # defensible, less overfit prior to use).
POINTS_PER_TURNOVER = 4.0     # commonly cited industry estimate for the point value of a turnover
                               # (field position + lost possession value combined). A starting prior,
                               # not independently derived here.
ROLLING_GAMES = 8             # trailing window, matches epa_ratings.py's window for consistency
SHRINKAGE_K = 15              # shrinkage strength in fumble count -- an 8-game window often has only
                               # 5-15 total fumbles for a team, so raw luck_index swings wildly on tiny
                               # samples. Shrinkage pulls small-sample luck estimates toward 0 (no luck
                               # assumed) rather than trusting a 3-fumble sample at full strength.
MAX_ADJUSTMENT_PTS = 7.0      # hard cap -- unshrunk values were producing 20-28 point swings, which is
                               # not a plausible single-game turnover-luck effect at any sample size.


def build_fumble_events(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per fumble with recovering team, keyed to the fumbling team's game."""
    fum = pbp[(pbp["fumble"] == 1) & pbp["fumbled_1_team"].notna() & pbp["fumble_recovery_1_team"].notna()].copy()
    fum["fumbling_team"] = fum["fumbled_1_team"]
    fum["recovered_by_fumbling_team"] = (fum["fumbled_1_team"] == fum["fumble_recovery_1_team"]).astype(int)
    return fum[["season", "week", "fumbling_team", "recovered_by_fumbling_team"]]


def build_team_week_fumble_stats(fumble_events: pd.DataFrame) -> pd.DataFrame:
    """
    Per (season, week, team): how many fumbles that team had (offense/ST side),
    and how many it recovered itself, vs. the league-average-implied recovery
    count for that same volume of fumbles.
    """
    grouped = (
        fumble_events.groupby(["season", "week", "fumbling_team"])
        .agg(fumbles_total=("recovered_by_fumbling_team", "size"), fumbles_recovered_self=("recovered_by_fumbling_team", "sum"))
        .reset_index()
        .rename(columns={"fumbling_team": "team"})
    )
    return grouped


def build_walk_forward_turnover_luck(team_week_fumbles: pd.DataFrame, rolling_games: int = ROLLING_GAMES) -> pd.DataFrame:
    """
    Trailing (walk-forward safe, across season boundaries) turnover luck index
    per team: actual self-recoveries minus expected self-recoveries at the
    league-average rate, over the trailing window. Positive = has been
    recovering MORE of its own fumbles than expected (lucky -- due for
    negative regression). Negative = recovering fewer than expected (unlucky
    -- due for positive regression).
    """
    df = team_week_fumbles.sort_values(["team", "season", "week"]).copy()
    out_rows = []
    for team, grp in df.groupby("team"):
        grp = grp.reset_index(drop=True)
        for i in range(len(grp)):
            history = grp.iloc[:i].tail(rolling_games)
            season, week = grp.loc[i, "season"], grp.loc[i, "week"]
            total_fumbles = history["fumbles_total"].sum()
            if total_fumbles == 0:
                luck_index, sample = 0.0, 0
            else:
                actual_recovered = history["fumbles_recovered_self"].sum()
                expected_recovered = total_fumbles * LEAGUE_RECOVERY_RATE
                raw_luck = actual_recovered - expected_recovered  # in units of "fumbles luckier than expected"
                shrink_weight = total_fumbles / (total_fumbles + SHRINKAGE_K)
                luck_index = raw_luck * shrink_weight  # shrink toward 0 (no assumed luck) on small samples
                sample = total_fumbles
            out_rows.append({"season": season, "week": week, "team": team, "turnover_luck_index": luck_index, "fumble_sample": sample})
    return pd.DataFrame(out_rows)


def build_turnover_adjustment(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Full pipeline: one row per (season, week, team) with turnover_adjustment_pts.
    NEGATIVE luck_index (unlucky) -> POSITIVE point adjustment (expect positive regression).
    POSITIVE luck_index (lucky) -> NEGATIVE point adjustment (expect negative regression).
    """
    fumble_events = build_fumble_events(pbp)
    team_week = build_team_week_fumble_stats(fumble_events)
    luck = build_walk_forward_turnover_luck(team_week)

    # a fumble recovered by the OTHER team is what actually costs points (a turnover) --
    # each "extra" self-recovery beyond expected represents roughly one turnover AVOIDED
    # relative to the baseline rate, so we convert luck_index directly to points and flip
    # sign: being luckier than expected means the team's rating is inflated by that amount,
    # so the correction (adjustment applied to the projection) is negative.
    luck["turnover_adjustment_pts"] = np.clip(-luck["turnover_luck_index"] * POINTS_PER_TURNOVER, -MAX_ADJUSTMENT_PTS, MAX_ADJUSTMENT_PTS)
    return luck[["season", "week", "team", "turnover_luck_index", "fumble_sample", "turnover_adjustment_pts"]]


if __name__ == "__main__":
    print("Import build_turnover_adjustment(pbp) -- see src/backtest/turnover_adjustment_backtest.py for usage.")
