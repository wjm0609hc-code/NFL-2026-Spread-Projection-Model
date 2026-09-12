"""
Team-level Monte Carlo simulation.

Instead of outputting a single point-estimate spread, this treats the model's
projected margin (from src/models/spread_projection.py) as the MEAN of a
distribution and simulates N game outcomes around it. This gets you the same
shape of output as FTN's model — a median projected score, and probabilities
that a given spread/total will hit — without building full player-level
drive simulation.

Key design choices:
  - Margin distribution: Student's t (fatter tails than normal — NFL margins
    have more blowouts/upsets than a normal distribution predicts). Falls back
    to normal if you prefer; both are exposed.
  - Sigma (std dev of margin) is NOT guessed — it should be calibrated from
    your backtest residuals (actual_margin - model_spread) across the 5-season
    sample. A placeholder league-average sigma (~13.5 points, roughly matching
    historical NFL margin variance) is used until you've run that calibration.
  - Total (combined score) simulated as a separate, correlated draw so you can
    price totals as well as spreads, matching FTN's two-pronged tool.
  - Correlation between margin and total is modeled via a shared "pace/error"
    factor rather than treated as fully independent, since high-scoring games
    tend to have more variance in margin too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_MARGIN_SIGMA = 13.5   # placeholder — REPLACE with your backtest residual std dev
DEFAULT_TOTAL_SIGMA = 10.0    # placeholder — REPLACE with your backtest residual std dev on totals
T_DIST_DOF = 8                # degrees of freedom for Student's t (lower = fatter tails)
N_SIMULATIONS = 10_000
MARGIN_TOTAL_CORR = 0.15      # mild positive correlation: shootouts correlate with bigger margins


def calibrate_sigma_from_backtest(scored_games: pd.DataFrame, model_col: str, actual_col: str) -> float:
    """
    Run this against your backtest output (src/backtest/harness.py results) to get
    a real sigma instead of the placeholder. Returns the std dev of residuals.
    """
    residuals = scored_games[actual_col] - scored_games[model_col]
    return float(residuals.std())


def simulate_game(
    projected_margin: float,
    projected_total: float,
    margin_sigma: float = DEFAULT_MARGIN_SIGMA,
    total_sigma: float = DEFAULT_TOTAL_SIGMA,
    n_sims: int = N_SIMULATIONS,
    dof: int = T_DIST_DOF,
    seed: int | None = None,
) -> dict:
    """
    projected_margin: home perspective, positive = home favored (note: this is the
                       OPPOSITE sign convention from spread_projection.py's "negative
                       = home favored" — flip sign when passing in your model's spread).
    projected_total:  projected combined score (home_score + away_score).

    Returns simulated home_score / away_score arrays plus summary stats, so you
    can compute probability of any given spread/total hitting.
    """
    rng = np.random.default_rng(seed)

    # correlated draws: shared latent factor pushes both margin and total together slightly
    shared = rng.standard_normal(n_sims)
    margin_noise = stats.t.rvs(dof, size=n_sims, random_state=rng)
    total_noise = stats.t.rvs(dof, size=n_sims, random_state=rng)

    margin_draws = projected_margin + margin_sigma * (
        np.sqrt(MARGIN_TOTAL_CORR) * shared + np.sqrt(1 - MARGIN_TOTAL_CORR) * margin_noise
    )
    total_draws = projected_total + total_sigma * (
        np.sqrt(MARGIN_TOTAL_CORR) * shared + np.sqrt(1 - MARGIN_TOTAL_CORR) * total_noise
    )
    total_draws = np.clip(total_draws, 0, None)  # scores can't be negative

    home_score = (total_draws + margin_draws) / 2.0
    away_score = (total_draws - margin_draws) / 2.0
    home_score = np.clip(home_score, 0, None)
    away_score = np.clip(away_score, 0, None)

    return {
        "home_score_sims": home_score,
        "away_score_sims": away_score,
        "margin_sims": margin_draws,
        "total_sims": total_draws,
        "median_home_score": float(np.median(home_score)),
        "median_away_score": float(np.median(away_score)),
        "median_margin": float(np.median(margin_draws)),
        "median_total": float(np.median(total_draws)),
    }


def prob_covers_spread(sim: dict, spread_line: float) -> float:
    """
    spread_line: standard book convention, home perspective, e.g. -3.5 means home
    favored by 3.5. Returns probability the HOME side covers that spread, i.e.
    P(margin_sims > -spread_line)... expressed carefully:
        home covers if (home_score - away_score) + spread_line > 0
        i.e. margin_sims > -spread_line
    """
    return float(np.mean(sim["margin_sims"] > -spread_line))


def prob_over_total(sim: dict, total_line: float) -> float:
    return float(np.mean(sim["total_sims"] > total_line))


def edge_pct(model_prob: float, american_odds: int) -> float:
    """
    Converts American odds to implied (vig-inclusive) probability, then computes
    the model's edge over that. This matches FTN's stated methodology: probability
    ignores vig, edge accounts for it.

    american_odds: e.g. -110, +120
    """
    if american_odds < 0:
        implied_prob = -american_odds / (-american_odds + 100)
    else:
        implied_prob = 100 / (american_odds + 100)
    return model_prob - implied_prob


def classify_edge(edge: float) -> str:
    if edge < 0:
        return "no edge"
    elif edge < 0.05:
        return "small edge"
    elif edge < 0.10:
        return "medium edge"
    else:
        return "big edge"


if __name__ == "__main__":
    # Example: home projected -3.2 (model convention) -> flip to +3.2 for simulate_game's
    # "positive = home favored" convention. Projected total 46.5.
    sim = simulate_game(projected_margin=3.2, projected_total=46.5, seed=42)
    print(f"Median score: Home {sim['median_home_score']:.1f} - Away {sim['median_away_score']:.1f}")

    prob = prob_covers_spread(sim, spread_line=-3.5)
    print(f"P(home covers -3.5) = {prob:.3f}")

    edge = edge_pct(prob, american_odds=-110)
    print(f"Edge vs -110: {edge:.3%} ({classify_edge(edge)})")
