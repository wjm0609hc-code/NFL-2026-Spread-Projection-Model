"""
Shrinks the raw model spread toward the market line by a factor k before it's
used for anything magnitude-sensitive (cover probability, edge %). This does
NOT make the model a more accurate estimate of the actual game margin -- see
the verification below -- it exists because the raw model (EPA + turnover-luck
+ rest) throws off point-spread magnitudes (e.g. a 19.6-point favorite) far
outside what its own backtested edge supports.

Verified against src/backtest/rest_weather_backtest.py's scored dataset
(2021-2025, n=1359, core_plus_rest_spread vs. spread_line_standard) before
this was wired in:

  k              1.0 (raw)   0.3       0.0342
  ATS win rate   0.535688    0.535688  0.535688   <- IDENTICAL, all decimals
  MAE            13.116      13.568    14.148     <- gets WORSE as k shrinks
  RMSE           16.719      17.371    18.081     <- gets WORSE as k shrinks

Why ATS is exactly invariant to k (for any k > 0): src/backtest/harness.py's
score_model() decides a model's ATS side purely from
sign(closing_spread - model_spread). Algebraically,
calibrated_spread = (1-k)*market_spread + k*model_spread, so
closing_spread - calibrated_spread = k*(closing_spread - model_spread) --
scaling by a positive k never flips that sign. So shrinking toward market is
"free" with respect to this repo's ATS metric, at any k. It is NOT free with
respect to magnitude: MAE/RMSE against the actual final margin get worse, not
better, as k shrinks, because the market line itself carries ~13-14 pts of
residual error against actual outcomes -- tracking it more closely doesn't
make the projection track real game margins any better.

Practical consequence (by design, not a bug): once a spread calibrated at a
small k is fed into monte_carlo.prob_covers_spread-style cover probability
against that SAME market line, the result sits close to 50% / ~no edge for
almost every game, because the calibrated spread is, by construction, mostly
the market line. That matches this repo's own backtested read that the real
edge here is thin (~53.6% ATS, well short of the >55% aspirational target in
docs/CHANGELOG.md) -- near-zero edge on most games is the honest output, not
a symptom of over-shrinking. If a game's direction is worth surfacing on its
own, use the RAW model spread's side (see generate_week_card.py's directional
lean column) rather than reading direction off the calibrated magnitude.
"""
from __future__ import annotations


def calibrate_final_spread(model_spread: float, market_spread_standard: float, k: float = 0.0342) -> float:
    """
    model_spread, market_spread_standard, and the return value all use the
    standard convention shared by spread_projection.py: negative = home
    favored.

    k is the fraction of the model's deviation from market that survives
    calibration (k=1.0 -> raw model spread, k=0.0 -> market spread exactly).
    """
    market_margin = -market_spread_standard
    model_margin = -model_spread
    calibrated_margin = market_margin + k * (model_margin - market_margin)
    return -calibrated_margin
