# Model Version Changelog

Each version should be backtested with `src/backtest/harness.py` before being
promoted, and results (MAE, RMSE, ATS win rate, CLV) recorded here so later
versions can be compared against a fixed baseline.

## v0 — Elo baseline (scaffold)
- Margin-of-victory-adjusted Elo, home field = 65 elo pts (~2.6 pts)
- Season regression: 33% reversion to mean at season start
- **Status:** not yet backtested — run `src/models/elo_baseline.py` then score
  with `src/backtest/harness.py` against nflverse schedule closing lines.

## v1 — EPA blend (planned)
- Blend Elo rating with rolling EPA/play power ratings (offense & defense split)
- Rolling window: last 8 games, min 4 games before trusting current-season data
- Weighting: 40% Elo / 60% EPA (starting point — recalibrate from backtest)

## v2 — QB-adjusted (planned)
- Layer in starting QB EPA delta vs. backup baseline
- Capped shift: max 7 points from a QB change alone
- Source: SportsDataIO depth chart / player game stats to detect starter changes

## v3 — Situational adjustments (planned)
- Rest day differential, travel distance, short week flags
- Divisional game flag (historically lower variance, sharper markets)
- Consider: weather for outdoor stadiums, primetime scheduling effects

## DVOA standalone track (parallel to v0-v2 above)
- DVOA is proprietary (Football Outsiders/FTN) — no API access, so team is
  manually sourcing historical weekly DVOA tables into `data/raw/dvoa/dvoa_{season}.csv`
- `src/features/dvoa_loader.py` handles the walk-forward lag (critical: FO's
  "DVOA through week W" snapshots include week W's own game — must shift by
  1 week before using as a pre-game feature, or the backtest leaks)
- `src/models/spread_projection.py` converts DVOA (%) to an implied point
  spread via a `DVOA_TO_POINTS` constant — this is an unrefit starting guess
  (0.35 pts per % DVOA edge) and must be recalibrated from backtest residuals
- `src/backtest/dvoa_backtest.py` runs DVOA standalone against the closing
  line and side-by-side against Elo and EPA standalone performance
- **Status:** pipeline built, blocked on sourced DVOA CSVs being placed in
  `data/raw/dvoa/`

## Simulation + live odds layer (FTN-parity phase)
Goal: match FTN's output shape — median projected score, probability, and
edge % vs. vig — without building full player-level drive simulation yet.

- `src/models/monte_carlo.py` — team-level Monte Carlo. Treats the model's
  projected margin/total as the mean of a Student's-t distribution (fatter
  tails than normal, matching real NFL margin variance) and simulates 10,000
  outcomes per game. Outputs median score, P(cover), P(over), and `edge_pct()`
  which nets model probability against vig-implied probability from American
  odds — same edge definition FTN uses (negative = no edge, 0-5% small,
  5-10% medium, 10%+ big, via `classify_edge()`).
  **Sigma is currently a placeholder (13.5 pts margin, 10.0 pts total) — MUST
  be recalibrated via `calibrate_sigma_from_backtest()` using actual backtest
  residuals once v0/v1/v2 backtests are run.**
- `src/ingestion/live_odds.py` — pulls real-time multi-book odds via
  SportsDataIO (uncached — lines move), and surfaces best available number
  per side/book, matching the "which sportsbook offers the most favorable
  odds" feature.
- `src/reports/weekly_model_output.py` — ties projection + simulation + live
  odds together into the actual weekly deliverable: one row per game with
  median score, cover probability, edge %, and best book.
- **Injury handling caveat (matches FTN's own stated limitation):** the
  pipeline assumes listed starters play. Questionable players must be
  manually haircut into `projected_margin`/`projected_total` before running
  weekly output, or the model will look falsely confident on that side.
- **Status:** built, not yet calibrated. Next steps: (1) run v0 Elo backtest
  to get real sigma, (2) test live odds pull against a real week, (3) decide
  whether team-level sim is sharp enough or whether to move to player-level
  simulation (deferred per decision — team-level first, player-level later
  if needed).

## Backtest results log

| Version | Seasons | MAE | RMSE | ATS Win % | Mean CLV | Notes |
|---------|---------|-----|------|-----------|----------|-------|
| Elo (standalone)  | 2021-2025 | 13.50 | 17.40 | 47.98% | +1.09 | run via run_full_backtest.py |
| EPA (standalone)  | 2021-2025 | 12.76 | 16.25 | 52.02% | -1.53 | best of the three so far; not yet stat-sig vs 52.4% breakeven |
| DVOA (standalone) | 2021-2025 | 16.31 | 20.73 | 49.96% | -1.61 | weakest standalone; 1,199 games (wk1 excluded, no leakage) |
| Blended (equal weight) | 2021-2025 | 13.60 | 17.48 | 49.74% | — | UNDERPERFORMED EPA alone — equal-weighting drags down the best signal; next: test weighted blends favoring EPA |
| Blended by week: Wk1-4 | 2021-2025 | 12.81 | 16.73 | 48.12% | — | n=320. Early season NOT meaningfully worse — see note below |
| Blended by week: Wk5+ | 2021-2025 | 13.84 | 17.70 | 50.24% | — | n=1039 |
| Weight sweep (train=2021-23, holdout=2024-25) | 2021-2025 | — | — | Train: 52.76% (30% elo/70% epa) | — | Best train weights did NOT generalize to holdout (49.63% vs EPA-alone's 50.18% on same holdout) — overfitting signature. No blend yet beats EPA alone out-of-sample. |

**Note on early vs. late season:** Elo and EPA both carry rating signal across season boundaries (Elo regresses toward mean but doesn't reset; EPA's rolling window pulls trailing games regardless of season). Only DVOA fully cold-starts each season (Week 1 = null). So contrary to intuition, weeks 1-4 were not meaningfully weaker in this backtest. Per-week resolution (Week 1-18) is too noisy (n=70-80/week) to draw week-specific conclusions.

**Note on the weight sweep (important):** with ~800 train games and ATS rates clustered in a 48-54% band, the gap between weight combinations is close to the noise floor. The train-optimal weighting did not hold up on the holdout — treat this as evidence that no tested blend yet has a real, stable edge over EPA alone, not as a reason to keep searching for better weights on this same data.

## QB adjustment layer (built from pbp passer data, no paid feed needed)
- Built passer-level trailing EPA/dropback ratings with shrinkage toward a replacement-level prior (fixed a real bug where thin-sample backups fell back to league-average instead of below-average, caught via a Nate Mullens 2023 sanity check) and forward-fill across weeks a passer didn't play (fixed a second bug where an injured starter's own rating incorrectly reset to replacement level on the week he didn't play)
- **Result: did NOT help.** Underperformed EPA-alone overall, on the QB-change subset, and even isolating to the very first game of a change (ruling out the "EPA rolling window already adapted" hypothesis). Leading explanation: backup QB starts are heavily public information: the market likely already prices these efficiently, leaving little edge for a simple trailing-performance-based adjustment.
- Layer built but not adopted as a model input.

## Turnover luck adjustment layer (fumble recovery regression)
- Confirmed empirically in this dataset: teams recover their own fumbles 52.1% of the time (matches the well-established ~50/50 finding). Built a walk-forward, shrinkage-adjusted "luck index" per team (actual vs. expected self-recoveries), converted to a capped point adjustment.
- **Result: modest, real but not fully proven edge.** EPA+turnover ATS = 53.20% overall (n=1,359) vs. EPA-alone's 52.02%. Mixed across time splits (train 53.74%→53.74%, i.e. slightly down; holdout 48.90%→52.39%, i.e. up) — since no constants were tuned to this data, this reflects genuine season-to-season variance rather than overfitting, but the sample is not large enough to fully separate real edge from noise (53.20% is well within 1 SE of the 52.4% breakeven line).
- **Adopted as a core model input alongside EPA**, per decision to move forward with EPA + turnover luck as main drivers.

## Rest adjustment layer (built from schedule data already fetched — home_rest/away_rest)
- Simple linear rest-differential adjustment (0.20 pts/day, capped at ±3 pts).
- **Result: small but consistently positive** in every split tested — all games (53.20%→53.57%), train (53.74%→53.87%), holdout (52.39%→53.12%), and the meaningful-mismatch subset, |rest_diff|>=3 days, n=291 (52.23%→53.61%). Same direction everywhere, unlike turnover luck's mixed result — better evidence of a real (if modest) effect.
- **Adopted as a core model input** alongside EPA + turnover luck.

## Wind diagnostic (not adopted as a spread adjustment)
- Tested whether high wind (>=15mph) correlates with core-spread prediction error. t-test: p=0.425 -- no statistically meaningful spread bias.
- **Conclusion: wind's known effect is on total scoring, not margin between two teams — confirmed empirically here.** Not built as a spread adjustment; could be revisited if a totals model is built later.

## Current model (as of this point): EPA + turnover luck + rest
- Combined ATS: ~53.5-53.6% across full 5-season sample -- best validated combination so far, though still short of statistical certainty and well short of the >55% aspirational target discussed.

