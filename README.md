# NFL Proprietary Spread Model

A from-scratch, backtested point-spread projection system for NFL games.

## Architecture

```
data/
  raw/            # untouched pulls from SportsDataIO + nflverse (parquet/csv, partitioned by season)
  processed/       # cleaned, feature-engineered team-week datasets
  cache/           # API response cache to avoid re-hitting rate limits

src/
  ingestion/       # data pull scripts (SportsDataIO client, nflverse loader, odds history)
  features/        # feature engineering: EPA rollups, power ratings, QB adjustments, rest/travel
  models/          # the rating engine(s): Elo baseline, EPA blend, final spread generator
  backtest/        # walk-forward backtest harness, ATS scoring, CLV tracking

config/            # league constants, team name mappings, season definitions, API keys (gitignored)
notebooks/         # exploratory analysis, not part of the pipeline
tests/             # unit tests for feature calculations and backtest logic
docs/              # methodology notes, changelog of model versions
```

## Data sources

- **SportsDataIO** (paid): injuries, real-time odds/lines, QB depth chart status, live game data
- **nflverse / nflfastR** (free): play-by-play back to 1999 — used as the backtest backbone since it's
  the most reliable source for EPA, success rate, and situational splits over a 5+ season window

## Design principles

1. **Walk-forward only.** No feature for a given game may use information not available before that
   game's kickoff. This is enforced in `src/features/build_features.py` via an `as_of_date` cutoff.
2. **Test against the closing line, not just final score.** The real benchmark is whether the model's
   projected spread beats the market close (CLV), not whether it "predicts" the winner.
3. **Versioned models.** Each model iteration gets a version tag (`v0_elo_baseline`, `v1_epa_blend`,
   `v2_qb_adjusted`, etc.) so backtest results are always comparable across versions.
4. **Cache aggressively, mutate rarely.** Raw pulls are immutable once written. Reprocessing happens
   in `processed/`, never by overwriting `raw/`.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp config/config.example.yaml config/config.yaml   # add your SportsDataIO API key
```

## Current status

Scaffold stage. See `docs/CHANGELOG.md` for build progress.
