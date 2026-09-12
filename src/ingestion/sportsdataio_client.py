"""
Thin client for SportsDataIO NFL v3 API.

Handles auth, rate-limit-friendly caching, and writes raw JSON responses
to data/raw/sportsdataio/ untouched. All parsing/cleaning happens downstream
in src/features/, never here.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests
import yaml


class SportsDataIOClient:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.api_key: str = cfg["sportsdataio"]["api_key"]
        self.base_url: str = cfg["sportsdataio"]["base_url"].rstrip("/")
        self.cache_dir = Path(cfg["paths"]["cache_dir"]) / "sportsdataio"
        self.raw_dir = Path(cfg["paths"]["raw_dir"]) / "sportsdataio"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()

    def _get(self, endpoint: str, use_cache: bool = True) -> Any:
        """
        endpoint like '/scores/json/ScoresByWeek/2024/5'
        Caches raw response so re-running a backtest doesn't re-hit the API.
        """
        cache_key = endpoint.strip("/").replace("/", "_") + ".json"
        cache_path = self.cache_dir / cache_key

        if use_cache and cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)

        url = f"{self.base_url}{endpoint}"
        resp = self._session.get(url, headers={"Ocp-Apim-Subscription-Key": self.api_key})
        resp.raise_for_status()
        data = resp.json()

        with open(cache_path, "w") as f:
            json.dump(data, f)

        # be polite to rate limits
        time.sleep(0.2)
        return data

    def get_scores_by_week(self, season: int, week: int) -> list[dict]:
        return self._get(f"/scores/json/ScoresByWeek/{season}/{week}")

    def get_odds_by_week(self, season: int, week: int) -> list[dict]:
        return self._get(f"/odds/json/GameOddsByWeek/{season}/{week}")

    def get_injuries(self, season: int) -> list[dict]:
        return self._get(f"/scores/json/Injuries/{season}", use_cache=False)  # injuries change daily

    def get_player_game_stats_by_week(self, season: int, week: int) -> list[dict]:
        return self._get(f"/scores/json/PlayerGameStatsByWeek/{season}/{week}")

    def dump_raw_season(self, season: int, weeks: range = range(1, 19)) -> None:
        """Pull and persist a full season of scores + odds to data/raw/ as parquet."""
        import pandas as pd

        all_scores, all_odds = [], []
        for wk in weeks:
            all_scores.extend(self.get_scores_by_week(season, wk))
            all_odds.extend(self.get_odds_by_week(season, wk))

        pd.DataFrame(all_scores).to_parquet(self.raw_dir / f"scores_{season}.parquet")
        pd.DataFrame(all_odds).to_parquet(self.raw_dir / f"odds_{season}.parquet")
        print(f"Wrote {len(all_scores)} game records and {len(all_odds)} odds records for {season}")


if __name__ == "__main__":
    client = SportsDataIOClient()
    # Example: pull current season scaffold test
    # client.dump_raw_season(2025)
