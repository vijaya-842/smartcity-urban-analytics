"""
NYC Open Data loader — downloads datasets via the Socrata API
and converts them to Spark DataFrames.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import requests
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType

log = logging.getLogger(__name__)

DATASETS = {
    "311_requests": {
        "endpoint": "https://data.cityofnewyork.us/resource/erm2-nwe9.json",
        "description": "NYC 311 Service Requests",
        "limit": 500_000,
    },
    "traffic_volume": {
        "endpoint": "https://data.cityofnewyork.us/resource/btm5-ppia.json",
        "description": "NYC Traffic Volume Counts",
        "limit": 200_000,
    },
    "collisions": {
        "endpoint": "https://data.cityofnewyork.us/resource/h9gi-nx95.json",
        "description": "NYPD Motor Vehicle Collisions",
        "limit": 300_000,
    },
}

APP_TOKEN = os.getenv("NYC_APP_TOKEN", "")   # Optional — increases rate limit


class NYCDataLoader:
    """Downloads NYC Open Data and caches as Parquet for fast re-reads."""

    def __init__(self, spark: SparkSession,
                 cache_dir: str = "data/raw") -> None:
        self.spark     = spark
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session  = requests.Session()
        if APP_TOKEN:
            self._session.headers["X-App-Token"] = APP_TOKEN

    def _fetch_json(self, endpoint: str, limit: int,
                    offset: int = 0) -> list[dict]:
        """Paginated fetch from Socrata endpoint."""
        params = {"$limit": min(limit, 50_000), "$offset": offset,
                  "$order": ":id"}
        resp = self._session.get(endpoint, params=params, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def _download_dataset(self, name: str, meta: dict) -> Path:
        """Download full dataset with pagination, save as JSONL."""
        out_path = self.cache_dir / f"{name}.jsonl"
        if out_path.exists():
            log.info("Cache hit: %s", out_path)
            return out_path

        log.info("Downloading %s (up to %s records)...",
                 meta["description"], f"{meta['limit']:,}")
        total, offset = 0, 0
        with open(out_path, "w") as f:
            while total < meta["limit"]:
                batch = self._fetch_json(meta["endpoint"], meta["limit"], offset)
                if not batch:
                    break
                for rec in batch:
                    f.write(str(rec).replace("'", '"') + "
")
                total  += len(batch)
                offset += len(batch)
                log.info("  Downloaded %s records", f"{total:,}")
                if len(batch) < 50_000:
                    break
        log.info("Saved %s records -> %s", f"{total:,}", out_path)
        return out_path

    def load(self, dataset: str,
             schema: Optional[StructType] = None) -> DataFrame:
        """Return a Spark DataFrame for the given NYC dataset name."""
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset '{dataset}'. "
                             f"Available: {list(DATASETS.keys())}")
        meta      = DATASETS[dataset]
        parquet_p = self.cache_dir / f"{dataset}.parquet"

        if parquet_p.exists():
            log.info("Loading from parquet cache: %s", parquet_p)
            return self.spark.read.parquet(str(parquet_p))

        json_p = self._download_dataset(dataset, meta)
        reader = self.spark.read.option("multiLine", True)
        if schema:
            reader = reader.schema(schema)
        else:
            reader = reader.option("inferSchema", True)

        df = reader.json(str(json_p))
        df.write.parquet(str(parquet_p))
        log.info("Cached %d rows as parquet", df.count())
        return df
