"""
PySpark distributed data cleaning pipeline for NYC datasets.
Handles nulls, type casting, deduplication, and borough normalisation.
"""
from __future__ import annotations

import logging
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType, IntegerType, TimestampType

log = logging.getLogger(__name__)

# Standard borough name mapping
BOROUGH_MAP = {
    "MANHATTAN":     "Manhattan",
    "BROOKLYN":      "Brooklyn",
    "QUEENS":        "Queens",
    "BRONX":         "Bronx",
    "STATEN ISLAND": "Staten Island",
    "STATEN IS":     "Staten Island",
}


def _log_quality(df: DataFrame, stage: str) -> None:
    count = df.count()
    log.info("[%s] Rows: %s", stage, f"{count:,}")


class NYCDataCleaner:
    """Stateless cleaning transformations — each method returns a new DataFrame."""

    # ── 311 Service Requests ─────────────────────────────────────────────────

    @staticmethod
    def clean_311(df: DataFrame) -> DataFrame:
        """Clean and type-cast 311 service request data."""
        _log_quality(df, "311 raw")

        # Normalise column names
        df = df.toDF(*[c.lower().replace(" ", "_") for c in df.columns])

        # Select relevant columns
        cols = ["unique_key", "created_date", "closed_date", "agency",
                "complaint_type", "descriptor", "status", "borough",
                "latitude", "longitude", "city", "zip_code"]
        available = [c for c in cols if c in df.columns]
        df = df.select(available)

        # Type casts
        df = (df
              .withColumn("created_date", F.to_timestamp("created_date"))
              .withColumn("closed_date",  F.to_timestamp("closed_date"))
              .withColumn("latitude",     F.col("latitude").cast(DoubleType()))
              .withColumn("longitude",    F.col("longitude").cast(DoubleType())))

        # Remove rows with no borough or valid date
        df = df.dropna(subset=["borough", "created_date"])

        # Normalise borough names
        borough_expr = F.create_map(
            *[x for pair in
              [(F.lit(k), F.lit(v)) for k, v in BOROUGH_MAP.items()]
              for x in pair]
        )
        df = df.withColumn(
            "borough",
            F.coalesce(borough_expr[F.upper(F.col("borough"))],
                       F.initcap(F.col("borough")))
        )

        # Derived features
        df = (df
              .withColumn("year",         F.year("created_date"))
              .withColumn("month",        F.month("created_date"))
              .withColumn("day_of_week",  F.dayofweek("created_date"))
              .withColumn("hour",         F.hour("created_date"))
              .withColumn("resolution_hours",
                          (F.unix_timestamp("closed_date") -
                           F.unix_timestamp("created_date")) / 3600))

        # Remove duplicates
        df = df.dropDuplicates(["unique_key"])
        _log_quality(df, "311 cleaned")
        return df

    # ── Traffic Volume ────────────────────────────────────────────────────────

    @staticmethod
    def clean_traffic(df: DataFrame) -> DataFrame:
        """Clean NYC traffic volume dataset."""
        _log_quality(df, "traffic raw")

        df = df.toDF(*[c.lower().replace(" ", "_") for c in df.columns])

        # Key columns
        keep = ["id", "yr", "m", "d", "hh", "mm", "vol", "boro",
                "direction", "fromst", "tost"]
        df = df.select([c for c in keep if c in df.columns])

        # Numeric casts and filter
        df = (df
              .withColumn("volume",    F.col("vol").cast(IntegerType()))
              .withColumn("year",      F.col("yr").cast(IntegerType()))
              .withColumn("month",     F.col("m").cast(IntegerType()))
              .withColumn("day",       F.col("d").cast(IntegerType()))
              .withColumn("hour",      F.col("hh").cast(IntegerType()))
              .filter(F.col("volume") >= 0)
              .filter(F.col("year").isNotNull()))

        df = df.dropna(subset=["volume", "year"])
        df = df.dropDuplicates()

        # Borough normalise
        df = df.withColumn("borough",
                           F.initcap(F.trim(F.col("boro"))))

        _log_quality(df, "traffic cleaned")
        return df

    # ── Collisions ────────────────────────────────────────────────────────────

    @staticmethod
    def clean_collisions(df: DataFrame) -> DataFrame:
        """Clean NYPD motor-vehicle collision data."""
        _log_quality(df, "collisions raw")

        df = df.toDF(*[c.lower().replace(" ", "_") for c in df.columns])

        df = (df
              .withColumn("crash_date",  F.to_date("crash_date"))
              .withColumn("latitude",    F.col("latitude").cast(DoubleType()))
              .withColumn("longitude",   F.col("longitude").cast(DoubleType()))
              .withColumn("persons_injured",
                          F.col("number_of_persons_injured").cast(IntegerType()))
              .withColumn("persons_killed",
                          F.col("number_of_persons_killed").cast(IntegerType()))
              .dropna(subset=["crash_date", "borough"])
              .filter(F.upper(F.col("borough")).isin(list(BOROUGH_MAP.keys())))
              .withColumn("borough", F.initcap(F.trim(F.col("borough"))))
              .withColumn("year",  F.year("crash_date"))
              .withColumn("month", F.month("crash_date"))
              .dropDuplicates(["collision_id"] if "collision_id" in df.columns else []))

        _log_quality(df, "collisions cleaned")
        return df
