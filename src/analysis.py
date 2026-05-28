"""
Distributed analytics on cleaned NYC datasets using PySpark.
Computes borough-level KPIs, trend analysis, and hotspot detection.
"""
from __future__ import annotations

import logging
from pyspark.sql import DataFrame, SparkSession, Window
import pyspark.sql.functions as F

log = logging.getLogger(__name__)


class UrbanAnalytics:
    """High-level analytics built on top of cleaned Spark DataFrames."""

    def __init__(self, spark: SparkSession) -> None:
        self.spark = spark

    # ── 311 Analysis ─────────────────────────────────────────────────────────

    def complaint_volume_by_borough(self, df: DataFrame) -> DataFrame:
        """Total and ranked complaint counts per borough."""
        w = Window.orderBy(F.desc("total_complaints"))
        return (df.groupBy("borough")
                  .agg(F.count("*").alias("total_complaints"))
                  .withColumn("rank", F.rank().over(w))
                  .orderBy("rank"))

    def top_complaint_types(self, df: DataFrame, n: int = 15) -> DataFrame:
        return (df.groupBy("complaint_type")
                  .count()
                  .orderBy(F.desc("count"))
                  .limit(n))

    def avg_resolution_time_by_agency(self, df: DataFrame) -> DataFrame:
        """Mean resolution hours grouped by agency and borough."""
        return (df.filter(F.col("resolution_hours").between(0, 720))
                  .groupBy("agency", "borough")
                  .agg(F.round(F.avg("resolution_hours"), 2).alias("avg_hours"),
                       F.count("*").alias("total_cases"))
                  .orderBy(F.desc("total_cases")))

    def monthly_complaint_trend(self, df: DataFrame) -> DataFrame:
        return (df.groupBy("year", "month")
                  .count()
                  .orderBy("year", "month"))

    def hourly_complaint_heatmap(self, df: DataFrame) -> DataFrame:
        """Matrix of complaints by hour × day_of_week for heatmap plotting."""
        return (df.groupBy("day_of_week", "hour")
                  .count()
                  .orderBy("day_of_week", "hour"))

    # ── Traffic Analysis ─────────────────────────────────────────────────────

    def avg_traffic_by_borough_hour(self, df: DataFrame) -> DataFrame:
        return (df.groupBy("borough", "hour")
                  .agg(F.round(F.avg("volume"), 1).alias("avg_volume"),
                       F.sum("volume").alias("total_volume"))
                  .orderBy("borough", "hour"))

    def yearly_traffic_trend(self, df: DataFrame) -> DataFrame:
        return (df.groupBy("year", "borough")
                  .agg(F.sum("volume").alias("total_volume"))
                  .orderBy("year", "borough"))

    def peak_traffic_hours(self, df: DataFrame) -> DataFrame:
        return (df.groupBy("hour")
                  .agg(F.sum("volume").alias("total_volume"))
                  .orderBy(F.desc("total_volume")))

    # ── Collision Analysis ────────────────────────────────────────────────────

    def collision_severity_by_borough(self, df: DataFrame) -> DataFrame:
        return (df.groupBy("borough")
                  .agg(F.count("*").alias("total_collisions"),
                       F.sum("persons_injured").alias("total_injured"),
                       F.sum("persons_killed").alias("total_killed"),
                       F.round(F.avg("persons_injured"), 3).alias("avg_injured_per_collision"))
                  .orderBy(F.desc("total_collisions")))

    def monthly_collision_trend(self, df: DataFrame) -> DataFrame:
        return (df.groupBy("year", "month")
                  .agg(F.count("*").alias("collisions"),
                       F.sum("persons_injured").alias("injured"),
                       F.sum("persons_killed").alias("killed"))
                  .orderBy("year", "month"))

    def hotspot_corridors(self, df: DataFrame,
                          min_collisions: int = 50) -> DataFrame:
        """Street corridors with high collision frequency."""
        if "on_street_name" not in df.columns:
            return self.spark.createDataFrame([], schema="street STRING, collisions LONG")
        return (df.filter(F.col("on_street_name").isNotNull())
                  .groupBy("borough", "on_street_name")
                  .count()
                  .filter(F.col("count") >= min_collisions)
                  .orderBy(F.desc("count"))
                  .withColumnRenamed("count", "collisions")
                  .limit(20))

    # ── Cross-dataset insights ────────────────────────────────────────────────

    def borough_safety_index(self, collision_df: DataFrame,
                             population: dict) -> DataFrame:
        """
        Normalised safety index per borough (collisions per 100k residents).
        Higher score = more collisions relative to population.
        """
        pop_rows = [(k, v) for k, v in population.items()]
        pop_df   = self.spark.createDataFrame(pop_rows, ["borough", "population"])

        collisions_per_borough = (
            collision_df.groupBy("borough").count()
                        .withColumnRenamed("count", "total_collisions")
        )
        return (collisions_per_borough
                .join(pop_df, on="borough", how="inner")
                .withColumn("collisions_per_100k",
                            F.round(F.col("total_collisions") /
                                    F.col("population") * 100_000, 2))
                .orderBy(F.desc("collisions_per_100k")))
