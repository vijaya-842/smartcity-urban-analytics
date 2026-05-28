"""
PySpark session factory with tuned configuration for local and cluster modes.
"""
from __future__ import annotations
from pyspark.sql import SparkSession


def create_spark_session(app_name: str = "SmartCity-Urban-Analytics",
                         mode: str = "local") -> SparkSession:
    """
    Build and return a configured SparkSession.

    Args:
        app_name: Logical name shown in Spark UI.
        mode:     "local" for development, "cluster" for production.
    """
    builder = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    )

    if mode == "local":
        builder = builder.master("local[*]")
    elif mode == "cluster":
        # In cluster mode, master is set by spark-submit --master
        builder = (
            builder
            .config("spark.executor.instances", "4")
            .config("spark.executor.cores", "4")
        )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark
