"""
SmartCity Urban Analytics — Main pipeline entry point.

Orchestrates: data loading -> cleaning -> analysis -> visualisation

Usage:
    python main.py                        # full pipeline (all datasets)
    python main.py --dataset 311          # single dataset
    python main.py --mode cluster         # Spark cluster mode
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger(__name__)

# NYC borough populations (2020 Census)
BOROUGH_POP = {
    "Manhattan":     1_694_251,
    "Brooklyn":      2_736_074,
    "Queens":        2_405_464,
    "Bronx":         1_472_654,
    "Staten Island":   495_747,
}


def run_pipeline(datasets: list[str], mode: str = "local") -> None:
    from config.spark_config import create_spark_session
    from src.data_loader   import NYCDataLoader
    from src.cleaning      import NYCDataCleaner
    from src.analysis      import UrbanAnalytics
    from src.visualizations import (
        plot_complaints_by_borough, plot_monthly_trend,
        plot_hourly_heatmap, plot_collision_severity,
        plot_traffic_by_hour,
    )

    spark     = create_spark_session(mode=mode)
    loader    = NYCDataLoader(spark)
    cleaner   = NYCDataCleaner()
    analytics = UrbanAnalytics(spark)

    Path("output/charts").mkdir(parents=True, exist_ok=True)

    if "311" in datasets:
        log.info("Processing 311 Service Requests...")
        raw311   = loader.load("311_requests")
        clean311 = cleaner.clean_311(raw311)

        borough_vol  = analytics.complaint_volume_by_borough(clean311)
        monthly      = analytics.monthly_complaint_trend(clean311)
        hourly_hw    = analytics.hourly_complaint_heatmap(clean311)

        plot_complaints_by_borough(borough_vol.toPandas())
        plot_monthly_trend(monthly.toPandas(), metric="count",
                           title="Monthly 311 Complaint Volume")
        plot_hourly_heatmap(hourly_hw.toPandas())

        log.info("Top complaint types:")
        analytics.top_complaint_types(clean311).show(10)

    if "traffic" in datasets:
        log.info("Processing Traffic Volume...")
        raw_tr  = loader.load("traffic_volume")
        clean_tr = cleaner.clean_traffic(raw_tr)

        peak = analytics.peak_traffic_hours(clean_tr)
        plot_traffic_by_hour(peak.toPandas())
        analytics.avg_traffic_by_borough_hour(clean_tr).show(10)

    if "collisions" in datasets:
        log.info("Processing NYPD Collisions...")
        raw_col   = loader.load("collisions")
        clean_col = cleaner.clean_collisions(raw_col)

        severity  = analytics.collision_severity_by_borough(clean_col)
        safety    = analytics.borough_safety_index(clean_col, BOROUGH_POP)
        plot_collision_severity(severity.toPandas())
        safety.show()
        analytics.hotspot_corridors(clean_col).show(10)

    spark.stop()
    log.info("Pipeline complete. Charts saved to output/charts/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+",
                        default=["311", "traffic", "collisions"],
                        choices=["311", "traffic", "collisions"])
    parser.add_argument("--mode", default="local", choices=["local", "cluster"])
    args = parser.parse_args()
    run_pipeline(args.dataset, args.mode)
