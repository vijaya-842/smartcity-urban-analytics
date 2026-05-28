"""
Borough-level and time-series visualizations for SmartCity analytics.
Converts Spark DataFrames to Pandas for plotting.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

OUTPUT = Path("output/charts")
OUTPUT.mkdir(parents=True, exist_ok=True)

BOROUGH_COLORS = {
    "Manhattan":     "#264653",
    "Brooklyn":      "#2A9D8F",
    "Queens":        "#E9C46A",
    "Bronx":         "#F4A261",
    "Staten Island": "#E76F51",
}
BOROUGHS = list(BOROUGH_COLORS.keys())


def _save(fig, name):
    p = OUTPUT / f"{name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {p}")


def plot_complaints_by_borough(pdf: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors  = [BOROUGH_COLORS.get(b, "#999") for b in pdf["borough"]]
    bars    = ax.barh(pdf["borough"], pdf["total_complaints"],
                      color=colors, edgecolor="white")
    ax.bar_label(bars, labels=[f"{v:,}" for v in pdf["total_complaints"]],
                 padding=4, fontsize=9)
    ax.set_title("311 Complaint Volume by Borough", fontsize=14,
                 fontweight="bold", pad=15)
    ax.set_xlabel("Total Complaints")
    ax.invert_yaxis()
    _save(fig, "complaints_by_borough")


def plot_monthly_trend(pdf: pd.DataFrame, metric: str = "count",
                       title: str = "Monthly Trend") -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(range(len(pdf)), pdf[metric], marker="o", linewidth=2.5,
            color="#2A9D8F", markersize=6)
    ax.fill_between(range(len(pdf)), pdf[metric], alpha=0.15, color="#2A9D8F")
    labels = pdf.apply(lambda r: f"{int(r['year'])}-{int(r['month']):02d}", axis=1)
    ax.set_xticks(range(len(pdf)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    _save(fig, f"trend_{metric}")


def plot_hourly_heatmap(pdf: pd.DataFrame) -> None:
    """Day-of-week x Hour heatmap for complaint intensity."""
    pivot = pdf.pivot(index="day_of_week", columns="hour", values="count").fillna(0)
    days  = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.imshow(pivot.values, aspect="auto",
                   cmap="YlOrRd", interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Complaints")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([f"{h:02d}:00" for h in pivot.columns], rotation=45, fontsize=8)
    ax.set_yticks(range(len(days)))
    ax.set_yticklabels(days)
    ax.set_title("311 Complaint Heatmap — Day of Week vs Hour",
                 fontsize=14, fontweight="bold", pad=15)
    _save(fig, "complaint_heatmap")


def plot_collision_severity(pdf: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(pdf))
    for ax, col, label, color in [
        (axes[0], "total_collisions", "Total Collisions", "#264653"),
        (axes[1], "total_injured",    "People Injured",   "#E76F51"),
    ]:
        bars = ax.bar(pdf["borough"], pdf[col], color=color, edgecolor="white")
        ax.bar_label(bars, labels=[f"{v:,}" for v in pdf[col]],
                     padding=4, fontsize=9)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Borough")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.suptitle("Motor Vehicle Collision Analysis by Borough",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "collision_severity")


def plot_traffic_by_hour(pdf: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(pdf["hour"], pdf["total_volume"],
           color="#2A9D8F", edgecolor="white")
    ax.set_title("NYC Traffic Volume by Hour of Day",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Hour"); ax.set_ylabel("Total Volume")
    ax.set_xticks(pdf["hour"])
    ax.set_xticklabels([f"{h:02d}:00" for h in pdf["hour"]], rotation=45)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    _save(fig, "traffic_by_hour")
