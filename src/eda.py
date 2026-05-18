from __future__ import annotations

import base64
from io import BytesIO

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def _encode_current_figure() -> str:
    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close()
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


def compute_summary_statistics(load_df: pd.DataFrame) -> dict[str, float | int]:
    series = load_df["load_mw"].dropna()
    return {
        "count": int(series.count()),
        "mean": round(float(series.mean()), 3),
        "median": round(float(series.median()), 3),
        "min": round(float(series.min()), 3),
        "max": round(float(series.max()), 3),
        "std": round(float(series.std()), 3),
        "q25": round(float(series.quantile(0.25)), 3),
        "q75": round(float(series.quantile(0.75)), 3),
        "missing_values": int(load_df["load_mw"].isna().sum()),
    }


def weekly_seasonality(load_df: pd.DataFrame) -> list[dict[str, float | int | str]]:
    working = load_df.copy()
    working["day_of_week"] = working.index.day_name()
    working["hour"] = working.index.hour
    grouped = (
        working.groupby(["day_of_week", "hour"], observed=True)["load_mw"]
        .mean()
        .reset_index()
        .rename(columns={"load_mw": "average_load_mw"})
    )
    grouped["average_load_mw"] = grouped["average_load_mw"].round(3)
    return grouped.to_dict(orient="records")


def load_curve_plot(load_df: pd.DataFrame) -> str:
    plt.figure(figsize=(11, 4.5))
    plt.plot(load_df.index, load_df["load_mw"], color="#2563eb", linewidth=1.3)
    plt.title("Hourly Electricity Load")
    plt.xlabel("Timestamp")
    plt.ylabel("Load (MW)")
    plt.grid(alpha=0.25)
    return _encode_current_figure()


def daily_boxplot(load_df: pd.DataFrame) -> str:
    working = load_df.copy()
    working["day"] = working.index.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    data = [working.loc[working["day"] == day, "load_mw"].dropna() for day in order]

    plt.figure(figsize=(10, 4.8))
    plt.boxplot(data, tick_labels=order, showfliers=False)
    plt.title("Daily Load Distribution")
    plt.xlabel("Day of week")
    plt.ylabel("Load (MW)")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.25)
    return _encode_current_figure()


def weekly_seasonality_plot(load_df: pd.DataFrame) -> str:
    working = load_df.copy()
    working["day_order"] = working.index.dayofweek
    working["hour"] = working.index.hour
    pivot = working.pivot_table(index="day_order", columns="hour", values="load_mw", aggfunc="mean")

    plt.figure(figsize=(11, 4.8))
    plt.imshow(pivot, aspect="auto", cmap="viridis")
    plt.colorbar(label="Average load (MW)")
    plt.title("Weekly Seasonality Heatmap")
    plt.xlabel("Hour of day")
    plt.ylabel("Day of week")
    plt.yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    return _encode_current_figure()


def build_eda_payload(load_df: pd.DataFrame) -> dict[str, object]:
    return {
        "summary_statistics": compute_summary_statistics(load_df),
        "weekly_seasonality": weekly_seasonality(load_df),
    }


def build_plot_payload(load_df: pd.DataFrame) -> dict[str, str]:
    return {
        "load_curve": load_curve_plot(load_df),
        "weekly_seasonality": weekly_seasonality_plot(load_df),
        "daily_boxplot": daily_boxplot(load_df),
    }
