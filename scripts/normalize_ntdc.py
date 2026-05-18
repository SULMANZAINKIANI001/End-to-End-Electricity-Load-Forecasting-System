from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def normalize_ntdc(input_path: Path, output_path: Path) -> None:
    raw = pd.read_csv(input_path)
    required_columns = {"DATE", "HOUR", "SYSLOAD"}
    missing_columns = required_columns.difference(raw.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required NTDC columns: {missing}")

    dates = pd.to_datetime(raw["DATE"], dayfirst=True, errors="coerce")
    hours = pd.to_numeric(raw["HOUR"], errors="coerce")
    load_mw = pd.to_numeric(raw["SYSLOAD"].astype(str).str.replace(",", "", regex=False), errors="coerce")

    normalized = pd.DataFrame(
        {
            "timestamp": dates + pd.to_timedelta(hours - 1, unit="h"),
            "load_mw": load_mw,
        }
    )
    normalized = normalized.dropna(subset=["timestamp", "load_mw"])
    normalized = normalized.sort_values("timestamp")
    normalized = normalized.drop_duplicates(subset=["timestamp"], keep="last")

    if normalized.empty:
        raise ValueError("No valid rows were produced from the NTDC dataset.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output_path, index=False, date_format="%Y-%m-%d %H:%M:%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize the Kaggle NTDC Pakistan load dataset.")
    parser.add_argument("--input", type=Path, default=Path("data/raw/ntdc/NTDC_2015_2020.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/pakistan_load.csv"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    normalize_ntdc(args.input, args.output)
