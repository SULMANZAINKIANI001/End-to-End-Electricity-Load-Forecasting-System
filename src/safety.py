from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd


def finite_float(value: Any, default: float | None = None, *, minimum: float | None = None) -> float | None:
    """Return a JSON-safe finite float, or a default when the value is invalid."""

    try:
        if value is None or pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(numeric):
        return default
    if minimum is not None:
        numeric = max(numeric, minimum)
    return numeric


def sanitize_for_json(value: Any) -> Any:
    """Recursively convert pandas/numpy/non-finite values into strict JSON values."""

    if value is None:
        return None
    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, (pd.Timestamp, datetime, date)):
        if pd.isna(value):
            return None
        return value.isoformat()
    if isinstance(value, np.generic):
        return sanitize_for_json(value.item())
    if isinstance(value, np.ndarray):
        return [sanitize_for_json(item) for item in value.tolist()]
    if isinstance(value, Decimal):
        return finite_float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, int):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, Mapping):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [sanitize_for_json(item) for item in value]
    return value


def clean_nonnegative_series(values: Any, *, max_invalid_ratio: float = 0.2) -> tuple[np.ndarray, dict[str, Any]]:
    """Interpolate bounded numeric history and reject severely damaged inputs."""

    series = pd.Series(np.asarray(values, dtype=np.float64).ravel()).replace([np.inf, -np.inf], np.nan)
    total = int(len(series))
    invalid_before = int(series.isna().sum())
    invalid_ratio = invalid_before / total if total else 1.0
    if total == 0:
        raise ValueError("At least one numeric load observation is required.")
    if invalid_ratio > max_invalid_ratio:
        raise ValueError(f"Too many invalid load observations ({invalid_ratio:.1%}).")
    series = series.interpolate(limit=6, limit_direction="both").ffill(limit=3).bfill(limit=3)
    if series.isna().any():
        raise ValueError("Load history still contains missing values after safe interpolation.")
    cleaned = np.maximum(series.to_numpy(dtype=np.float64), 0.0)
    return cleaned, {
        "total": total,
        "invalid_before": invalid_before,
        "invalid_ratio": round(invalid_ratio, 4),
        "filled": invalid_before,
    }
