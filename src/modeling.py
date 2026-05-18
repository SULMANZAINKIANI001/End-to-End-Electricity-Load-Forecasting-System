from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from src.config import settings
from src.safety import clean_nonnegative_series, finite_float


SEQUENCE_LENGTH = 168
MODEL_PATH = settings.model_path
SCALER_PATH = settings.scaler_path
METADATA_PATH = settings.metadata_path


def normalize_country_code(country_code: str) -> str:
    return country_code.upper().replace("-", "_")


def artifact_dir_for_country(country_code: str) -> Path:
    return settings.models_dir / normalize_country_code(country_code)


def artifact_paths_for_country(country_code: str) -> tuple[Path, Path, Path]:
    artifact_dir = artifact_dir_for_country(country_code)
    return artifact_dir / "load_lstm.keras", artifact_dir / "load_scaler.joblib", artifact_dir / "metadata.json"


def legacy_artifacts_match_country(country_code: str) -> bool:
    if not METADATA_PATH.exists():
        return False
    try:
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except Exception:
        return False
    return str(metadata.get("country_code", "")).upper() == country_code.upper()


def create_sequences(values: np.ndarray, sequence_length: int = SEQUENCE_LENGTH, horizon: int = 24) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    x_values: list[np.ndarray] = []
    y_values: list[np.ndarray] = []
    stop = len(values) - sequence_length - horizon + 1
    for index in range(max(stop, 0)):
        x_values.append(values[index : index + sequence_length])
        y_values.append(values[index + sequence_length : index + sequence_length + horizon, 0])
    return np.asarray(x_values, dtype=np.float32), np.asarray(y_values, dtype=np.float32)


def split_sequences(x_values: np.ndarray, y_values: np.ndarray) -> tuple[np.ndarray, ...]:
    train_end = int(len(x_values) * 0.7)
    val_end = int(len(x_values) * 0.85)
    return (
        x_values[:train_end],
        y_values[:train_end],
        x_values[train_end:val_end],
        y_values[train_end:val_end],
        x_values[val_end:],
        y_values[val_end:],
    )


def build_lstm_model(sequence_length: int = SEQUENCE_LENGTH, horizon: int = 24):
    from tensorflow import keras

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(sequence_length, 1)),
            keras.layers.LSTM(96, return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(64),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(horizon),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray, scaler) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    true_inverse = scaler.inverse_transform(y_true.reshape(-1, 1)).ravel()
    pred_inverse = scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
    mae = mean_absolute_error(true_inverse, pred_inverse)
    rmse = mean_squared_error(true_inverse, pred_inverse) ** 0.5
    non_zero = np.where(true_inverse == 0, np.nan, true_inverse)
    mape = np.nanmean(np.abs((true_inverse - pred_inverse) / non_zero)) * 100
    return {"mae": round(float(mae), 3), "rmse": round(float(rmse), 3), "mape": round(float(mape), 3)}


def residual_summary(y_true: np.ndarray, y_pred: np.ndarray, scaler) -> dict[str, float]:
    true_inverse = scaler.inverse_transform(y_true.reshape(-1, 1)).ravel()
    pred_inverse = scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
    residuals = true_inverse - pred_inverse
    return {
        "mean_residual_mw": round(float(np.mean(residuals)), 3),
        "std_residual_mw": round(float(np.std(residuals)), 3),
        "p90_abs_residual_mw": round(float(np.quantile(np.abs(residuals), 0.9)), 3),
    }


def confidence_bounds(predictions: np.ndarray, metadata: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    predictions, _ = clean_nonnegative_series(predictions, max_invalid_ratio=0.5)
    residuals = metadata.get("residuals", {}) if isinstance(metadata, dict) else {}
    if isinstance(residuals, dict):
        band = residuals.get("p90_abs_residual_mw") or residuals.get("std_residual_mw")
    else:
        band = None
    if not isinstance(band, (int, float)) or band <= 0:
        metrics = metadata.get("metrics", {}) if isinstance(metadata, dict) else {}
        band = metrics.get("rmse", 0) if isinstance(metrics, dict) else 0
    safe_band = finite_float(band, None, minimum=0.0)
    base_band = safe_band if safe_band and safe_band > 0 else float(np.std(predictions) * 0.2)
    if not np.isfinite(base_band) or base_band <= 0:
        base_band = max(float(np.nanmean(predictions)) * 0.05, 1.0)
    horizon = len(predictions)
    # The residual summary is already measured in MW on held-out forecast errors.
    # Use only mild horizon widening so short operational forecasts do not show
    # unrealistic reserve envelopes.
    steps = np.arange(horizon, dtype=np.float64)
    growth = 1.0 + np.minimum(steps * 0.01, 0.5)
    scaled_band = base_band * growth
    lower = np.maximum(np.nan_to_num(predictions - scaled_band, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    upper = np.maximum(np.nan_to_num(predictions + scaled_band, nan=0.0, posinf=0.0, neginf=0.0), lower)
    return lower, upper


def iterative_forecast(model, scaler, recent_load_mw: np.ndarray, horizon: int, sequence_length: int = SEQUENCE_LENGTH) -> np.ndarray:
    recent_load_mw, _ = clean_nonnegative_series(recent_load_mw)
    if len(recent_load_mw) < sequence_length:
        raise ValueError(f"At least {sequence_length} recent hourly observations are required.")

    scaled = scaler.transform(np.asarray(recent_load_mw, dtype=np.float32).reshape(-1, 1)).ravel()
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
    scaled_history = scaled.tolist()
    predictions: list[float] = []

    while len(predictions) < horizon:
        window = np.asarray(scaled_history[-sequence_length:], dtype=np.float32).reshape(1, sequence_length, 1)
        batch_pred = np.asarray(model.predict(window, verbose=0), dtype=np.float64).ravel()
        batch_pred = np.nan_to_num(batch_pred, nan=scaled_history[-1], posinf=1.0, neginf=0.0)
        batch_pred = np.clip(batch_pred, 0.0, 1.5).tolist()
        needed = horizon - len(predictions)
        selected = batch_pred[:needed]
        predictions.extend(selected)
        scaled_history.extend(selected)

    inverse = scaler.inverse_transform(np.asarray(predictions).reshape(-1, 1)).ravel()
    return np.maximum(np.nan_to_num(inverse, nan=0.0, posinf=0.0, neginf=0.0), 0.0)


def seasonal_naive_forecast(recent_load_mw: np.ndarray, horizon: int, season_length: int = 24) -> np.ndarray:
    """Forecast from recent history when trained model artifacts are not available.

    When at least 7 days of history are available, builds a day-of-week
    adjusted hourly profile so weekday/weekend patterns are respected.
    Falls back to simple last-24h repetition for shorter windows.
    """

    values, _ = clean_nonnegative_series(recent_load_mw, max_invalid_ratio=0.5)
    values = values.astype(np.float32)
    if len(values) == 0:
        raise ValueError("At least one recent load observation is required.")

    if len(values) >= season_length * 7:
        num_days = len(values) // season_length
        tail = values[-(num_days * season_length):]
        day_matrix = tail.reshape(num_days, season_length)
        dow_profiles: dict[int, list[float]] = {}
        for day_idx in range(num_days):
            dow = day_idx % 7
            dow_profiles.setdefault(dow, []).append(day_matrix[day_idx])
        dow_avg: dict[int, np.ndarray] = {}
        for dow, profiles in dow_profiles.items():
            dow_avg[dow] = np.nanmean(np.array(profiles), axis=0).astype(np.float32)
        if len(dow_avg) < 7:
            pattern = np.nanmean(list(dow_avg.values()), axis=0)
            repeats = int(np.ceil(horizon / season_length))
            return np.maximum(np.nan_to_num(np.tile(pattern, repeats)[:horizon], nan=0.0), 0.0)
        result = np.zeros(horizon, dtype=np.float32)
        for step in range(horizon):
            day_offset = step // season_length
            hour = step % season_length
            dow = day_offset % 7
            result[step] = dow_avg[dow][hour]
        return np.maximum(np.nan_to_num(result, nan=0.0), 0.0)

    pattern_length = min(season_length, len(values))
    pattern = values[-pattern_length:]
    repeats = int(np.ceil(horizon / pattern_length))
    return np.maximum(np.nan_to_num(np.tile(pattern, repeats)[:horizon], nan=0.0), 0.0)


def benchmark_metrics(actual: np.ndarray, benchmark: np.ndarray) -> dict[str, float]:
    if len(actual) == 0 or len(benchmark) == 0:
        return {}
    limit = min(len(actual), len(benchmark))
    actual_values, _ = clean_nonnegative_series(actual[:limit], max_invalid_ratio=0.5)
    benchmark_values, _ = clean_nonnegative_series(benchmark[:limit], max_invalid_ratio=0.5)
    mae = float(np.mean(np.abs(actual_values - benchmark_values)))
    rmse = float(np.sqrt(np.mean((actual_values - benchmark_values) ** 2)))
    non_zero = np.where(actual_values == 0, np.nan, actual_values)
    mape = float(np.nanmean(np.abs((actual_values - benchmark_values) / non_zero)) * 100)
    return {"mae": round(mae, 3), "rmse": round(rmse, 3), "mape": round(mape, 3)}


def save_metadata(metadata: dict[str, object], path: Path = METADATA_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_artifacts(
    model_path: Path | None = None,
    scaler_path: Path | None = None,
    metadata_path: Path | None = None,
    *,
    country_code: str | None = None,
):
    if country_code:
        model_path, scaler_path, metadata_path = artifact_paths_for_country(country_code)
        if not model_path.exists() and legacy_artifacts_match_country(country_code):
            model_path, scaler_path, metadata_path = MODEL_PATH, SCALER_PATH, METADATA_PATH
    else:
        model_path = model_path or MODEL_PATH
        scaler_path = scaler_path or SCALER_PATH
        metadata_path = metadata_path or METADATA_PATH

    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    if not model_path.exists() or not scaler_path.exists():
        return None, None, metadata

    try:
        from tensorflow import keras
    except ModuleNotFoundError:
        return None, None, metadata

    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler, metadata
