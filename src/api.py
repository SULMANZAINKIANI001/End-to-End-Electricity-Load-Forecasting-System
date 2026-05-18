from __future__ import annotations

import logging
import subprocess
import sys
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import timedelta

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Query, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config import settings
from src.data import (
    DEFAULT_COUNTRY_CODE,
    SUPPORTED_COUNTRIES,
    DataFetchError,
    fetch_load_data_with_metadata,
    get_available_data_range,
    public_supported_countries,
)
from src.eda import build_eda_payload, build_plot_payload
from src.explain import build_forecast_explanation
from src.logging_config import configure_logging
from src.modeling import SEQUENCE_LENGTH, artifact_paths_for_country, confidence_bounds, iterative_forecast, load_artifacts, seasonal_naive_forecast
from src.safety import clean_nonnegative_series, finite_float, sanitize_for_json
import src.monitoring as monitoring


MODEL_CACHE: dict[str, tuple[object | None, object | None, dict[str, object]]] = {}
MODEL_CACHE_LOCK = threading.RLock()
TRAIN_JOBS: dict[str, dict[str, object]] = {}
TRAIN_LOCK = threading.RLock()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging()
    _get_artifacts_for_country(settings.primary_country_code)
    logger.info("Application started. cached_models=%s", list(MODEL_CACHE))
    yield


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
allow_all_origins = "*" in settings.cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=not allow_all_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_size_limit(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            size = int(content_length)
        except ValueError:
            size = 0
        if size > settings.max_request_bytes:
            return JSONResponse(
                status_code=413,
                content={"detail": f"Request body is too large. Limit is {settings.max_request_bytes} bytes."},
            )
    response = await call_next(request)
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return await request_validation_exception_handler(request, exc)


@app.exception_handler(DataFetchError)
async def data_fetch_exception_handler(_: Request, exc: DataFetchError):
    monitoring.record_error("data_fetch", str(exc))
    return JSONResponse(status_code=502, content={"detail": str(exc), "error_type": "data_fetch_error"})


@app.exception_handler(ValueError)
async def value_error_exception_handler(_: Request, exc: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(exc), "error_type": "validation_error"})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = str(uuid.uuid4())
    logger.exception("Unhandled API error request_id=%s path=%s", request_id, request.url.path)
    monitoring.record_error("internal_error", f"{request_id}: {type(exc).__name__}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error while processing the request.",
            "request_id": request_id,
            "error_type": "internal_error",
        },
    )


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    country_code: str = Field(default=DEFAULT_COUNTRY_CODE, min_length=2)
    start: str | None = Field(default=None, description="History start date or timestamp.")
    end: str | None = Field(default=None, description="History end date or timestamp.")
    horizon: int = Field(default=24, ge=1, le=168)

    @field_validator("country_code")
    @classmethod
    def validate_country(cls, v: str) -> str:
        normalized = v.upper()
        if normalized not in SUPPORTED_COUNTRIES:
            supported = ", ".join(sorted(SUPPORTED_COUNTRIES))
            raise ValueError(f"Unsupported country_code `{v}`. Supported: {supported}")
        return normalized


class ForecastPoint(BaseModel):
    timestamp: str
    predicted_load_mw: float
    lower_bound_mw: float
    upper_bound_mw: float
    baseline_load_mw: float


class HistoryPoint(BaseModel):
    timestamp: str
    actual_load_mw: float


class ForecastResponse(BaseModel):
    country_code: str
    horizon: int
    forecast_method: str
    data_source: dict[str, object]
    history_window: dict[str, object]
    history: list[HistoryPoint]
    forecast: list[ForecastPoint]
    model_metadata: dict[str, object]
    explanation: dict[str, str]
    warnings: list[str] = []
    forecast_summary: dict[str, object] = {}
    hourly_profile: dict[str, float] = {}
    data_freshness: dict[str, object] = {}
    recent_accuracy: dict[str, object] = {}


class ExplainForecastPoint(BaseModel):
    model_config = ConfigDict(extra="ignore")

    timestamp: str | None = None
    predicted_load_mw: float | None = None
    lower_bound_mw: float | None = None
    upper_bound_mw: float | None = None
    baseline_load_mw: float | None = None


class ExplainRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    country_code: str | None = None
    horizon: int | None = Field(default=None, ge=1, le=168)
    forecast_method: str | None = None
    data_source: dict[str, object] = Field(default_factory=dict)
    history_window: dict[str, object] = Field(default_factory=dict)
    forecast: list[ExplainForecastPoint] = Field(default_factory=list, max_length=settings.max_explain_forecast_points)
    model_metadata: dict[str, object] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list, max_length=25)
    forecast_summary: dict[str, object] = Field(default_factory=dict)
    recent_accuracy: dict[str, object] = Field(default_factory=dict)


class TrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    country_code: str = Field(default=settings.primary_country_code, min_length=2)
    start: str | None = None
    end: str | None = None
    years: int = Field(default=2, ge=1, le=5)
    horizon: int = Field(default=24, ge=1, le=168)
    epochs: int = Field(default=10, ge=1)
    batch_size: int = Field(default=64, ge=8, le=512)

    @field_validator("country_code")
    @classmethod
    def validate_country(cls, v: str) -> str:
        normalized = v.upper()
        if normalized not in SUPPORTED_COUNTRIES:
            supported = ", ".join(sorted(SUPPORTED_COUNTRIES))
            raise ValueError(f"Unsupported country_code `{v}`. Supported: {supported}")
        return normalized

    @field_validator("epochs")
    @classmethod
    def clamp_epochs(cls, v: int) -> int:
        if v > settings.max_train_epochs:
            raise ValueError(f"epochs cannot exceed MAX_TRAIN_EPOCHS={settings.max_train_epochs}.")
        return v


def _get_artifacts_for_country(country_code: str) -> tuple[object | None, object | None, dict[str, object]]:
    normalized_country = country_code.upper()
    with MODEL_CACHE_LOCK:
        if normalized_country not in MODEL_CACHE:
            MODEL_CACHE[normalized_country] = load_artifacts(country_code=normalized_country)
        return MODEL_CACHE[normalized_country]


def _refresh_artifacts_for_country(country_code: str) -> tuple[object | None, object | None, dict[str, object]]:
    normalized_country = country_code.upper()
    artifacts = load_artifacts(country_code=normalized_country)
    with MODEL_CACHE_LOCK:
        MODEL_CACHE[normalized_country] = artifacts
    return artifacts


def _model_ready_for_country(country_code: str) -> bool:
    model, scaler, metadata = _get_artifacts_for_country(country_code)
    return model is not None and scaler is not None and str(metadata.get("country_code", "")).upper() == country_code.upper()


def _country_model_status(country_code: str) -> dict[str, object]:
    normalized_country = country_code.upper()
    model, scaler, metadata = _get_artifacts_for_country(normalized_country)
    model_path, scaler_path, metadata_path = artifact_paths_for_country(normalized_country)
    model_country = str(metadata.get("country_code", "")).upper() if metadata else None
    ready = model is not None and scaler is not None and model_country == normalized_country
    return {
        "country_code": normalized_country,
        "model_loaded": model is not None and scaler is not None,
        "model_ready": ready,
        "artifact_country_code": model_country,
        "model_exists": model_path.exists(),
        "scaler_exists": scaler_path.exists(),
        "metadata_exists": metadata_path.exists(),
        "forecast_method": "lstm" if ready else "seasonal_naive_fallback",
        "training_command": f"py -3.10 -m src.train --country-code {normalized_country} --years 2 --horizon 24 --epochs 30",
        "metadata": metadata,
    }


@app.get("/health")
def health() -> dict[str, object]:
    primary = _country_model_status(settings.primary_country_code)
    return sanitize_for_json({
        "status": "ok",
        "environment": settings.environment,
        "model_loaded": primary["model_loaded"],
        "metadata": primary["metadata"],
    })


@app.get("/settings")
def app_settings() -> dict[str, object]:
    return sanitize_for_json({
        "max_history_days": settings.max_history_days,
        "forecast_history_days": settings.forecast_history_days,
        "default_history_days": min(90, settings.max_history_days),
        "max_forecast_horizon": 168,
        "default_forecast_horizon": 24,
        "primary_country_code": settings.primary_country_code,
        "environment": settings.environment,
    })


@app.get("/model/status")
def model_status(country_code: str = Query(settings.primary_country_code, min_length=2)) -> dict[str, object]:
    status = _country_model_status(country_code)
    primary_ready = _model_ready_for_country(settings.primary_country_code)
    return sanitize_for_json({
        **status,
        "primary_country_code": settings.primary_country_code,
        "primary_model_ready": primary_ready,
        "warnings": monitoring.model_warnings(status["metadata"]),
    })


@app.get("/monitoring/status")
def monitoring_status() -> dict[str, object]:
    primary_status = _country_model_status(settings.primary_country_code)
    model_loaded = bool(primary_status["model_loaded"])
    country_statuses = {country: _country_model_status(country) for country in SUPPORTED_COUNTRIES}
    return sanitize_for_json({
        "status": "ok",
        "started_at": monitoring.STARTED_AT.isoformat(),
        "environment": settings.environment,
        "primary_country_code": settings.primary_country_code,
        "demo_countries_enabled": settings.enable_demo_countries,
        "model_loaded": model_loaded,
        "model_metadata": primary_status["metadata"],
        "country_models": country_statuses,
        "last_forecast": monitoring.LAST_FORECAST,
        "recent_errors": monitoring.RECENT_ERRORS,
        "warnings": monitoring.model_warnings(primary_status["metadata"]),
        "deployment_checks": monitoring.deployment_checks(model_loaded, primary_status["metadata"]),
    })


@app.get("/countries")
def countries(
    include_demo: bool = Query(settings.enable_demo_countries),
    forecast_ready_only: bool = Query(False),
) -> dict[str, object]:
    selected = public_supported_countries(include_demo=include_demo)
    enriched: dict[str, object] = {}
    for code, metadata in selected.items():
        model_status = _country_model_status(code)
        enriched[code] = {
            **metadata,
            "model_ready": model_status["model_ready"],
            "forecast_method": model_status["forecast_method"],
            "artifact_country_code": model_status["artifact_country_code"],
            "metrics": model_status["metadata"].get("metrics", {}) if isinstance(model_status["metadata"], dict) else {},
        }
    if forecast_ready_only:
        enriched = {code: metadata for code, metadata in enriched.items() if metadata.get("model_ready")}
    return sanitize_for_json({
        "countries": enriched,
        "mode": "forecast_ready" if forecast_ready_only else ("all_sources" if include_demo else "real_data_only"),
        "demo_available": any(value.get("data_class") == "historical_demo" for value in SUPPORTED_COUNTRIES.values()),
    })


@app.get("/data/range")
def data_range(country_code: str = Query(DEFAULT_COUNTRY_CODE, min_length=2)) -> dict[str, object]:
    try:
        return sanitize_for_json(get_available_data_range(country_code))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except DataFetchError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def _is_date_only(value: str | None) -> bool:
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    return len(stripped) == 10 and stripped.count("-") == 2


def _parse_range(start: str | None, end: str | None) -> tuple[pd.Timestamp, pd.Timestamp]:
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(days=2)
    if _is_date_only(end):
        end_ts = end_ts + pd.Timedelta(days=1)
    start_ts = pd.Timestamp(start) if start else end_ts - pd.Timedelta(days=30)
    if end_ts <= start_ts:
        raise HTTPException(status_code=422, detail="end must be after start.")
    return start_ts, end_ts


def _fetch_or_400(country_code: str, start: str | None, end: str | None):
    start_ts, end_ts = _parse_range(start, end)
    return _fetch_range_or_400(country_code, start_ts, end_ts)


def _fetch_range_or_400(country_code: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp):
    try:
        return fetch_load_data_with_metadata(country_code, start_ts, end_ts)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except DataFetchError as exc:
        monitoring.record_error("data_fetch", str(exc))
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def _align_to_index_timezone(timestamp: pd.Timestamp, index: pd.DatetimeIndex) -> pd.Timestamp:
    aligned = pd.Timestamp(timestamp)
    if index.tz is None:
        return aligned.tz_localize(None) if aligned.tzinfo is not None else aligned
    if aligned.tzinfo is None:
        return aligned.tz_localize(index.tz)
    return aligned.tz_convert(index.tz)


def _empty_forecast_summary(reason: str = "no_forecast_points") -> dict[str, object]:
    return {
        "available": False,
        "reason": reason,
        "average_load_mw": None,
        "min_load_mw": None,
        "max_load_mw": None,
        "std_load_mw": None,
        "peak_at": None,
        "trough_at": None,
        "trend_direction": "unknown",
        "trend_change_pct": None,
        "peak_to_avg_ratio": None,
        "load_change_rate_mw_per_hour": None,
        "baseline_gap_mw": None,
    }


def _empty_recent_accuracy(reason: str = "insufficient_history") -> dict[str, object]:
    return {
        "available": False,
        "reason": reason,
        "method": "seasonal_naive_backtest_last_24h",
        "mae_mw": None,
        "rmse_mw": None,
        "mape_pct": None,
        "note": "Measures how well repeating yesterday matches today. Not a full model backtest.",
    }


def _clean_model_input_df(load_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    cleaned_values, quality = clean_nonnegative_series(load_df["load_mw"].to_numpy(), max_invalid_ratio=0.2)
    cleaned_df = load_df.copy()
    cleaned_df["load_mw"] = cleaned_values
    return cleaned_df, quality


def _safe_round(value: object, digits: int = 3) -> float:
    return round(float(finite_float(value, 0.0, minimum=0.0) or 0.0), digits)


def _compute_forecast_summary(predictions: np.ndarray, forecast_points: list[ForecastPoint], baseline: np.ndarray) -> dict[str, object]:
    """Compute aggregate forecast statistics for the dashboard."""
    if len(predictions) == 0:
        return _empty_forecast_summary()
    try:
        predictions, _ = clean_nonnegative_series(predictions, max_invalid_ratio=0.5)
    except ValueError:
        return _empty_forecast_summary("invalid_forecast_values")
    if len(predictions) == 0:
        return _empty_forecast_summary()
    baseline = np.asarray(baseline, dtype=np.float64).ravel()
    baseline = np.nan_to_num(baseline, nan=0.0, posinf=0.0, neginf=0.0)
    baseline = np.maximum(baseline, 0.0)
    peak_idx = int(np.argmax(predictions))
    trough_idx = int(np.argmin(predictions))
    avg_load = float(np.mean(predictions))
    std_load = float(np.std(predictions)) if len(predictions) > 1 else 0.0
    first, last = float(predictions[0]), float(predictions[-1])
    trend_pct = ((last - first) / first * 100) if first != 0 else 0.0
    if trend_pct > 3:
        trend_direction = "rising"
    elif trend_pct < -3:
        trend_direction = "falling"
    else:
        trend_direction = "stable"
    peak_to_avg = float(predictions[peak_idx]) / avg_load if avg_load > 0 else 1.0
    last_actual_val = predictions[0]
    load_change_rate = float(np.mean(np.diff(predictions))) if len(predictions) > 1 else 0.0
    baseline_gap = float(np.mean(predictions - baseline)) if len(baseline) == len(predictions) else 0.0
    return {
        "available": True,
        "reason": None,
        "average_load_mw": round(avg_load, 3),
        "min_load_mw": round(float(np.min(predictions)), 3),
        "max_load_mw": round(float(np.max(predictions)), 3),
        "std_load_mw": round(std_load, 3),
        "peak_at": forecast_points[peak_idx].timestamp if peak_idx < len(forecast_points) else None,
        "trough_at": forecast_points[trough_idx].timestamp if trough_idx < len(forecast_points) else None,
        "trend_direction": trend_direction,
        "trend_change_pct": round(trend_pct, 2),
        "peak_to_avg_ratio": round(peak_to_avg, 3),
        "load_change_rate_mw_per_hour": round(load_change_rate, 3),
        "baseline_gap_mw": round(baseline_gap, 3),
    }


def _compute_hourly_profile(predictions: np.ndarray, forecast_points: list[ForecastPoint]) -> dict[str, float]:
    """Compute time-of-day average load profile from forecast points."""
    try:
        predictions, _ = clean_nonnegative_series(predictions, max_invalid_ratio=0.5)
    except ValueError:
        predictions = np.zeros(len(forecast_points), dtype=np.float64)
    hour_groups: dict[str, list[float]] = {
        "night_00_05": [], "morning_06_11": [], "afternoon_12_17": [], "evening_18_23": [],
    }
    for i, pt in enumerate(forecast_points):
        try:
            ts = pd.Timestamp(pt.timestamp)
            hour = ts.hour
        except Exception:
            continue
        val = finite_float(predictions[i], 0.0, minimum=0.0) if i < len(predictions) else 0.0
        if hour < 6:
            hour_groups["night_00_05"].append(val)
        elif hour < 12:
            hour_groups["morning_06_11"].append(val)
        elif hour < 18:
            hour_groups["afternoon_12_17"].append(val)
        else:
            hour_groups["evening_18_23"].append(val)
    return {k: round(float(np.mean(v)), 3) if v else 0.0 for k, v in hour_groups.items()}


def _compute_data_freshness(data_source: dict[str, object]) -> dict[str, object]:
    """Compute how stale the source data is relative to now."""
    latest = data_source.get("latest_timestamp")
    if not latest:
        return {"latest_timestamp": None, "hours_behind_now": None, "staleness": "unknown"}
    try:
        latest_ts = pd.Timestamp(latest)
        now = pd.Timestamp.now(tz="UTC")
        if latest_ts.tzinfo is None:
            latest_ts = latest_ts.tz_localize("UTC")
        delta_hours = (now - latest_ts).total_seconds() / 3600
        if delta_hours < 3:
            staleness = "fresh"
        elif delta_hours < 24:
            staleness = "stale"
        else:
            staleness = "very_stale"
        return {
            "latest_timestamp": str(latest),
            "hours_behind_now": round(delta_hours, 1),
            "staleness": staleness,
        }
    except Exception:
        return {"latest_timestamp": str(latest), "hours_behind_now": None, "staleness": "unknown"}


def _compute_recent_accuracy(load_df: pd.DataFrame) -> dict[str, object]:
    """Quick backtest: compare last 24h of actuals against a naive seasonal forecast."""
    if len(load_df) < 48:
        return _empty_recent_accuracy()
    try:
        actuals, _ = clean_nonnegative_series(load_df["load_mw"].tail(24).to_numpy(dtype=np.float64), max_invalid_ratio=0.25)
        prior, _ = clean_nonnegative_series(load_df["load_mw"].iloc[-48:-24].to_numpy(dtype=np.float64), max_invalid_ratio=0.25)
    except ValueError:
        return _empty_recent_accuracy("invalid_recent_values")
    if len(prior) < 24:
        return _empty_recent_accuracy()
    naive_forecast = prior[-24:]
    errors = np.abs(actuals - naive_forecast)
    mae = float(np.mean(errors))
    non_zero = np.where(actuals == 0, np.nan, actuals)
    mape = finite_float(np.nanmean(np.abs((actuals - naive_forecast) / non_zero)) * 100)
    rmse = float(np.sqrt(np.mean((actuals - naive_forecast) ** 2)))
    return {
        "available": True,
        "method": "seasonal_naive_backtest_last_24h",
        "mae_mw": round(mae, 3),
        "rmse_mw": round(rmse, 3),
        "mape_pct": round(float(mape), 3) if mape is not None else None,
        "note": "Measures how well repeating yesterday matches today. Not a full model backtest.",
    }


@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest) -> ForecastResponse:
    logger.info("Forecast requested country=%s horizon=%s", request.country_code, request.horizon)
    requested_start, requested_end = _parse_range(request.start, request.end)
    max_runtime_window = pd.Timedelta(days=settings.forecast_history_days)
    effective_start = max(requested_start, requested_end - max_runtime_window)
    model_input_was_capped = effective_start > requested_start
    load_df, data_source = _fetch_range_or_400(request.country_code, requested_start, requested_end)
    load_df, history_quality = _clean_model_input_df(load_df)
    if history_quality.get("filled"):
        source_quality = data_source.setdefault("data_quality", {})
        if isinstance(source_quality, dict):
            source_quality.update(history_quality)
    effective_start_for_index = _align_to_index_timezone(effective_start, load_df.index)
    model_input_df = load_df.loc[load_df.index >= effective_start_for_index]
    model_input_df, model_input_quality = _clean_model_input_df(model_input_df)
    model, scaler, metadata = _get_artifacts_for_country(request.country_code)
    model_country = str(metadata.get("country_code", "")).upper() if metadata else ""
    request_country = request.country_code.upper()
    model_ready = model is not None and scaler is not None and model_country == request_country
    minimum_rows = SEQUENCE_LENGTH if model_ready else 24
    if len(model_input_df) < minimum_rows:
        raise HTTPException(
            status_code=400,
            detail=f"At least {minimum_rows} recent hourly records are required; received {len(model_input_df)}.",
        )

    if model_ready:
        recent_values = model_input_df["load_mw"].tail(SEQUENCE_LENGTH).to_numpy()
        predictions = iterative_forecast(model, scaler, recent_values, request.horizon, sequence_length=SEQUENCE_LENGTH)
        forecast_method = "lstm"
        model_metadata = metadata
    else:
        recent_values = model_input_df["load_mw"].tail(24).to_numpy()
        predictions = seasonal_naive_forecast(recent_values, request.horizon, season_length=24)
        forecast_method = "seasonal_naive_fallback"
        model_metadata = {
            "warning": "LSTM model artifacts are not available. Showing a seasonal naive fallback forecast.",
            "next_step": "Train the model with `python -m src.train` on Python 3.10 to enable LSTM forecasts.",
        }
        if model is not None and scaler is not None and model_country and model_country != request_country:
            model_metadata["warning"] = (
                f"Loaded LSTM artifacts are for {model_country}, but this request is for {request_country}. "
                "Showing a seasonal naive fallback instead of using the wrong model."
            )
            model_metadata["next_step"] = f"Train matching artifacts with `python -m src.train --country-code {request_country}`."

    predictions = np.maximum(np.nan_to_num(np.asarray(predictions, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    baseline = seasonal_naive_forecast(model_input_df["load_mw"].tail(24).to_numpy(), request.horizon, season_length=24)
    baseline = np.maximum(np.nan_to_num(np.asarray(baseline, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    lower_bounds, upper_bounds = confidence_bounds(predictions, model_metadata)
    lower_bounds = np.maximum(np.nan_to_num(lower_bounds, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    upper_bounds = np.maximum(np.nan_to_num(upper_bounds, nan=0.0, posinf=0.0, neginf=0.0), lower_bounds)
    upper_bounds = np.maximum(upper_bounds, predictions)
    lower_bounds = np.minimum(lower_bounds, predictions)
    last_timestamp = model_input_df.index.max()
    history_points = [
        HistoryPoint(timestamp=str(ts), actual_load_mw=_safe_round(val))
        for ts, val in zip(load_df.index, load_df["load_mw"])
    ]
    forecast_points = [
        ForecastPoint(
            timestamp=(last_timestamp + timedelta(hours=index + 1)).isoformat(),
            predicted_load_mw=_safe_round(value),
            lower_bound_mw=_safe_round(lower_bounds[index]),
            upper_bound_mw=_safe_round(upper_bounds[index]),
            baseline_load_mw=_safe_round(baseline[index]),
        )
        for index, value in enumerate(predictions)
    ]
    forecast_summary = _compute_forecast_summary(predictions, forecast_points, baseline)
    hourly_profile = _compute_hourly_profile(predictions, forecast_points)
    data_freshness = _compute_data_freshness(data_source)
    recent_accuracy = _compute_recent_accuracy(load_df)
    forecast_warnings: list[str] = []
    if isinstance(model_metadata, dict) and model_metadata.get("warning"):
        forecast_warnings.append(str(model_metadata["warning"]))
    if request_country == "PK":
        forecast_warnings.append("Pakistan uses historical CSV demo data; ENTSO-E does not publish Pakistan load.")
    if data_source.get("source") == "smard_dynamic_fallback":
        data_source["source_detail"] = "SMARD real Germany/Luxembourg hourly load is active for this forecast."
    if data_source.get("source") == "entsoe_export":
        forecast_warnings.append("DE_LU is using a normalized manual ENTSO-E export, not the REST API token path.")
    if model_input_was_capped:
        forecast_warnings.append(
            f"Forecast engine used the latest {settings.forecast_history_days} days for runtime speed. "
            "The chart still shows the full selected history window."
        )
    response_payload = {
        "country_code": request.country_code,
        "horizon": request.horizon,
        "forecast_method": forecast_method,
        "data_source": data_source,
        "history_window": {
            "selected_start": request.start or requested_start.isoformat(),
            "selected_end": request.end or requested_end.isoformat(),
            "selected_end_inclusive": bool(_is_date_only(request.end)),
            "effective_start": effective_start.isoformat(),
            "effective_end_exclusive": requested_end.isoformat(),
            "actual_start": load_df.index.min().isoformat(),
            "actual_end": load_df.index.max().isoformat(),
            "model_input_start": model_input_df.index.min().isoformat(),
            "model_input_end": model_input_df.index.max().isoformat(),
            "forecast_start": forecast_points[0].timestamp if forecast_points else None,
            "forecast_end": forecast_points[-1].timestamp if forecast_points else None,
            "history_rows": int(len(load_df)),
            "model_input_rows": int(len(model_input_df)),
            "history_capped": model_input_was_capped,
            "max_runtime_history_days": settings.forecast_history_days,
            "note": "Start and end select the chart history window. Date-only end values are treated as inclusive user dates; forecast begins one hour after the latest actual row. Long windows use the latest configured runtime window for neural-network input.",
            "model_input_quality": model_input_quality,
        },
        "history": [point.model_dump() for point in history_points],
        "forecast": [point.model_dump() for point in forecast_points],
        "model_metadata": model_metadata,
        "warnings": forecast_warnings,
        "forecast_summary": forecast_summary,
        "hourly_profile": hourly_profile,
        "data_freshness": data_freshness,
        "recent_accuracy": recent_accuracy,
    }
    response_payload = sanitize_for_json(response_payload)
    explanation = sanitize_for_json(build_forecast_explanation(response_payload, allow_external=settings.ai_explanation_on_forecast))
    monitoring.record_forecast(
        {
            "country_code": request.country_code,
            "horizon": request.horizon,
            "forecast_method": forecast_method,
            "data_source": data_source.get("source", "unknown"),
            "history_start": str(load_df.index.min()),
            "history_end": str(load_df.index.max()),
            "model_input_start": str(model_input_df.index.min()),
            "model_input_end": str(model_input_df.index.max()),
        }
    )
    return ForecastResponse(
        country_code=request.country_code,
        horizon=request.horizon,
        forecast_method=forecast_method,
        data_source=sanitize_for_json(data_source),
        history_window=sanitize_for_json(response_payload["history_window"]),
        history=history_points,
        forecast=forecast_points,
        model_metadata=sanitize_for_json(model_metadata),
        explanation=explanation,
        warnings=response_payload["warnings"],
        forecast_summary=forecast_summary,
        hourly_profile=hourly_profile,
        data_freshness=data_freshness,
        recent_accuracy=recent_accuracy,
    )


@app.post("/explain")
def explain(payload: ExplainRequest) -> dict[str, str]:
    safe_payload = sanitize_for_json(payload.model_dump())
    return sanitize_for_json(build_forecast_explanation(safe_payload, allow_external=True))


def _training_command(request: TrainRequest) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.train",
        "--country-code",
        request.country_code,
        "--years",
        str(request.years),
        "--horizon",
        str(request.horizon),
        "--epochs",
        str(request.epochs),
        "--batch-size",
        str(request.batch_size),
    ]
    if request.start:
        command.extend(["--start", request.start])
    if request.end:
        command.extend(["--end", request.end])
    return command


def _run_training_job(job_id: str, request: TrainRequest) -> None:
    with TRAIN_LOCK:
        TRAIN_JOBS[job_id].update({"status": "running"})
    try:
        completed = subprocess.run(
            _training_command(request),
            check=False,
            capture_output=True,
            text=True,
            timeout=60 * 60 * 6,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr[-4000:] or completed.stdout[-4000:] or "Training command failed.")
        _refresh_artifacts_for_country(request.country_code)
        with TRAIN_LOCK:
            TRAIN_JOBS[job_id].update(
                {
                    "status": "completed",
                    "returncode": completed.returncode,
                    "stdout_tail": completed.stdout[-4000:],
                    "artifact_paths": [str(path) for path in artifact_paths_for_country(request.country_code)],
                }
            )
    except Exception as exc:
        logger.exception("Training job failed job_id=%s country=%s", job_id, request.country_code)
        with TRAIN_LOCK:
            TRAIN_JOBS[job_id].update({"status": "failed", "error": str(exc)})


@app.post("/train")
def train(
    request: TrainRequest,
    background_tasks: BackgroundTasks,
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
) -> dict[str, object]:
    if not settings.enable_train_endpoint:
        raise HTTPException(status_code=403, detail="Training endpoint is disabled. Set ENABLE_TRAIN_ENDPOINT=true to enable it.")
    if not settings.train_api_token or x_admin_token != settings.train_api_token:
        raise HTTPException(status_code=401, detail="Invalid or missing admin token.")
    with TRAIN_LOCK:
        for job in TRAIN_JOBS.values():
            if job.get("country_code") == request.country_code and job.get("status") in {"queued", "running"}:
                raise HTTPException(status_code=409, detail=f"A training job is already running for {request.country_code}.")
        job_id = str(uuid.uuid4())
        TRAIN_JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "country_code": request.country_code,
            "request": sanitize_for_json(request.model_dump()),
        }
    background_tasks.add_task(_run_training_job, job_id, request)
    return sanitize_for_json({"job_id": job_id, "status": "queued"})


@app.get("/train/status/{job_id}")
def train_status(job_id: str) -> dict[str, object]:
    with TRAIN_LOCK:
        job = TRAIN_JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Training job not found.")
        return sanitize_for_json(job)


@app.get("/eda/summary")
def eda_summary(
    country_code: str = Query(DEFAULT_COUNTRY_CODE, min_length=2),
    start: str | None = Query(None),
    end: str | None = Query(None),
) -> dict[str, object]:
    load_df, _ = _fetch_or_400(country_code, start, end)
    return sanitize_for_json({"country_code": country_code, **build_eda_payload(load_df)})


@app.get("/eda/plots")
def eda_plots(
    country_code: str = Query(DEFAULT_COUNTRY_CODE, min_length=2),
    start: str | None = Query(None),
    end: str | None = Query(None),
) -> dict[str, object]:
    load_df, _ = _fetch_or_400(country_code, start, end)
    return sanitize_for_json({"country_code": country_code, "plots": build_plot_payload(load_df)})
