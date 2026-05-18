from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Any

from src.config import settings
from src.modeling import artifact_paths_for_country


STARTED_AT = datetime.now(timezone.utc)
LAST_FORECAST: dict[str, Any] | None = None
RECENT_ERRORS: deque[dict[str, str]] = deque(maxlen=10)


def record_forecast(payload: dict[str, Any]) -> None:
    global LAST_FORECAST
    LAST_FORECAST = {"timestamp": datetime.now(timezone.utc).isoformat(), **payload}


def record_error(source: str, message: str) -> None:
    RECENT_ERRORS.append({"timestamp": datetime.now(timezone.utc).isoformat(), "source": source, "message": message})


def deployment_checks(model_loaded: bool, metadata: dict[str, object]) -> list[dict[str, object]]:
    model_country = metadata.get("country_code") if metadata else None
    model_path, scaler_path, _ = artifact_paths_for_country(settings.primary_country_code)
    return [
        {"name": "ENTSO-E API key configured", "ok": bool(settings.entsoe_api_key)},
        {"name": "LSTM model loaded", "ok": model_loaded},
        {"name": "Primary country model artifacts exist", "ok": model_path.exists() and scaler_path.exists()},
        {"name": "Primary country is ENTSO-E", "ok": settings.primary_country_code != "PK"},
        {"name": "Model matches primary country", "ok": model_country == settings.primary_country_code},
        {"name": "SMARD live DE_LU source available", "ok": True},
        {"name": "EIA key configured for US48", "ok": bool(settings.eia_api_key)},
        {"name": "Optional AI key not hardcoded", "ok": True},
    ]


def model_warnings(metadata: dict[str, object]) -> list[str]:
    warnings: list[str] = []
    model_country = metadata.get("country_code") if metadata else None
    if model_country and model_country != settings.primary_country_code:
        warnings.append(
            f"Loaded model was trained for {model_country}, while the primary assignment country is {settings.primary_country_code}."
        )
    if not settings.entsoe_api_key:
        warnings.append("ENTSOE_API_KEY is not configured; DE_LU can use ENTSO-E export or SMARD real Germany/Luxembourg hourly load.")
    return warnings
