from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Electricity Load Forecasting API")
    app_version: str = os.getenv("APP_VERSION", "1.0.0")
    environment: str = os.getenv("ENVIRONMENT", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    entsoe_api_key: str | None = os.getenv("ENTSOE_API_KEY")
    eia_api_key: str | None = os.getenv("EIA_API_KEY")
    pakistan_load_csv: Path = Path(os.getenv("PAKISTAN_LOAD_CSV", "data/pakistan_load.csv"))
    de_lu_load_csv: Path = Path(os.getenv("DE_LU_LOAD_CSV", "data/de_lu_load.csv"))
    entsoe_export_raw_path: Path = Path(os.getenv("ENTSOE_EXPORT_RAW_PATH", "data/de_lu_load_raw.csv"))
    models_dir: Path = Path(os.getenv("MODELS_DIR", "models"))
    model_path: Path = Path(os.getenv("MODEL_PATH", "models/load_lstm.keras"))
    scaler_path: Path = Path(os.getenv("SCALER_PATH", "models/load_scaler.joblib"))
    metadata_path: Path = Path(os.getenv("METADATA_PATH", "models/metadata.json"))
    data_cache_ttl_seconds: int = int(os.getenv("DATA_CACHE_TTL_SECONDS", "900"))
    max_history_days: int = int(os.getenv("MAX_HISTORY_DAYS", "730"))
    forecast_history_days: int = int(os.getenv("FORECAST_HISTORY_DAYS", "45"))
    primary_country_code: str = os.getenv("PRIMARY_COUNTRY_CODE", "DE_LU")
    enable_demo_countries: bool = _env_bool("ENABLE_DEMO_COUNTRIES", False)
    stale_model_days: int = int(os.getenv("STALE_MODEL_DAYS", "30"))
    opencode_api_key: str | None = os.getenv("OPENCODE_API_KEY")
    opencode_base_url: str = os.getenv("OPENCODE_BASE_URL", "https://opencode.ai/zen/v1")
    opencode_model: str = os.getenv("OPENCODE_MODEL", "nemotron-3-super-free")
    ai_explanation_timeout_seconds: int = _env_int("AI_EXPLANATION_TIMEOUT_SECONDS", 5)
    ai_explanation_max_tokens: int = _env_int("AI_EXPLANATION_MAX_TOKENS", 900)
    ai_explanation_retries: int = _env_int("AI_EXPLANATION_RETRIES", 0)
    ai_explanation_on_forecast: bool = _env_bool("AI_EXPLANATION_ON_FORECAST", False)
    ai_explanation_cooldown_seconds: int = _env_int("AI_EXPLANATION_COOLDOWN_SECONDS", 300)
    max_request_bytes: int = _env_int("MAX_REQUEST_BYTES", 1_000_000)
    enable_train_endpoint: bool = _env_bool("ENABLE_TRAIN_ENDPOINT", False)
    train_api_token: str | None = os.getenv("TRAIN_API_TOKEN")
    max_train_epochs: int = _env_int("MAX_TRAIN_EPOCHS", 30)
    max_explain_forecast_points: int = _env_int("MAX_EXPLAIN_FORECAST_POINTS", 336)
    api_retry_attempts: int = _env_int("API_RETRY_ATTEMPTS", 3)
    api_retry_backoff_seconds: float = _env_float("API_RETRY_BACKOFF_SECONDS", 0.75)
    cors_origins: tuple[str, ...] = tuple(
        origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",") if origin.strip()
    )


settings = Settings()
