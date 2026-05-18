from __future__ import annotations

import os
import random
import threading
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from entsoe import EntsoePandasClient
from pytz.exceptions import AmbiguousTimeError, NonExistentTimeError

from src.config import settings


DEFAULT_TIMEZONE = "Europe/Brussels"
DEFAULT_COUNTRY_CODE = "DE_LU"
COUNTRY_TIMEZONES = {
    "PK": "Asia/Karachi",
    "DE_LU": "Europe/Brussels",
    "FR": "Europe/Paris",
    "NL": "Europe/Amsterdam",
    "BE": "Europe/Brussels",
    "US48": "UTC",
}
SUPPORTED_COUNTRIES = {
    "DE_LU": {
        "name": "Germany/Luxembourg - ENTSO-E or SMARD live",
        "source": "entsoe_or_smard",
        "timezone": COUNTRY_TIMEZONES["DE_LU"],
        "data_class": "real_dynamic",
        "production_ready": True,
        "assignment_compliant": True,
        "live_capable": True,
        "description": "Primary assignment path. Uses ENTSO-E when configured, ENTSO-E export when imported, or SMARD real Germany/Luxembourg hourly load.",
    },
    "US48": {
        "name": "United States Lower 48 - EIA live",
        "source": "eia",
        "timezone": COUNTRY_TIMEZONES["US48"],
        "data_class": "real_live",
        "production_ready": True,
        "assignment_compliant": False,
        "live_capable": True,
        "description": "Live U.S. hourly demand from EIA Open Data. Useful as a second real data source.",
    },
    "FR": {
        "name": "France - ENTSO-E live",
        "source": "entsoe",
        "timezone": COUNTRY_TIMEZONES["FR"],
        "data_class": "real_live",
        "production_ready": True,
        "assignment_compliant": True,
        "live_capable": True,
        "description": "ENTSO-E live country. Requires ENTSOE_API_KEY.",
    },
    "NL": {
        "name": "Netherlands - ENTSO-E live",
        "source": "entsoe",
        "timezone": COUNTRY_TIMEZONES["NL"],
        "data_class": "real_live",
        "production_ready": True,
        "assignment_compliant": True,
        "live_capable": True,
        "description": "ENTSO-E live country. Requires ENTSOE_API_KEY.",
    },
    "BE": {
        "name": "Belgium - ENTSO-E live",
        "source": "entsoe",
        "timezone": COUNTRY_TIMEZONES["BE"],
        "data_class": "real_live",
        "production_ready": True,
        "assignment_compliant": True,
        "live_capable": True,
        "description": "ENTSO-E live country. Requires ENTSOE_API_KEY.",
    },
    "PK": {
        "name": "Pakistan - historical CSV demo",
        "source": "csv",
        "timezone": COUNTRY_TIMEZONES["PK"],
        "data_class": "historical_demo",
        "production_ready": False,
        "assignment_compliant": False,
        "live_capable": False,
        "description": "Historical Pakistan CSV only. Hidden by default because it is not a live public API source.",
    },
}
_FETCH_CACHE_MAX = 32
_FETCH_CACHE: dict[tuple[str, str, str], tuple[float, pd.DataFrame]] = {}
_FETCH_CACHE_LOCK = threading.RLock()
SMARD_BASE_URL = "https://www.smard.de/app/chart_data"
SMARD_TOTAL_LOAD_FILTER = 410
SMARD_REGION = "DE-LU"
EIA_RTO_REGION_ENDPOINT = "https://api.eia.gov/v2/electricity/rto/region-data/data/"


class DataFetchError(RuntimeError):
    """Raised when ENTSO-E data cannot be fetched or normalized."""


def _to_timestamp(value: str | datetime | pd.Timestamp, timezone: str = DEFAULT_TIMEZONE) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        try:
            return timestamp.tz_localize(timezone)
        except (AmbiguousTimeError, ValueError) as exc:
            message = str(exc).lower()
            if "nonexistent" in message:
                return timestamp.tz_localize(timezone, nonexistent="shift_forward")
            if "ambiguous" not in message and "cannot infer dst" not in message:
                raise
            return timestamp.tz_localize(timezone, ambiguous=False)
        except NonExistentTimeError:
            return timestamp.tz_localize(timezone, nonexistent="shift_forward")
    return timestamp.tz_convert(timezone)


def _request_json_with_retry(url: str, *, params: object | None = None, timeout: int = 20) -> dict:
    last_error: Exception | None = None
    attempts = max(settings.api_retry_attempts, 1)
    for attempt in range(attempts):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= attempts - 1:
                break
            sleep_for = settings.api_retry_backoff_seconds * (2**attempt) + random.uniform(0, 0.2)
            time.sleep(sleep_for)
    status = getattr(getattr(last_error, "response", None), "status_code", "unknown")
    raise DataFetchError(f"External API request failed with status {status}.") from last_error


def _normalize_load_response(raw: pd.Series | pd.DataFrame, timezone: str = DEFAULT_TIMEZONE) -> pd.DataFrame:
    if raw is None or len(raw) == 0:
        raise DataFetchError("No load data was found for the selected period.")

    if isinstance(raw, pd.DataFrame):
        if "Actual Load" in raw.columns:
            series = raw["Actual Load"]
        else:
            numeric = raw.select_dtypes(include="number")
            if numeric.empty:
                raise DataFetchError("ENTSO-E response did not contain numeric load values.")
            series = numeric.iloc[:, 0]
    else:
        series = raw

    frame = series.rename("load_mw").to_frame()
    frame.index = pd.to_datetime(frame.index)
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize(timezone)
    else:
        frame.index = frame.index.tz_convert(timezone)

    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame = frame.resample("h").mean()
    missing_before = int(frame["load_mw"].isna().sum())
    frame["load_mw"] = frame["load_mw"].interpolate(method="time", limit=6, limit_direction="both").ffill(limit=3).bfill(limit=3)
    remaining_missing = int(frame["load_mw"].isna().sum())
    frame = frame.dropna(subset=["load_mw"])
    frame["load_mw"] = frame["load_mw"].clip(lower=0)
    frame.index.name = "timestamp"
    frame.attrs["data_quality"] = {
        "rows_after_hourly_resample": int(len(frame)),
        "missing_before_fill": missing_before,
        "missing_after_fill": remaining_missing,
        "filled_values": max(missing_before - remaining_missing, 0),
        "warnings": ["Missing hourly values were interpolated/filled."] if missing_before else [],
    }
    return frame


def _with_source(frame: pd.DataFrame, source: str, detail: str = "") -> pd.DataFrame:
    existing_quality = frame.attrs.get("data_quality", {})
    frame.attrs["source"] = source
    frame.attrs["source_detail"] = detail
    frame.attrs["latest_timestamp"] = frame.index.max().isoformat() if not frame.empty else None
    if existing_quality:
        frame.attrs["data_quality"] = existing_quality
    return frame


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path, sep=None, engine="python")


def _first_matching_column(columns: list[str], required: tuple[str, ...], rejected: tuple[str, ...] = ()) -> str | None:
    for column in columns:
        normalized = column.lower()
        if all(token in normalized for token in required) and not any(token in normalized for token in rejected):
            return column
    return None


def normalize_entsoe_export_file(
    input_path: Path | str,
    output_path: Path | str = settings.de_lu_load_csv,
    *,
    timezone: str = COUNTRY_TIMEZONES["DE_LU"],
) -> pd.DataFrame:
    """Normalize manually exported ENTSO-E Total Load CSV/XLSX files.

    ENTSO-E exports vary by locale and selected columns. This parser accepts the
    common MTU/date-time interval column and chooses the Actual Total Load MW
    column while avoiding forecast columns.
    """

    input_file = Path(input_path)
    if not input_file.exists():
        raise DataFetchError(f"ENTSO-E export file not found: {input_file}")

    raw = _read_table(input_file)
    raw = raw.dropna(how="all")
    raw.columns = [str(column).strip() for column in raw.columns]
    columns = list(raw.columns)
    timestamp_column = (
        _first_matching_column(columns, ("mtu",))
        or _first_matching_column(columns, ("time",))
        or _first_matching_column(columns, ("date",))
        or _first_matching_column(columns, ("timestamp",))
    )
    load_column = (
        _first_matching_column(columns, ("actual", "load"), rejected=("forecast", "day-ahead", "ahead"))
        or _first_matching_column(columns, ("actual", "total"), rejected=("forecast", "day-ahead", "ahead"))
        or _first_matching_column(columns, ("load", "mw"), rejected=("forecast", "day-ahead", "ahead"))
    )
    if timestamp_column is None or load_column is None:
        raise DataFetchError(
            "Could not detect timestamp and Actual Total Load columns in ENTSO-E export. "
            f"Found columns: {', '.join(columns)}"
        )

    timestamp_text = raw[timestamp_column].astype(str).str.replace("\u2013", "-", regex=False)
    timestamp_text = timestamp_text.str.split(" - ", n=1).str[0].str.strip()
    timestamps = pd.to_datetime(timestamp_text, errors="coerce", dayfirst=True)
    load_values = (
        raw[load_column]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace(" ", "", regex=False)
        .str.extract(r"([-+]?\d*\.?\d+)", expand=False)
    )
    series = pd.Series(pd.to_numeric(load_values, errors="coerce").to_numpy(), index=timestamps)
    series = series[series.index.notna()].dropna()
    frame = _normalize_load_response(series, timezone=timezone)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    frame.reset_index().to_csv(output_file, index=False)
    return _with_source(frame, "entsoe_export", f"Normalized from {input_file}")


def load_normalized_country_csv(csv_path: Path, timezone: str, source: str, detail: str) -> pd.DataFrame:
    if not csv_path.exists():
        raise DataFetchError(f"Normalized load CSV was not found at {csv_path}.")
    raw = pd.read_csv(csv_path)
    required_columns = {"timestamp", "load_mw"}
    missing_columns = required_columns.difference(raw.columns)
    if missing_columns:
        raise DataFetchError(f"{csv_path} is missing required columns: {', '.join(sorted(missing_columns))}.")
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    raw["load_mw"] = pd.to_numeric(raw["load_mw"], errors="coerce")
    series = raw.dropna(subset=["timestamp", "load_mw"]).set_index("timestamp")["load_mw"]
    return _with_source(_normalize_load_response(series, timezone=timezone), source, detail)


def load_pakistan_csv_dataset() -> pd.DataFrame:
    csv_path = Path(os.getenv("PAKISTAN_LOAD_CSV", str(settings.pakistan_load_csv)))
    if not csv_path.exists():
        raise DataFetchError(
            "Pakistan is not available from ENTSO-E. Provide a CSV at "
            f"`{csv_path}` or set PAKISTAN_LOAD_CSV. Required columns: timestamp, load_mw."
        )

    try:
        return load_normalized_country_csv(csv_path, COUNTRY_TIMEZONES["PK"], "pakistan_csv_demo", "Historical NTDC/Kaggle CSV")
    except DataFetchError:
        raise
    except Exception as exc:
        raise DataFetchError(f"Failed to read Pakistan load CSV: {exc}") from exc


def _load_pakistan_csv(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frame = load_pakistan_csv_dataset()
    filtered = frame.loc[(frame.index >= start) & (frame.index < end)]
    if filtered.empty:
        raise DataFetchError("Pakistan load CSV has no rows in the selected date range.")
    _with_source(filtered, frame.attrs.get("source", "pakistan_csv_demo"), frame.attrs.get("source_detail", ""))
    return filtered


def load_de_lu_export_dataset() -> pd.DataFrame:
    return load_normalized_country_csv(
        settings.de_lu_load_csv,
        COUNTRY_TIMEZONES["DE_LU"],
        "entsoe_export",
        "Normalized manual ENTSO-E Transparency Platform export",
    )


def _load_de_lu_export(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frame = load_de_lu_export_dataset()
    filtered = frame.loc[(frame.index >= start) & (frame.index < end)]
    if filtered.empty:
        raise DataFetchError("DE_LU ENTSO-E export CSV has no rows in the selected date range.")
    _with_source(filtered, frame.attrs.get("source", "entsoe_export"), frame.attrs.get("source_detail", ""))
    return filtered


def _smard_json(path: str) -> dict:
    url = f"{SMARD_BASE_URL}/{path.lstrip('/')}"
    return _request_json_with_retry(url, timeout=20)


def fetch_smard_de_lu_load(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    start_local = _to_timestamp(start, COUNTRY_TIMEZONES["DE_LU"])
    end_local = _to_timestamp(end, COUNTRY_TIMEZONES["DE_LU"])
    index_payload = _smard_json(f"{SMARD_TOTAL_LOAD_FILTER}/{SMARD_REGION}/index_hour.json")
    timestamps = sorted(int(value) for value in index_payload.get("timestamps", []))
    if not timestamps:
        raise DataFetchError("SMARD did not return any available timestamps for DE-LU load.")

    start_ms = int((start_local - pd.Timedelta(days=8)).tz_convert("UTC").timestamp() * 1000)
    end_ms = int(end_local.tz_convert("UTC").timestamp() * 1000)
    selected_timestamps = [value for value in timestamps if start_ms <= value <= end_ms]
    if not selected_timestamps:
        older = [value for value in timestamps if value <= end_ms]
        if older:
            selected_timestamps = [max(older)]
        else:
            raise DataFetchError("SMARD has no data available before the requested end date.")

    rows: list[tuple[pd.Timestamp, float]] = []
    for timestamp in selected_timestamps:
        payload = _smard_json(
            f"{SMARD_TOTAL_LOAD_FILTER}/{SMARD_REGION}/"
            f"{SMARD_TOTAL_LOAD_FILTER}_{SMARD_REGION}_hour_{timestamp}.json"
        )
        for point in payload.get("series", []):
            if not isinstance(point, list) or len(point) < 2 or point[1] is None:
                continue
            ts = pd.to_datetime(int(point[0]), unit="ms", utc=True).tz_convert(COUNTRY_TIMEZONES["DE_LU"])
            rows.append((ts, float(point[1])))

    if not rows:
        raise DataFetchError("SMARD returned no load values for DE-LU.")
    series = pd.Series([value for _, value in rows], index=[timestamp for timestamp, _ in rows])
    frame = _normalize_load_response(series, timezone=COUNTRY_TIMEZONES["DE_LU"])
    filtered = frame.loc[(frame.index >= start_local) & (frame.index < end_local)]
    if filtered.empty:
        raise DataFetchError("SMARD has no rows in the selected DE_LU date range.")
    return _with_source(
        filtered,
        "smard_dynamic_fallback",
        "SMARD real Germany/Luxembourg hourly load from Bundesnetzagentur, filter 410, region DE-LU",
    )


def fetch_eia_us48_load(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if not settings.eia_api_key:
        raise DataFetchError("Missing EIA_API_KEY environment variable.")
    start_utc = _to_timestamp(start, COUNTRY_TIMEZONES["US48"]).tz_convert("UTC")
    end_utc = _to_timestamp(end, COUNTRY_TIMEZONES["US48"]).tz_convert("UTC")
    base_params = [
        ("api_key", settings.eia_api_key),
        ("frequency", "hourly"),
        ("data[0]", "value"),
        ("facets[respondent][]", "US48"),
        ("facets[type][]", "D"),
        ("start", start_utc.strftime("%Y-%m-%dT%H")),
        ("end", end_utc.strftime("%Y-%m-%dT%H")),
        ("sort[0][column]", "period"),
        ("sort[0][direction]", "asc"),
    ]
    rows: list[dict[str, object]] = []
    page_size = 5000
    offset = 0
    total: int | None = None
    for _ in range(20):
        params = base_params + [("length", str(page_size)), ("offset", str(offset))]
        payload = _request_json_with_retry(EIA_RTO_REGION_ENDPOINT, params=params, timeout=20)
        if payload.get("error"):
            raise DataFetchError(f"EIA API error: {payload['error'].get('message', payload['error'])}")
        response_body = payload.get("response", {})
        page_rows = response_body.get("data", [])
        total = int(response_body.get("total", 0) or 0)
        rows.extend(page_rows)
        if not page_rows or len(page_rows) < page_size or (total and len(rows) >= total):
            break
        offset += page_size
    if not rows:
        raise DataFetchError("EIA returned no hourly demand rows for US48.")
    timestamps = pd.to_datetime([row.get("period") for row in rows], errors="coerce", utc=True)
    values = pd.to_numeric([row.get("value") for row in rows], errors="coerce")
    series = pd.Series(values, index=timestamps).dropna()
    frame = _normalize_load_response(series, timezone=COUNTRY_TIMEZONES["US48"])
    return _with_source(frame, "eia_live", "EIA hourly RTO demand, respondent US48, type D")


def _get_cached(cache_key: tuple[str, str, str]) -> pd.DataFrame | None:
    with _FETCH_CACHE_LOCK:
        cached = _FETCH_CACHE.get(cache_key)
        if cached is None:
            return None
        created_at, frame = cached
        if time.time() - created_at > settings.data_cache_ttl_seconds:
            _FETCH_CACHE.pop(cache_key, None)
            return None
        return frame.copy()


def _set_cached(cache_key: tuple[str, str, str], frame: pd.DataFrame) -> None:
    with _FETCH_CACHE_LOCK:
        if len(_FETCH_CACHE) >= _FETCH_CACHE_MAX:
            oldest_key = min(_FETCH_CACHE, key=lambda k: _FETCH_CACHE[k][0])
            _FETCH_CACHE.pop(oldest_key, None)
        _FETCH_CACHE[cache_key] = (time.time(), frame.copy())


def _validate_country(country_code: str) -> str:
    normalized_country = country_code.upper()
    if normalized_country not in SUPPORTED_COUNTRIES:
        supported = ", ".join(SUPPORTED_COUNTRIES)
        raise ValueError(f"Unsupported country_code `{country_code}`. Supported values: {supported}.")
    return normalized_country


def public_supported_countries(include_demo: bool = False) -> dict[str, dict[str, object]]:
    if include_demo:
        return SUPPORTED_COUNTRIES
    return {
        code: metadata
        for code, metadata in SUPPORTED_COUNTRIES.items()
        if metadata.get("data_class") != "historical_demo"
    }


def get_available_data_range(country_code: str) -> dict[str, object]:
    normalized_country = _validate_country(country_code)
    if normalized_country == "PK":
        frame = load_pakistan_csv_dataset()
        return {
            "country_code": normalized_country,
            "source": frame.attrs.get("source", "pakistan_csv_demo"),
            "source_detail": frame.attrs.get("source_detail", ""),
            "start": frame.index.min().isoformat(),
            "end": frame.index.max().isoformat(),
            "rows": int(len(frame)),
        }
    if normalized_country == "DE_LU" and settings.de_lu_load_csv.exists():
        frame = load_de_lu_export_dataset()
        return {
            "country_code": normalized_country,
            "source": frame.attrs.get("source", "entsoe_export"),
            "source_detail": frame.attrs.get("source_detail", ""),
            "start": frame.index.min().isoformat(),
            "end": frame.index.max().isoformat(),
            "rows": int(len(frame)),
        }

    return {
        "country_code": normalized_country,
        "source": SUPPORTED_COUNTRIES[normalized_country]["source"],
        "source_detail": "Live range is discovered when fetching the selected period.",
        "start": None,
        "end": None,
        "rows": None,
    }


def fetch_load_data(
    country_code: str,
    start: str | datetime | pd.Timestamp,
    end: str | datetime | pd.Timestamp,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch hourly ActualTotalLoad data from ENTSO-E.

    Args:
        country_code: ENTSO-E bidding zone code, for example ``DE_LU``.
        start: Inclusive start timestamp or date.
        end: Exclusive end timestamp or date.

    Returns:
        DataFrame indexed by timezone-aware hourly timestamps with a ``load_mw`` column.
    """

    normalized_country = _validate_country(country_code)
    timezone = COUNTRY_TIMEZONES.get(normalized_country, DEFAULT_TIMEZONE)
    start_ts = _to_timestamp(start, timezone=timezone)
    end_ts = _to_timestamp(end, timezone=timezone)
    if end_ts <= start_ts:
        raise ValueError("end must be after start.")
    if (end_ts - start_ts).days > settings.max_history_days:
        raise ValueError(f"Date range cannot exceed {settings.max_history_days} days.")

    cache_key = (normalized_country, start_ts.isoformat(), end_ts.isoformat())
    if use_cache:
        cached = _get_cached(cache_key)
        if cached is not None:
            return cached

    if normalized_country == "PK":
        frame = _load_pakistan_csv(start_ts, end_ts)
        _set_cached(cache_key, frame)
        return frame

    if normalized_country == "US48":
        frame = fetch_eia_us48_load(start_ts, end_ts)
        _set_cached(cache_key, frame)
        return frame

    api_key = settings.entsoe_api_key
    source_errors: list[str] = []

    if api_key:
        last_error: Exception | None = None
        for attempt in range(max(settings.api_retry_attempts, 1)):
            try:
                client = EntsoePandasClient(api_key=api_key)
                raw = client.query_load(country_code=normalized_country, start=start_ts, end=end_ts)
                frame = _with_source(_normalize_load_response(raw, timezone=timezone), "entsoe_live", "ENTSO-E Transparency API")
                _set_cached(cache_key, frame)
                return frame
            except ValueError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt < max(settings.api_retry_attempts, 1) - 1:
                    time.sleep(settings.api_retry_backoff_seconds * (2**attempt) + random.uniform(0, 0.2))
        source_errors.append(f"ENTSO-E API failed: {last_error}")
    else:
        source_errors.append("ENTSOE_API_KEY is not configured")

    if normalized_country == "DE_LU":
        try:
            frame = _load_de_lu_export(start_ts, end_ts)
            _set_cached(cache_key, frame)
            return frame
        except Exception as exc:
            source_errors.append(f"ENTSO-E export fallback failed: {exc}")
        try:
            frame = fetch_smard_de_lu_load(start_ts, end_ts)
            _set_cached(cache_key, frame)
            return frame
        except Exception as exc:
            source_errors.append(f"SMARD fallback failed: {exc}")

    if not api_key and normalized_country != "DE_LU":
        raise DataFetchError("Missing ENTSOE_API_KEY environment variable.")
    raise DataFetchError("; ".join(source_errors))


def fetch_load_data_with_metadata(
    country_code: str,
    start: str | datetime | pd.Timestamp,
    end: str | datetime | pd.Timestamp,
    *,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = fetch_load_data(country_code, start, end, use_cache=use_cache)
    metadata = {
        "source": frame.attrs.get("source", "unknown"),
        "source_detail": frame.attrs.get("source_detail", ""),
        "latest_timestamp": frame.attrs.get("latest_timestamp") or (frame.index.max().isoformat() if not frame.empty else None),
        "rows": int(len(frame)),
    }
    return frame, metadata
