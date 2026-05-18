from __future__ import annotations

import json
import hashlib
import logging
import time
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.config import settings


logger = logging.getLogger(__name__)

LLM_MIN_RESPONSE_CHARS = 180
LLM_RESPONSE_CACHE: dict[str, str] = {}
LLM_DISABLED_UNTIL = 0.0
LLM_LAST_ERROR = ""


SOURCE_LABELS = {
    "entsoe_live": "ENTSO-E live API",
    "entsoe_export": "manual ENTSO-E export",
    "smard_dynamic_fallback": "SMARD live DE-LU",
    "eia_live": "EIA live API",
    "pakistan_csv_demo": "Pakistan historical CSV",
}


METHOD_LABELS = {
    "lstm": "trained LSTM neural network",
    "seasonal_naive_fallback": "seasonal baseline fallback",
}


def _direction(first: float, last: float) -> str:
    change_pct = ((last - first) / first * 100) if first else 0.0
    if change_pct > 3:
        return f"rising by about {change_pct:.1f}%"
    if change_pct < -3:
        return f"falling by about {abs(change_pct):.1f}%"
    return "mostly stable"


def _direction_action(direction: str) -> str:
    if "rising" in direction:
        return "Demand is expected to increase, so keep enough supply margin for the peak period."
    if "falling" in direction:
        return "Demand is expected to soften, so avoid unnecessary over-commitment."
    return "Demand is expected to stay fairly steady, so focus on the forecast peak hour."


def _format_load(value_mw: float) -> str:
    value_mw = float(value_mw)
    if abs(value_mw) >= 10_000:
        return f"{value_mw / 1000:,.1f} GW ({value_mw:,.0f} MW)"
    return f"{value_mw:,.0f} MW"


def _format_time(value: str) -> str:
    try:
        timestamp = pd.Timestamp(value)
        suffix = " UTC" if timestamp.tzinfo is not None and timestamp.utcoffset() == pd.Timedelta(0) else ""
        return timestamp.strftime("%b %d, %Y %H:%M") + suffix
    except Exception:
        return str(value)


def _format_window(start: Any, end: Any) -> str:
    if not start or not end:
        return "not reported"
    return f"{_format_time(str(start))} to {_format_time(str(end))}"


def _source_sentence(source: str, detail: str) -> str:
    label = SOURCE_LABELS.get(source, source or "unknown source")
    if source == "smard_dynamic_fallback":
        return f"{label}: real Germany/Luxembourg hourly load data from Bundesnetzagentur SMARD."
    if source == "entsoe_export":
        return f"{label}: real ENTSO-E assignment data imported from a Transparency Platform export."
    if source == "entsoe_live":
        return f"{label}: real live assignment-compliant ENTSO-E data."
    if source == "eia_live":
        return f"{label}: real live U.S. hourly electricity demand data."
    if source == "pakistan_csv_demo":
        return "Pakistan historical CSV: demo-only data, not a live public API feed."
    return f"{label}{f': {detail}' if detail else ''}."


def _freshness_sentence(freshness: dict[str, Any]) -> str:
    if not freshness:
        return " freshness unknown"
    staleness = freshness.get("staleness", "unknown")
    hours_behind = freshness.get("hours_behind_now")
    labels = {"fresh": "Fresh (< 3h old)", "stale": "Moderately stale", "very_stale": "Very stale (> 24h old)", "unknown": "Staleness unknown"}
    label = labels.get(staleness, "Staleness unknown")
    if isinstance(hours_behind, (int, float)):
        return f"{label} ({hours_behind:.0f}h behind now)."
    return f"{label}."


def _backtest_sentence(accuracy: dict[str, Any]) -> str:
    if not accuracy or not accuracy.get("available"):
        return accuracy.get("reason", "not enough data for a recent backtest") if accuracy else "not available"
    mape = accuracy.get("mape_pct")
    mae = accuracy.get("mae_mw")
    parts = []
    if isinstance(mape, (int, float)):
        parts.append(f"MAPE {mape:.1f}%")
    if isinstance(mae, (int, float)):
        parts.append(f"MAE {mae:.0f} MW")
    return f"Last-24h seasonal naive backtest: {', '.join(parts) if parts else 'computed'}."


def _confidence_label(mape: Any) -> str:
    if not isinstance(mape, (int, float)):
        return "Unknown"
    if mape <= 5:
        return "High"
    if mape <= 10:
        return "Good"
    if mape <= 15:
        return "Moderate"
    return "Low"


def _risk_note(source: str, method: str) -> str:
    if source == "pakistan_csv_demo":
        return "This is historical demo data, so do not treat it as a live Pakistan grid forecast."
    if method != "lstm":
        return "This is a baseline forecast because a matching LSTM model is not available yet."
    if source == "eia_live":
        return "This is real live U.S. demand data, but it is not the ENTSO-E assignment source."
    if source == "smard_dynamic_fallback":
        return "This is real Germany/Luxembourg hourly load data from SMARD."
    return "This forecast uses the configured real data source for the selected country."


def _operational_risk_level(confidence: str, volatility_pct: float, trend: str) -> tuple[str, str]:
    if confidence in {"Unknown", "Low"}:
        return "Elevated", "Model accuracy is limited, so use wider manual reserves and rerun with fresh data."
    if volatility_pct >= 20:
        return "Elevated", "Forecast swings are large compared with average load, so plan around the upper band."
    if volatility_pct >= 10 or "rising" in trend or "falling" in trend:
        return "Watch", "Conditions are manageable, but the direction is clear enough to monitor peak-hour reserves."
    return "Normal", "Load is comparatively steady, so standard reserve planning is reasonable."


def _to_plain_number(value: Any) -> float | int | str | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric.is_integer():
        return int(numeric)
    return round(numeric, 3)


def _build_forecast_context(payload: dict[str, Any]) -> dict[str, Any]:
    forecast = payload.get("forecast", [])
    values = np.asarray([point["predicted_load_mw"] for point in forecast], dtype=float) if forecast else np.asarray([])
    lower_values = np.asarray([point.get("lower_bound_mw", point["predicted_load_mw"]) for point in forecast], dtype=float) if forecast else np.asarray([])
    upper_values = np.asarray([point.get("upper_bound_mw", point["predicted_load_mw"]) for point in forecast], dtype=float) if forecast else np.asarray([])
    baseline_values = np.asarray([point.get("baseline_load_mw", point["predicted_load_mw"]) for point in forecast], dtype=float) if forecast else np.asarray([])
    peak_idx = int(np.argmax(values)) if len(values) else None
    trough_idx = int(np.argmin(values)) if len(values) else None
    first_value = float(values[0]) if len(values) else None
    last_value = float(values[-1]) if len(values) else None
    trend_pct = ((last_value - first_value) / first_value * 100) if first_value else None

    def point_at(index: int | None) -> dict[str, Any] | None:
        if index is None or not forecast:
            return None
        point = forecast[index]
        return {
            "timestamp": point.get("timestamp"),
            "predicted_load_mw": _to_plain_number(point.get("predicted_load_mw")),
            "lower_bound_mw": _to_plain_number(point.get("lower_bound_mw")),
            "upper_bound_mw": _to_plain_number(point.get("upper_bound_mw")),
            "baseline_load_mw": _to_plain_number(point.get("baseline_load_mw")),
        }

    metadata = payload.get("model_metadata", {}) if isinstance(payload.get("model_metadata"), dict) else {}
    metrics = metadata.get("metrics", {}) if isinstance(metadata.get("metrics"), dict) else {}
    baseline_metrics = metadata.get("baseline_metrics", {}) if isinstance(metadata.get("baseline_metrics"), dict) else {}
    history_window = payload.get("history_window", {}) if isinstance(payload.get("history_window"), dict) else {}
    source = payload.get("data_source", {}) if isinstance(payload.get("data_source"), dict) else {}

    return {
        "country_code": payload.get("country_code"),
        "horizon_hours": payload.get("horizon") or len(forecast),
        "forecast_method": payload.get("forecast_method"),
        "data_source": {
            "source": source.get("source"),
            "source_detail": source.get("source_detail"),
            "latest_timestamp": source.get("latest_timestamp"),
            "rows": source.get("rows"),
        },
        "history_window": {
            "selected_start": history_window.get("selected_start"),
            "selected_end": history_window.get("selected_end"),
            "selected_end_inclusive": history_window.get("selected_end_inclusive"),
            "actual_start": history_window.get("actual_start"),
            "actual_end": history_window.get("actual_end"),
            "history_rows": history_window.get("history_rows"),
            "model_input_start": history_window.get("model_input_start"),
            "model_input_end": history_window.get("model_input_end"),
            "model_input_rows": history_window.get("model_input_rows"),
            "history_capped": history_window.get("history_capped"),
            "max_runtime_history_days": history_window.get("max_runtime_history_days"),
            "forecast_start": history_window.get("forecast_start"),
            "forecast_end": history_window.get("forecast_end"),
        },
        "computed_forecast_stats": {
            "trend_percent": _to_plain_number(trend_pct),
            "average_load_mw": _to_plain_number(float(np.mean(values)) if len(values) else None),
            "minimum": point_at(trough_idx),
            "maximum": point_at(peak_idx),
            "average_uncertainty_half_width_mw": _to_plain_number(float(np.mean(upper_values - lower_values) / 2) if len(values) else None),
            "average_lstm_minus_baseline_mw": _to_plain_number(float(np.mean(values - baseline_values)) if len(values) else None),
        },
        "forecast_summary": payload.get("forecast_summary", {}),
        "hourly_profile": payload.get("hourly_profile", {}),
        "data_freshness": payload.get("data_freshness", {}),
        "recent_accuracy": payload.get("recent_accuracy", {}),
        "model_quality": {
            "training_rows": metadata.get("rows"),
            "trained_at": metadata.get("trained_at"),
            "model_country_code": metadata.get("country_code"),
            "data_source": metadata.get("data_source"),
            "metrics": {
                "mae_mw": metrics.get("mae"),
                "rmse_mw": metrics.get("rmse"),
                "mape_percent": metrics.get("mape"),
            },
            "seasonal_baseline_metrics": {
                "mae_mw": baseline_metrics.get("mae"),
                "rmse_mw": baseline_metrics.get("rmse"),
                "mape_percent": baseline_metrics.get("mape"),
            },
        },
        "warnings": payload.get("warnings", []),
    }


def _llm_cache_key(context: dict[str, Any]) -> str:
    raw = json.dumps(context, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_deterministic_explanation(payload: dict[str, Any]) -> str:
    forecast = payload.get("forecast", [])
    if not forecast:
        return "No forecast points were available to explain."

    values = np.asarray([point["predicted_load_mw"] for point in forecast], dtype=float)
    lower_values = np.asarray([point.get("lower_bound_mw", point["predicted_load_mw"]) for point in forecast], dtype=float)
    upper_values = np.asarray([point.get("upper_bound_mw", point["predicted_load_mw"]) for point in forecast], dtype=float)
    forecast_summary = payload.get("forecast_summary", {}) if isinstance(payload.get("forecast_summary"), dict) else {}
    data_freshness = payload.get("data_freshness", {}) if isinstance(payload.get("data_freshness"), dict) else {}
    recent_accuracy = payload.get("recent_accuracy", {}) if isinstance(payload.get("recent_accuracy"), dict) else {}
    hourly_profile = payload.get("hourly_profile", {}) if isinstance(payload.get("hourly_profile"), dict) else {}
    method = payload.get("forecast_method", "unknown")
    source_payload = payload.get("data_source", {})
    source = source_payload.get("source", "unknown") if isinstance(source_payload, dict) else "unknown"
    source_detail = source_payload.get("source_detail", "") if isinstance(source_payload, dict) else ""
    metadata = payload.get("model_metadata", {})
    history_window = payload.get("history_window", {})
    metrics = metadata.get("metrics", {}) if isinstance(metadata, dict) else {}
    baseline_values = np.asarray([point.get("baseline_load_mw", point["predicted_load_mw"]) for point in forecast], dtype=float)
    interval_width = float(np.mean(upper_values - lower_values)) if len(values) else 0.0
    peak_idx = int(np.argmax(values))
    trough_idx = int(np.argmin(values))
    average_load = float(np.mean(values))
    baseline_gap = float(np.mean(values - baseline_values)) if len(baseline_values) else 0.0

    mape = metrics.get("mape") if isinstance(metrics, dict) else None
    confidence = _confidence_label(mape)

    method_label = METHOD_LABELS.get(str(method), str(method))
    trend = _direction(float(values[0]), float(values[-1]))
    baseline_sentence = (
        f"The LSTM forecast is about {_format_load(abs(baseline_gap))} {'higher' if baseline_gap >= 0 else 'lower'} than the seasonal baseline on average."
        if abs(baseline_gap) >= 1
        else "The LSTM forecast is very close to the seasonal baseline on average."
    )
    mape_sentence = (
        f"Recent test MAPE is {mape:.2f}%, meaning the model's average test error was about {mape:.0f}% of actual load."
        if isinstance(mape, (int, float))
        else "No saved test MAPE is available for this country yet."
    )
    action = _direction_action(trend)
    data_rows = source_payload.get("rows") if isinstance(source_payload, dict) else None
    latest_row = source_payload.get("latest_timestamp") if isinstance(source_payload, dict) else None
    requested_start = history_window.get("selected_start") if isinstance(history_window, dict) else None
    requested_end = history_window.get("selected_end") if isinstance(history_window, dict) else None
    end_word = "inclusive" if isinstance(history_window, dict) and history_window.get("selected_end_inclusive") else "exclusive"
    actual_start = history_window.get("actual_start") if isinstance(history_window, dict) else None
    actual_end = history_window.get("actual_end") if isinstance(history_window, dict) else None
    model_input_start = history_window.get("model_input_start") if isinstance(history_window, dict) else None
    model_input_end = history_window.get("model_input_end") if isinstance(history_window, dict) else None
    model_input_rows = history_window.get("model_input_rows") if isinstance(history_window, dict) else None
    forecast_start = history_window.get("forecast_start") if isinstance(history_window, dict) else None
    forecast_end = history_window.get("forecast_end") if isinstance(history_window, dict) else None
    trained_rows = metadata.get("rows") if isinstance(metadata, dict) else None
    baseline_mape = metadata.get("baseline_metrics", {}).get("mape") if isinstance(metadata, dict) and isinstance(metadata.get("baseline_metrics"), dict) else None
    model_better = ""
    if isinstance(mape, (int, float)) and isinstance(baseline_mape, (int, float)):
        improvement = max(0.0, baseline_mape - mape)
        model_better = f" The LSTM beats the seasonal baseline by about {improvement:.1f} percentage points of MAPE."
    model_input_note = "same as the graph history"
    if isinstance(history_window, dict) and history_window.get("history_capped"):
        model_input_note = (
            f"{_format_window(model_input_start, model_input_end)} "
            f"({model_input_rows if model_input_rows is not None else 'not reported'} recent hourly rows)"
        )
    risk_note = _risk_note(str(source), str(method))
    peak_time = _format_time(forecast[peak_idx]["timestamp"])
    trough_time = _format_time(forecast[trough_idx]["timestamp"])

    std_dev = float(np.std(values)) if len(values) > 1 else 0.0
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    range_val = max_val - min_val
    volatility_pct = (std_dev / average_load * 100) if average_load > 0 else 0.0
    stability = "stable" if volatility_pct < 10 else "moderate" if volatility_pct < 20 else "variable"

    hour_values = {}
    for point in forecast:
        ts = point.get("timestamp", "")
        if "T" in ts:
            hour = int(ts.split("T")[1].split(":")[0]) if len(ts.split("T")) > 1 else 0
            if hour not in hour_values:
                hour_values[hour] = []
            hour_values[hour].append(point["predicted_load_mw"])
    if hourly_profile:
        night_avg = hourly_profile.get("night_00_05", 0)
        morning_avg = hourly_profile.get("morning_06_11", 0)
        afternoon_avg = hourly_profile.get("afternoon_12_17", 0)
        evening_avg = hourly_profile.get("evening_18_23", 0)
    else:
        night_avg = np.mean([v for h, vs in hour_values.items() if h < 6 for v in vs]) if any(h < 6 for h in hour_values) else 0
        morning_avg = np.mean([v for h, vs in hour_values.items() if 6 <= h < 12 for v in vs]) if any(6 <= h < 12 for h in hour_values) else 0
        afternoon_avg = np.mean([v for h, vs in hour_values.items() if 12 <= h < 18 for v in vs]) if any(12 <= h < 18 for h in hour_values) else 0
        evening_avg = np.mean([v for h, vs in hour_values.items() if h >= 18 for v in vs]) if any(h >= 18 for h in hour_values) else 0

    peak_to_avg_ratio = values[peak_idx] / average_load if average_load > 0 else 1.0
    peak_value = float(values[peak_idx])
    peak_upper = float(upper_values[peak_idx]) if len(upper_values) > peak_idx else peak_value
    peak_lower = float(lower_values[peak_idx]) if len(lower_values) > peak_idx else peak_value
    reserve_pct = 0.05
    capacity_target = max(peak_upper, peak_value * (1 + reserve_pct))
    reserve_above_peak = max(capacity_target - peak_value, 0.0)
    planning_min = float(np.min(lower_values)) if len(lower_values) else min_val
    planning_max = float(np.max(upper_values)) if len(upper_values) else max_val
    risk_level, risk_reason = _operational_risk_level(confidence, volatility_pct, trend)
    compliance_note = _risk_note(str(source), str(method))
    reserve_margin_note = (
        f"Peak is {peak_to_avg_ratio:.0%} of average load. "
        f"For a simple 5% operating reserve, target about {_format_load(capacity_target)} total available capacity; "
        f"that is roughly {_format_load(reserve_above_peak)} above the expected peak."
        if peak_to_avg_ratio > 1.1 else
        "Load is relatively flat. Standard reserve margins apply, but still check the upper confidence band."
    )
    peak_band_note = (
        f"At the peak hour, the model's band is {_format_load(peak_lower)} to {_format_load(peak_upper)}. "
        "Use the upper side for reserve planning."
    )

    return "\n".join(
        [
            "Executive forecast brief",
            f"Decision signal: demand is {trend} over the next {len(values)} hours.",
            f"Expected average load: {_format_load(average_load)}.",
            f"Peak watch: {_format_load(peak_value)} around {peak_time}.",
            f"Low-load period: {_format_load(values[trough_idx])} around {trough_time}.",
            f"Operational risk: {risk_level}. {risk_reason}",
            "",
            "Operator guidance",
            f"Recommended action: {action}",
            reserve_margin_note,
            peak_band_note,
            f"Average uncertainty band: +/- {_format_load(interval_width / 2)} around the forecast line.",
            f"Full safe planning envelope: {_format_load(planning_min)} to {_format_load(planning_max)}. "
            f"The center forecast itself ranges from {_format_load(min_val)} to {_format_load(max_val)}.",
            "",
            "Why the forecast looks this way",
            f"The load profile is {stability}: standard deviation is {_format_load(std_dev)}, about {volatility_pct:.1f}% of average load.",
            baseline_sentence,
            "If actual demand starts moving outside the shaded band, rerun the forecast and treat the old result as stale.",
            "",
            "Time-of-day pattern",
            f"Night 00-05: {_format_load(night_avg) if night_avg else 'N/A'}.",
            f"Morning 06-11: {_format_load(morning_avg) if morning_avg else 'N/A'}.",
            f"Afternoon 12-17: {_format_load(afternoon_avg) if afternoon_avg else 'N/A'}.",
            f"Evening 18-23: {_format_load(evening_avg) if evening_avg else 'N/A'}.",
            "",
            "How to read the forecast",
            "Blue line: expected load from the model.",
            "Dotted baseline: yesterday/seasonal pattern used as a simple benchmark.",
            "Shaded band: uncertainty range. For operations, the upper band near the peak matters more than the average line.",
            "",
            "Data used",
            f"Real data source: {_source_sentence(str(source), str(source_detail))}",
            f"Data freshness: {_freshness_sentence(data_freshness)}",
            f"Source caveat: {compliance_note}",
            f"Selected history window: {_format_window(requested_start, requested_end)} ({end_word} end).",
            f"Graph history shown: {_format_window(actual_start, actual_end)} using {data_rows if data_rows is not None else 'not reported'} hourly rows.",
            f"LSTM input window: {model_input_note}.",
            f"Forecast period: {_format_window(forecast_start, forecast_end)}.",
            f"Latest source timestamp: {_format_time(str(latest_row)) if latest_row else 'not reported'}.",
            "",
            "Model confidence",
            f"Forecast engine: {method_label}.",
            f"Training data size: {trained_rows if trained_rows is not None else 'not reported'} hourly rows.",
            f"Accuracy check: {mape_sentence}{model_better}",
            f"Recent backtest: {_backtest_sentence(recent_accuracy)}",
            f"Confidence level: {confidence} for normal operating days. Confidence drops during unusual weather, holidays, outages, or reporting delays.",
            "Uncertainty growth: the confidence band uses residual error with mild horizon widening, so early hours are still the most reliable.",
            "",
            "Next action checklist",
            "1. Schedule enough generation/import capacity for the peak hour and upper band.",
            "2. Recheck the forecast after the next actual hourly data update.",
            "3. If weather, outages, holidays, or abnormal grid events are expected, add a manual operator adjustment.",
        ]
    )


def build_llm_explanation(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    global LLM_DISABLED_UNTIL, LLM_LAST_ERROR

    if not settings.opencode_api_key:
        return None, "OPENCODE_API_KEY is not configured."

    now = time.monotonic()
    if now < LLM_DISABLED_UNTIL:
        remaining = int(LLM_DISABLED_UNTIL - now)
        return None, f"External AI is cooling down for {remaining}s after the last provider issue: {LLM_LAST_ERROR}"

    forecast_context = _build_forecast_context(payload)
    cache_key = _llm_cache_key(forecast_context)
    if cache_key in LLM_RESPONSE_CACHE:
        return LLM_RESPONSE_CACHE[cache_key], None

    prompt = (
        "Write a concise electricity-load forecast explanation for an operations dashboard.\n\n"
        "Rules:\n"
        "- Use only the JSON values. Do not invent weather, outages, prices, or events.\n"
        "- Keep it simple and practical for a manager/operator.\n"
        "- Use these exact headings: Executive forecast brief, Operator guidance, Data used, Model confidence, Next action checklist.\n"
        "- Clearly separate graph history from LSTM input when history_capped is true.\n"
        "- Mention peak, low, average load, uncertainty band, model error, and baseline comparison.\n"
        "- Be clear that reserve margin is extra capacity above expected peak, not the total capacity target.\n"
        "- Keep the whole answer under 260 words.\n"
        "- Return the final dashboard explanation only. Do not include hidden reasoning or analysis notes.\n\n"
        f"Forecast JSON:\n{json.dumps(forecast_context, separators=(',', ':'), default=str)}"
    )
    url = f"{settings.opencode_base_url.rstrip('/')}/chat/completions"
    request_body = {
        "model": settings.opencode_model,
        "messages": [
            {
                "role": "system",
                "content": "You explain electricity demand forecasts to power-system operators. Return only the final user-facing explanation.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.15,
        "max_tokens": settings.ai_explanation_max_tokens,
    }
    last_error = "Unknown AI provider error."
    max_attempts = max(1, settings.ai_explanation_retries + 1)
    for attempt in range(max_attempts):
        try:
            response = requests.post(
                url,
                headers={"Authorization": f"Bearer {settings.opencode_api_key}", "Content-Type": "application/json"},
                json=request_body,
                timeout=settings.ai_explanation_timeout_seconds,
            )
            provider_error = ""
            if response.status_code >= 400:
                try:
                    error_payload = response.json().get("error", {})
                    provider_error = str(error_payload.get("message") or error_payload.get("type") or response.text)
                    provider_error_type = str(error_payload.get("type") or "")
                except ValueError:
                    provider_error = response.text
                    provider_error_type = ""
                if response.status_code == 429 or provider_error_type == "FreeUsageLimitError":
                    last_error = (
                        "AI provider free usage limit reached. Using the built-in explanation for this run; "
                        "try again after the provider quota resets."
                    )
                    LLM_LAST_ERROR = "free usage limit reached"
                    LLM_DISABLED_UNTIL = time.monotonic() + settings.ai_explanation_cooldown_seconds
                    logger.warning("Optional OpenCode explanation rate-limited on attempt %s: %s", attempt + 1, provider_error)
                    if attempt < max_attempts - 1:
                        continue
                    return None, last_error
                if provider_error_type == "ModelError" or "not supported" in provider_error.lower():
                    last_error = (
                        f"AI model `{settings.opencode_model}` is not supported by this provider account. "
                        "Using the built-in explanation for this run."
                    )
                    LLM_LAST_ERROR = f"model `{settings.opencode_model}` is not supported"
                    LLM_DISABLED_UNTIL = time.monotonic() + settings.ai_explanation_cooldown_seconds
                    logger.warning("Optional OpenCode explanation model error: %s", provider_error)
                    return None, last_error
            if response.status_code == 429:
                last_error = "AI provider rate limit reached. Using the built-in explanation for this run; try again after the provider quota resets."
                LLM_LAST_ERROR = "rate limit reached"
                LLM_DISABLED_UNTIL = time.monotonic() + settings.ai_explanation_cooldown_seconds
                logger.warning("Optional OpenCode explanation rate-limited on attempt %s.", attempt + 1)
                if attempt < max_attempts - 1:
                    continue
                return None, last_error
            response.raise_for_status()
            data = response.json()
            message = data["choices"][0]["message"]
            content = str(message.get("content") or "").strip()
            if len(content) >= LLM_MIN_RESPONSE_CHARS and "Forecast brief" in content:
                LLM_RESPONSE_CACHE[cache_key] = content
                return content, None
            reasoning_content = str(message.get("reasoning_content") or "").strip()
            if len(reasoning_content) >= LLM_MIN_RESPONSE_CHARS and "Forecast brief" in reasoning_content:
                LLM_RESPONSE_CACHE[cache_key] = reasoning_content
                return reasoning_content, None
            last_error = "AI provider returned a too-short or malformed explanation."
            LLM_LAST_ERROR = "too-short or malformed response"
            LLM_DISABLED_UNTIL = time.monotonic() + min(60, settings.ai_explanation_cooldown_seconds)
            logger.warning("Optional OpenCode explanation was too short or malformed; using built-in fallback.")
            return None, last_error
        except Exception as exc:
            if isinstance(exc, requests.HTTPError) and exc.response is not None and exc.response.status_code == 429:
                last_error = "AI provider rate limit reached. Using the built-in explanation for this run; try again after the provider quota resets."
            elif isinstance(exc, requests.Timeout):
                last_error = (
                    f"AI provider did not respond within {settings.ai_explanation_timeout_seconds}s. "
                    "Using the built-in explanation so the forecast stays fast."
                )
                LLM_LAST_ERROR = "provider timeout"
                LLM_DISABLED_UNTIL = time.monotonic() + min(60, settings.ai_explanation_cooldown_seconds)
            else:
                last_error = f"AI provider failed: {exc}"
            logger.warning("Optional OpenCode explanation failed: %s", exc)
            if attempt < max_attempts - 1:
                continue
            return None, last_error
    return None, last_error


def build_forecast_explanation(payload: dict[str, Any], *, allow_external: bool = True) -> dict[str, str]:
    llm_error: str | None = None
    if allow_external:
        llm_text, llm_error = build_llm_explanation(payload)
        if llm_text:
            return {"provider": "opencode", "model": settings.opencode_model, "text": llm_text}
    response = {
        "provider": "built-in",
        "model": "deterministic",
        "text": build_deterministic_explanation(payload),
    }
    if llm_error:
        response["note"] = llm_error
    return response
