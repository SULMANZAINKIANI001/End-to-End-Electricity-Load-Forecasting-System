import base64
import html
import logging
import os
from datetime import date, timedelta
from urllib.parse import urlparse

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

logger = logging.getLogger(__name__)


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
DEFAULT_LOCAL_API_URLS = ("http://localhost:8000", "http://127.0.0.1:8000")
DEFAULT_COUNTRIES = {
    "DE_LU - Germany/Luxembourg | LSTM": "DE_LU",
    "US48 - EIA live | LSTM": "US48",
}
HISTORY_PRESETS = {
    "30 days": 30,
    "90 days": 90,
    "1 year": 365,
    "2 years": 730,
    "Custom": None,
}
DEFAULT_HISTORY_PRESET = "90 days"
UI_RESULT_VERSION = "operator-console-country-payload-v3"
FALLBACK_SETTINGS = {
    "max_history_days": 730,
    "forecast_history_days": 45,
    "default_history_days": 90,
    "max_forecast_horizon": 168,
    "default_forecast_horizon": 24,
}

SOURCE_LABELS = {
    "entsoe_live": "ENTSO-E live",
    "entsoe_export": "ENTSO-E export",
    "smard_dynamic_fallback": "SMARD live DE-LU",
    "eia_live": "EIA live",
    "pakistan_csv_demo": "Pakistan CSV demo",
    "csv": "CSV",
    "entsoe_or_smard": "ENTSO-E / SMARD",
    "eia": "EIA",
}

SOURCE_TRUST = {
    "entsoe_live": ("Primary", "Assignment-compliant live ENTSO-E API path."),
    "entsoe_export": ("Real", "Manual ENTSO-E export normalized for training and forecasting."),
    "smard_dynamic_fallback": ("Live DE-LU", "SMARD real Germany/Luxembourg hourly load is active."),
    "eia_live": ("Live", "U.S. hourly electricity demand from EIA Open Data."),
    "pakistan_csv_demo": ("Demo", "Historical Pakistan CSV. Not live ENTSO-E data."),
}

DATA_CLASS_LABELS = {
    "real_live": "Real live API",
    "real_dynamic": "Real dynamic source",
    "historical_demo": "Historical demo",
}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ops-bg: #eef3f8;
            --ops-panel: #ffffff;
            --ops-panel-soft: #f7faff;
            --ops-rail: #0b1220;
            --ops-rail-soft: #111c2f;
            --ops-border: #d6e0ea;
            --ops-border-strong: #b8c6d6;
            --ops-text: #0f1f33;
            --ops-muted: #66768a;
            --ops-blue: #1d4ed8;
            --ops-blue-soft: #e8f0ff;
            --ops-green: #047857;
            --ops-green-soft: #e9f8f2;
            --ops-amber: #b45309;
            --ops-amber-soft: #fff7ed;
            --ops-red: #dc2626;
            --ops-red-soft: #fff1f2;
            --ops-shadow: 0 10px 30px rgba(15, 31, 51, 0.08);
        }
        .stApp {
            background: var(--ops-bg);
            color: var(--ops-text);
        }
        header[data-testid="stHeader"] {
            height: 0;
            min-height: 0;
            background: transparent;
        }
        div[data-testid="stToolbar"] {
            display: none;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1220 0%, #111c2f 100%);
            border-right: 1px solid #22304a;
            box-shadow: 8px 0 24px rgba(15, 31, 51, 0.18);
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong {
            color: #f8fafc;
        }
        section[data-testid="stSidebar"] .stCaptionContainer,
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: #b8c5d6;
        }
        section[data-testid="stSidebar"] h4,
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h4 {
            color: #f8fafc !important;
            font-size: 0.84rem;
            margin: 0.65rem 0 0.2rem 0;
            text-transform: uppercase;
            letter-spacing: .05em;
        }
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
        section[data-testid="stSidebar"] div[data-baseweb="input"] > div {
            background: #f8fafc;
            border-color: #cbd5e1;
            border-radius: 8px;
        }
        section[data-testid="stSidebar"] div[data-baseweb="select"] span,
        section[data-testid="stSidebar"] div[data-baseweb="select"] input,
        section[data-testid="stSidebar"] div[data-baseweb="input"] input {
            color: var(--ops-text) !important;
            -webkit-text-fill-color: var(--ops-text) !important;
            opacity: 1 !important;
        }
        section[data-testid="stSidebar"] div[data-baseweb="select"] svg {
            fill: var(--ops-text);
        }
        section[data-testid="stSidebar"] [data-testid="stDateInput"] input {
            color: var(--ops-text) !important;
            -webkit-text-fill-color: var(--ops-text) !important;
        }
        section[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
            background: var(--ops-blue);
            border-color: var(--ops-blue);
        }
        section[data-testid="stSidebar"] button[kind="primary"],
        section[data-testid="stSidebar"] button[kind="secondary"] {
            border-radius: 8px;
            font-weight: 700;
            min-height: 42px;
        }
        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.75rem;
        }
        .block-container {
            padding-top: 0.6rem;
            padding-bottom: 2.5rem;
            max-width: 1540px;
        }
        h1 {
            color: var(--ops-text);
            letter-spacing: 0;
            font-size: 2.15rem;
            line-height: 1.15;
            margin-bottom: 0.2rem;
        }
        h2, h3 {
            color: var(--ops-text);
            letter-spacing: 0;
        }
        [data-testid="stCaptionContainer"], p, label {
            color: var(--ops-muted);
        }
        .ops-shell {
            border: 1px solid var(--ops-border);
            background: #ffffff;
            border-radius: 8px;
            padding: 12px 16px;
            box-shadow: 0 4px 18px rgba(15, 31, 51, 0.06);
            margin: 0 0 12px 0;
        }
        .ops-header {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            align-items: center;
            gap: 18px;
        }
        .ops-header-main {
            min-width: 320px;
        }
        .ops-title {
            font-size: 1.42rem;
            line-height: 1.1;
            font-weight: 800;
            color: var(--ops-text);
            margin: 0 0 5px 0;
        }
        .ops-subtitle {
            color: var(--ops-muted);
            font-size: 0.82rem;
            line-height: 1.35;
            max-width: 820px;
        }
        .ops-status-row {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            align-items: center;
            justify-content: flex-end;
            max-width: 650px;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 4px 8px;
            font-size: 0.68rem;
            font-weight: 800;
            border: 1px solid transparent;
            white-space: nowrap;
        }
        .tone-ok {background: var(--ops-green-soft); color: var(--ops-green); border-color: #a7f3d0;}
        .tone-info {background: var(--ops-blue-soft); color: var(--ops-blue); border-color: #bfdbfe;}
        .tone-warn {background: var(--ops-amber-soft); color: var(--ops-amber); border-color: #fed7aa;}
        .tone-bad {background: var(--ops-red-soft); color: var(--ops-red); border-color: #fecdd3;}
        .tone-neutral {background: #f1f5f9; color: #475569; border-color: #dbe4ee;}
        .ops-card,
        .metric-card {
            border: 1px solid var(--ops-border);
            border-radius: 8px;
            padding: 12px 14px;
            background: var(--ops-panel);
            min-height: 92px;
            box-shadow: 0 1px 2px rgba(16, 32, 51, 0.04);
            position: relative;
            overflow: hidden;
        }
        .ops-card:before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: var(--ops-blue);
            opacity: 0.9;
        }
        .ops-card.ok:before {background: var(--ops-green);}
        .ops-card.warn:before {background: var(--ops-amber);}
        .ops-card.bad:before {background: var(--ops-red);}
        .ops-card.neutral:before {background: #64748b;}
        .metric-label,
        .ops-card-label {font-size: 0.76rem; color: var(--ops-muted); margin-bottom: 7px; font-weight: 700; text-transform: uppercase; letter-spacing: .04em;}
        .metric-value,
        .ops-card-value {font-size: 1.18rem; font-weight: 800; color: var(--ops-text); line-height: 1.18; overflow-wrap: anywhere;}
        .metric-note,
        .ops-card-note {font-size: 0.8rem; color: var(--ops-muted); margin-top: 7px; line-height: 1.35;}
        .section-head {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 14px;
            margin: 18px 0 10px 0;
            border-bottom: 1px solid var(--ops-border);
            padding-bottom: 10px;
        }
        .section-title {
            color: var(--ops-text);
            font-size: 1.15rem;
            font-weight: 800;
            margin: 0;
        }
        .section-subtitle {
            color: var(--ops-muted);
            font-size: 0.88rem;
            margin-top: 3px;
        }
        .decision-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 14px;
            margin: 12px 0;
        }
        .decision-card {
            border: 1px solid var(--ops-border);
            border-radius: 8px;
            background: var(--ops-panel);
            padding: 14px 16px;
            min-height: 118px;
            box-shadow: 0 1px 2px rgba(16, 32, 51, 0.04);
            border-top: 3px solid var(--ops-blue);
        }
        .decision-title {
            color: var(--ops-muted);
            font-size: 0.78rem;
            margin-bottom: 8px;
        }
        .decision-main {
            color: var(--ops-text);
            font-size: 1.08rem;
            font-weight: 700;
            line-height: 1.25;
        }
        .decision-note {
            color: var(--ops-muted);
            font-size: 0.78rem;
            margin-top: 8px;
            line-height: 1.35;
        }
        @media (max-width: 1100px) {
            .decision-grid {grid-template-columns: repeat(2, minmax(0, 1fr));}
        }
        @media (max-width: 720px) {
            .decision-grid {grid-template-columns: 1fr;}
        }
        .source-card {
            border: 1px solid #bfdbfe;
            border-left: 4px solid var(--ops-blue);
            border-radius: 6px;
            padding: 12px 14px;
            background: var(--ops-panel-soft);
            margin: 8px 0 12px 0;
            color: var(--ops-text);
        }
        .source-card strong {color: var(--ops-text);}
        .source-card.ok {border-color: #a7f3d0; border-left-color: var(--ops-green); background: var(--ops-green-soft);}
        .source-card.warn {border-color: #fed7aa; border-left-color: var(--ops-amber); background: var(--ops-amber-soft);}
        .source-card.bad {border-color: #fecdd3; border-left-color: var(--ops-red); background: var(--ops-red-soft);}
        .range-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin: 10px 0 12px 0;
        }
        .range-item {
            border: 1px solid var(--ops-border);
            border-radius: 8px;
            background: var(--ops-panel);
            padding: 12px 14px;
        }
        .range-label {
            font-size: 0.76rem;
            color: var(--ops-muted);
            margin-bottom: 5px;
        }
        .range-value {
            font-size: 0.92rem;
            color: var(--ops-text);
            font-weight: 700;
            line-height: 1.35;
        }
        .chart-panel {
            border: 1px solid var(--ops-border);
            background: var(--ops-panel);
            border-radius: 8px;
            padding: 12px 12px 4px 12px;
            box-shadow: var(--ops-shadow);
            margin-top: 8px;
        }
        .table-panel {
            border: 1px solid var(--ops-border);
            background: var(--ops-panel);
            border-radius: 8px;
            padding: 12px;
            margin: 10px 0;
        }
        .empty-state {
            border: 1px dashed var(--ops-border-strong);
            border-radius: 8px;
            padding: 18px;
            background: #fbfdff;
            color: var(--ops-muted);
        }
        @media (max-width: 900px) {
            .range-strip {grid-template-columns: 1fr;}
        }
        .status-ok {color: var(--ops-green); font-weight: 700;}
        .status-warn {color: #b45309; font-weight: 700;}
        .status-bad {color: var(--ops-red); font-weight: 700;}
        .explain-box {
            border: 1px solid #bfdbfe;
            border-radius: 8px;
            border-left: 4px solid var(--ops-blue);
            padding: 16px 18px 14px 18px;
            background: #fbfdff;
            color: var(--ops-text);
            box-shadow: 0 1px 2px rgba(16, 32, 51, 0.04);
        }
        .brief-heading {
            color: var(--ops-text);
            font-weight: 800;
            font-size: 1.02rem;
            margin: 14px 0 6px 0;
            padding-top: 8px;
            border-top: 1px solid #e7edf4;
        }
        .brief-heading:first-child {
            margin-top: 0;
            padding-top: 0;
            border-top: 0;
        }
        .brief-line {
            color: var(--ops-text);
            font-size: 0.92rem;
            line-height: 1.42;
            margin: 3px 0;
        }
        .brief-action {
            color: var(--ops-text);
            background: #f7faff;
            border: 1px solid #e1e9f5;
            border-radius: 6px;
            padding: 8px 10px;
            margin: 6px 0;
            font-size: 0.9rem;
            line-height: 1.38;
        }
        div[data-testid="stMetric"] {
            background: var(--ops-panel);
            border: 1px solid var(--ops-border);
            border-radius: 8px;
            padding: 10px 12px;
            box-shadow: 0 1px 2px rgba(16, 32, 51, 0.04);
        }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
            color: var(--ops-muted) !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: var(--ops-text) !important;
        }
        div[data-testid="stAlert"] {
            border-radius: 8px;
        }
        div[data-testid="stTabs"] button {
            font-weight: 700;
            color: var(--ops-muted);
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--ops-blue);
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {
            border-radius: 8px;
        }
        @media (max-width: 760px) {
            .ops-title {font-size: 1.35rem;}
            .ops-header {display: block;}
            .ops-status-row {justify-content: flex-start; margin-top: 12px;}
            .block-container {padding-left: 1rem; padding-right: 1rem;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def source_label(source: str | None) -> str:
    return SOURCE_LABELS.get(source or "", source or "unknown")


def trust_label(source: str | None) -> tuple[str, str]:
    return SOURCE_TRUST.get(source or "", ("Unknown", "Source will be shown after data is fetched."))


def tone_for_ready(ok: bool) -> str:
    return "ok" if ok else "warn"


def tone_for_source(source: str | None) -> str:
    if source in {"entsoe_live", "entsoe_export", "smard_dynamic_fallback", "eia_live", "entsoe_or_smard", "entsoe", "eia"}:
        return "ok"
    if source == "pakistan_csv_demo":
        return "warn"
    return "neutral"


def tone_for_reliability(label: str) -> str:
    return {"High": "ok", "Good": "info", "Moderate": "warn", "Low": "bad"}.get(label, "neutral")


def status_pill(label: str, tone: str = "neutral") -> str:
    return f'<span class="status-pill tone-{html.escape(tone)}">{html.escape(label)}</span>'


def reset_saved_forecast() -> None:
    st.session_state.pop("last_forecast_result", None)
    st.session_state.pop("last_forecast_payload", None)
    st.session_state["last_forecast_result_version"] = UI_RESULT_VERSION


def forecast_payload_label(payload: dict | None) -> str:
    if not isinstance(payload, dict):
        return "another selection"
    return (
        f"{payload.get('country_code', 'unknown')}, "
        f"{payload.get('start', 'unknown')} to {payload.get('end', 'unknown')}, "
        f"{payload.get('horizon', 'unknown')}h"
    )


def forecast_result_is_stale(result: dict | None) -> bool:
    if not isinstance(result, dict):
        return False
    explanation = result.get("explanation", {})
    text = str(explanation.get("text", "")) if isinstance(explanation, dict) else ""
    if "sqrt scaling" in text or "+/- 131.3 GW" in text:
        return True
    forecast = result.get("forecast", [])
    if isinstance(forecast, list) and forecast:
        try:
            widths = [
                float(point["upper_bound_mw"]) - float(point["lower_bound_mw"])
                for point in forecast
                if isinstance(point, dict)
            ]
            predictions = [float(point["predicted_load_mw"]) for point in forecast if isinstance(point, dict)]
            if widths and predictions:
                avg_half_width = sum(widths) / len(widths) / 2
                avg_prediction = sum(predictions) / len(predictions)
                return avg_prediction > 0 and avg_half_width / avg_prediction > 0.25
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return True
    return False


def format_operator_brief(text: object) -> str:
    raw = str(text or "No explanation available.")
    heading_names = {
        "Executive forecast brief",
        "Operator guidance",
        "Why the forecast looks this way",
        "Time-of-day pattern",
        "How to read the forecast",
        "Data used",
        "Model confidence",
        "Next action checklist",
    }
    parts: list[str] = []
    previous_blank = False
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line:
            previous_blank = True
            continue
        escaped = html.escape(line)
        if line in heading_names:
            parts.append(f'<div class="brief-heading">{escaped}</div>')
        elif len(line) > 2 and line[0].isdigit() and line[1:3] in {". ", ") "}:
            parts.append(f'<div class="brief-action">{escaped}</div>')
        else:
            margin = " style=\"margin-top:8px;\"" if previous_blank and parts else ""
            parts.append(f'<div class="brief-line"{margin}>{escaped}</div>')
        previous_blank = False
    return "".join(parts)


def fmt_mw(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    numeric = float(value)
    if abs(numeric) >= 10_000:
        return f"{numeric / 1000:,.1f} GW"
    return f"{numeric:,.0f} MW"


def fmt_pct(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}%"


def metric_card(label: str, value: str, note: str = "") -> None:
    ops_card(label, value, note, tone="neutral")


def ops_card(label: str, value: str, note: str = "", tone: str = "neutral") -> None:
    st.markdown(
        f"""
        <div class="ops-card {html.escape(tone)}">
            <div class="ops-card-label">{html.escape(label)}</div>
            <div class="ops-card-value">{html.escape(value)}</div>
            <div class="ops-card-note">{html.escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str = "", right: str = "") -> None:
    st.markdown(
        f"""
        <div class="section-head">
            <div>
                <div class="section-title">{html.escape(title)}</div>
                <div class="section-subtitle">{html.escape(subtitle)}</div>
            </div>
            <div>{right}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def source_status_card(
    *,
    source: str | None,
    trust: str,
    detail: str,
    freshness: str = "",
    rows: object = "",
    latest: object = "",
) -> None:
    tone = tone_for_source(source)
    parts = [
        f"<strong>Source:</strong> {html.escape(source_label(source))}",
        f"<strong>Status:</strong> {html.escape(trust)}",
    ]
    if freshness:
        parts.append(f"<strong>Freshness:</strong> {html.escape(freshness)}")
    if rows:
        parts.append(f"<strong>Rows:</strong> {html.escape(str(rows))}")
    if latest:
        parts.append(f"<strong>Latest:</strong> {html.escape(str(latest))}")
    st.markdown(
        f"""
        <div class="source-card {html.escape(tone)}">
            {' | '.join(parts)}<br>
            {html.escape(detail)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_text(ok: bool) -> str:
    return "Ready" if ok else "Action needed"


def country_label(code: str, metadata: dict) -> str:
    data_class = DATA_CLASS_LABELS.get(str(metadata.get("data_class", "")), "Data source")
    name = str(metadata.get("name", code)).split(" - ")[0]
    method = "LSTM" if metadata.get("model_ready") else "Needs training"
    metrics = metadata.get("metrics", {}) if isinstance(metadata.get("metrics"), dict) else {}
    mape = metrics.get("mape")
    suffix = f"{method} | {float(mape):.2f}% error" if mape is not None else method
    return f"{code} - {name} | {suffix}"


def fmt_time(value: object) -> str:
    try:
        timestamp = pd.Timestamp(value)
        return timestamp.strftime("%b %d, %Y %H:%M")
    except Exception:
        logger.debug("fmt_time fallback for value=%r", value, exc_info=True)
        return str(value)


def fmt_date_range(start: object, end: object) -> str:
    try:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        return f"{start_ts.strftime('%b %d, %Y %H:%M')} -> {end_ts.strftime('%b %d, %Y %H:%M')}"
    except Exception:
        logger.debug("fmt_date_range fallback", exc_info=True)
        return "not available"


def fmt_selected_range(history_window: dict, fallback_start: object, fallback_end: object) -> str:
    start = history_window.get("selected_start", fallback_start)
    end = history_window.get("selected_end", fallback_end)
    suffix = " inclusive" if history_window.get("selected_end_inclusive") else ""
    try:
        start_label = pd.Timestamp(start).strftime("%b %d, %Y")
        end_label = pd.Timestamp(end).strftime("%b %d, %Y")
        return f"{start_label} -> {end_label}{suffix}"
    except Exception:
        logger.debug("fmt_selected_range fallback", exc_info=True)
        return "not available"


def trend_summary(first_value: float, last_value: float) -> tuple[str, str]:
    if first_value == 0:
        return "Stable", "No clear percentage change."
    change_pct = (last_value - first_value) / first_value * 100
    if change_pct > 3:
        return "Rising demand", f"Up {change_pct:.1f}% across the horizon."
    if change_pct < -3:
        return "Falling demand", f"Down {abs(change_pct):.1f}% across the horizon."
    return "Stable demand", f"Only {change_pct:.1f}% change across the horizon."


def render_forecast_chart(forecast_df: pd.DataFrame, history_df: pd.DataFrame, history_window: dict) -> None:
    fig = go.Figure()
    if not history_df.empty:
        fig.add_trace(
            go.Scattergl(
                x=history_df["timestamp"],
                y=history_df["actual_load_mw"],
                mode="lines",
                name="Actual history",
                line={"color": "#0f766e", "width": 1.35},
                opacity=0.92,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["timestamp"],
            y=forecast_df["upper_bound_mw"],
            line={"width": 0},
            name="Upper band",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["timestamp"],
            y=forecast_df["lower_bound_mw"],
            fill="tonexty",
            line={"width": 0},
            name="Confidence band",
            fillcolor="rgba(29, 78, 216, 0.14)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["timestamp"],
            y=forecast_df["baseline_load_mw"],
            mode="lines",
            name="Seasonal baseline",
            line={"color": "#64748b", "dash": "dot", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["timestamp"],
            y=forecast_df["predicted_load_mw"],
            mode="lines+markers",
            name="LSTM forecast",
            line={"color": "#1d4ed8", "width": 3},
            marker={"size": 5, "color": "#1d4ed8"},
        )
    )
    forecast_start = history_window.get("forecast_start")
    if forecast_start:
        forecast_start_ts = pd.Timestamp(forecast_start).to_pydatetime()
        fig.add_shape(
            type="line",
            x0=forecast_start_ts,
            x1=forecast_start_ts,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line={"color": "#dc2626", "width": 1.4, "dash": "dash"},
        )
        fig.add_annotation(
            x=forecast_start_ts,
            y=1,
            xref="x",
            yref="paper",
            text="Forecast starts",
            showarrow=False,
            yshift=12,
            font={"size": 11, "color": "#dc2626"},
        )
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Load (MW)",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12, "x": 0, "font": {"size": 12}},
        margin={"l": 24, "r": 18, "t": 42, "b": 34},
        height=500,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font={"color": "#0f1f33"},
        xaxis={"gridcolor": "#e7edf4", "zeroline": False},
        yaxis={"gridcolor": "#e7edf4", "zeroline": False},
    )
    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def api_get(path: str, params: dict | None = None) -> dict:
    return api_request("GET", path, params=params)


def api_post(path: str, payload: dict) -> dict:
    return api_request("POST", path, json=payload, timeout=120)


def api_candidates() -> list[str]:
    candidates = [API_BASE_URL]
    parsed = urlparse(API_BASE_URL)
    if parsed.hostname == "api":
        candidates.extend(DEFAULT_LOCAL_API_URLS)
    else:
        candidates.append("http://api:8000")
        candidates.extend(DEFAULT_LOCAL_API_URLS)
    extra_urls = [
        value.strip().rstrip("/")
        for value in os.getenv("API_FALLBACK_URLS", "").split(",")
        if value.strip()
    ]
    candidates.extend(extra_urls)
    deduped: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in deduped:
            deduped.append(candidate)
    return deduped


def api_request(method: str, path: str, *, params: dict | None = None, json: dict | None = None, timeout: int = 45) -> dict:
    connection_errors: list[str] = []
    timeout_errors: list[str] = []
    for base_url in api_candidates():
        try:
            response = requests.request(
                method,
                f"{base_url}{path}",
                params=params,
                json=json,
                timeout=timeout,
            )
            response.raise_for_status()
            st.session_state["active_api_base_url"] = base_url
            return response.json()
        except requests.HTTPError:
            raise
        except requests.Timeout as exc:
            timeout_errors.append(f"{base_url}: timed out after {timeout}s")
            if method.upper() == "POST":
                raise requests.Timeout(
                    "The API is reachable but the request took too long. "
                    "Use a shorter history range or wait for the backend to finish."
                ) from exc
        except requests.RequestException as exc:
            connection_errors.append(f"{base_url}: {exc.__class__.__name__}")
    if timeout_errors:
        raise requests.Timeout("API request timed out: " + "; ".join(timeout_errors))
    raise requests.ConnectionError("Could not connect to API candidates: " + "; ".join(connection_errors))


def show_api_error(error: Exception) -> None:
    if isinstance(error, requests.HTTPError):
        try:
            detail = error.response.json().get("detail", error.response.text)
        except ValueError:
            detail = error.response.text
        st.error(f"API error: {detail}")
        return
    if isinstance(error, requests.Timeout):
        st.error(str(error))
        st.info("Forecast requests use a capped recent history window for speed. If this persists, restart Docker and try a smaller date range.")
        return
    st.error(f"Could not connect to the API. Tried: {', '.join(api_candidates())}.")


def show_dashboard_error(error: Exception) -> None:
    st.error("The API returned a forecast, but the dashboard could not render one section of the result.")
    with st.expander("Technical detail", expanded=False):
        st.code(f"{error.__class__.__name__}: {error}")


@st.cache_data(ttl=300)
def load_countries(include_demo: bool = False, forecast_ready_only: bool = True) -> dict[str, str]:
    try:
        payload = api_get(
            "/countries",
            params={"include_demo": include_demo, "forecast_ready_only": forecast_ready_only},
        )
        countries = payload.get("countries", {})
        mapped = {country_label(code, value): code for code, value in countries.items()}
        return mapped or DEFAULT_COUNTRIES
    except Exception:
        logger.warning("fetch_country_options fallback", exc_info=True)
        return DEFAULT_COUNTRIES


@st.cache_data(ttl=300)
def load_country_catalog(include_demo: bool = False, forecast_ready_only: bool = True) -> dict:
    try:
        return api_get(
            "/countries",
            params={"include_demo": include_demo, "forecast_ready_only": forecast_ready_only},
        )
    except Exception:
        logger.warning("fetch_countries fallback", exc_info=True)
        return {"countries": {}}


@st.cache_data(ttl=300)
def load_data_range(country_code: str) -> dict:
    try:
        return api_get("/data/range", params={"country_code": country_code})
    except Exception:
        logger.warning("fetch_data_range fallback country=%s", country_code, exc_info=True)
        return {}


@st.cache_data(ttl=300)
def load_app_settings() -> dict:
    try:
        return {**FALLBACK_SETTINGS, **api_get("/settings")}
    except Exception:
        logger.warning("fetch_settings fallback", exc_info=True)
        return FALLBACK_SETTINGS.copy()


@st.cache_data(ttl=60)
def load_model_status(country_code: str) -> dict:
    try:
        return api_get("/model/status", params={"country_code": country_code})
    except Exception:
        logger.warning("fetch_model_status fallback country=%s", country_code, exc_info=True)
        return {}


@st.cache_data(ttl=30)
def load_monitoring_status() -> dict:
    try:
        return api_get("/monitoring/status")
    except Exception:
        logger.warning("fetch_monitoring fallback", exc_info=True)
        return {}


st.set_page_config(page_title="Electricity Load Forecasting", layout="wide")
inject_styles()
if st.session_state.get("last_forecast_result_version") != UI_RESULT_VERSION:
    reset_saved_forecast()
elif forecast_result_is_stale(st.session_state.get("last_forecast_result")):
    reset_saved_forecast()
    st.info("Old saved forecast cleared because the uncertainty-band logic was updated. Run forecast again for corrected values.")

with st.sidebar:
    st.markdown("### Operations Console")
    st.caption("Forecast controls")
    st.markdown("#### Source & model")
    show_demo_countries = st.toggle("Show historical demo countries", value=False)
    show_untrained_countries = st.toggle("Show countries needing training", value=False)
    forecast_ready_only = not show_untrained_countries
    countries = load_countries(show_demo_countries, forecast_ready_only)
    catalog = load_country_catalog(show_demo_countries, forecast_ready_only).get("countries", {})
    app_settings = load_app_settings()
    st.caption(
        "Default mode shows only countries with a ready LSTM model. "
        "Enable training/demo toggles only when you intentionally want baseline or demo behavior."
    )
    country_options = list(countries.keys())
    country_by_code = {code: label for label, code in countries.items()}
    preferred_country_code = str(
        st.session_state.get("selected_country_code")
        or app_settings.get("primary_country_code")
        or "DE_LU"
    )
    default_country_name = country_by_code.get(preferred_country_code) or (country_options[0] if country_options else "")
    if country_options and st.session_state.get("country_selector_v2") not in country_options:
        st.session_state["country_selector_v2"] = default_country_name
    country_name = st.selectbox(
        "Country",
        country_options,
        index=country_options.index(default_country_name) if default_country_name in country_options else 0,
        key="country_selector_v2",
    )
    country_code = countries[country_name]
    st.session_state["selected_country_code"] = country_code
    country_info = catalog.get(country_code, {})
    available_range = load_data_range(country_code)
    max_history_days = max(1, int(app_settings.get("max_history_days", FALLBACK_SETTINGS["max_history_days"])))
    forecast_history_days = max(1, int(app_settings.get("forecast_history_days", FALLBACK_SETTINGS["forecast_history_days"])))
    max_forecast_horizon = max(1, int(app_settings.get("max_forecast_horizon", FALLBACK_SETTINGS["max_forecast_horizon"])))
    default_forecast_horizon = min(
        max_forecast_horizon,
        max(1, int(app_settings.get("default_forecast_horizon", FALLBACK_SETTINGS["default_forecast_horizon"]))),
    )

    latest_safe_date = date.today() - timedelta(days=2)
    has_static_range = bool(available_range.get("start") and available_range.get("end"))
    if has_static_range:
        data_start = pd.Timestamp(available_range["start"]).date()
        data_end = pd.Timestamp(available_range["end"]).date()
        min_selectable_date = data_start
        max_selectable_date = data_end
        default_end = data_end
        st.caption(f"Available source data: {data_start.isoformat()} to {data_end.isoformat()}")
    else:
        default_end = latest_safe_date
        min_selectable_date = latest_safe_date - timedelta(days=max_history_days - 1)
        max_selectable_date = latest_safe_date

    st.markdown("#### History window")
    st.caption("Start/end select historical data used for forecasting. Forecast begins after the end date.")
    st.caption(f"For speed, the LSTM engine uses the latest {forecast_history_days} days from long ranges, while the chart shows the selected window.")
    if country_code == "PK":
        st.caption("Pakistan is bounded by the local CSV range.")
    elif has_static_range:
        st.caption(f"This source supports up to {max_history_days} selected days within the imported data range.")
    else:
        st.caption(f"Maximum selectable history: {max_history_days} days. Recommended fast range: {forecast_history_days}-90 days.")

    preset_options = list(HISTORY_PRESETS)
    preset_index = preset_options.index(DEFAULT_HISTORY_PRESET)
    history_preset = st.radio(
        "History length",
        preset_options,
        index=preset_index,
        key=f"{country_code}-history-preset",
    )
    range_is_valid = True
    preset_days = HISTORY_PRESETS[history_preset]

    if preset_days is None:
        default_custom_start = max(min_selectable_date, default_end - timedelta(days=min(90, max_history_days) - 1))
        start_date = st.date_input(
            "Start date",
            default_custom_start,
            min_value=min_selectable_date,
            max_value=max_selectable_date,
            key=f"{country_code}-custom-start",
        )
        end_date = st.date_input(
            "End date",
            default_end,
            min_value=min_selectable_date,
            max_value=max_selectable_date,
            key=f"{country_code}-custom-end",
        )
        selected_days = (end_date - start_date).days + 1
        if end_date < start_date:
            range_is_valid = False
            st.error("End date cannot be before start date.")
        elif selected_days > max_history_days:
            range_is_valid = False
            st.error(f"Selected history is {selected_days} days. Maximum allowed is {max_history_days} days.")
    else:
        selected_days = min(int(preset_days), max_history_days)
        if int(preset_days) > max_history_days:
            st.warning(f"{history_preset} is longer than the configured limit, so it is capped to {max_history_days} days.")
        end_date = st.date_input(
            "End date",
            default_end,
            min_value=min_selectable_date,
            max_value=max_selectable_date,
            key=f"{country_code}-preset-end-{history_preset}",
        )
        start_date = max(min_selectable_date, end_date - timedelta(days=selected_days - 1))
        st.text_input("Start date", start_date.isoformat(), disabled=True, key=f"{country_code}-preset-start-{history_preset}")
        if start_date == min_selectable_date and ((end_date - start_date).days + 1) < selected_days:
            st.info("The selected preset starts at the earliest available source row for this country.")

    selected_history_days = max(0, (end_date - start_date).days + 1)
    if selected_history_days > forecast_history_days:
        st.info(
            f"Selected window: {selected_history_days} days. The graph will show this window; the LSTM engine will use the latest "
            f"{forecast_history_days} days from it to stay responsive."
        )

    st.markdown("#### Forecast action")
    horizon = st.slider(
        "Forecast horizon (hours)",
        min_value=1,
        max_value=max_forecast_horizon,
        value=default_forecast_horizon,
    )
    run_forecast = st.button("Run forecast", type="primary", use_container_width=True, disabled=not range_is_valid)
    if st.session_state.get("last_forecast_result"):
        if st.button("Clear forecast", use_container_width=True):
            reset_saved_forecast()
            st.rerun()
    if country_code == "PK":
        st.warning("Pakistan is historical CSV demo data, not live public API data.")
    if country_code == "US48":
        st.caption("US48 uses EIA live data when EIA_API_KEY is configured.")
    if country_info.get("description"):
        st.caption(str(country_info["description"]))
    if country_info and not country_info.get("model_ready", False):
        st.warning("This country does not have LSTM artifacts yet. Forecasts will use the seasonal baseline until trained.")

model_status = load_model_status(country_code)
model_loaded = bool(model_status.get("model_ready", model_status.get("model_loaded")))
artifact_country = model_status.get("artifact_country_code")
primary_country = model_status.get("primary_country_code")
metadata = model_status.get("metadata", {}) if isinstance(model_status.get("metadata"), dict) else {}
metrics = metadata.get("metrics", {}) if isinstance(metadata.get("metrics"), dict) else {}
model_status_label = "LSTM ready" if model_loaded else "Baseline active"
primary_ready = bool(model_status.get("primary_model_ready"))
source_profile = DATA_CLASS_LABELS.get(str(country_info.get("data_class", "")), "Data source")
short_source_profile = {
    "Real dynamic source": "Real dynamic",
    "Real live API": "Real live",
    "Historical demo": "Demo",
}.get(source_profile, source_profile)
assignment_label = "ENTSO-E compliant" if country_info.get("assignment_compliant") else "Non-ENTSO-E source"
live_label = "Live capable" if country_info.get("live_capable") else "Historical source"
status_row = " ".join(
    [
        status_pill(model_status_label, tone_for_ready(model_loaded)),
        status_pill(short_source_profile, "info"),
        status_pill(assignment_label, "ok" if country_info.get("assignment_compliant") else "warn"),
        status_pill(live_label, "ok" if country_info.get("live_capable") else "warn"),
    ]
)
st.markdown(
    f"""
    <div class="ops-shell">
        <div class="ops-header">
            <div class="ops-header-main">
                <div class="ops-title">Electricity Load Forecasting Operations</div>
                <div class="ops-subtitle">
                    Live-source load forecasting with LSTM predictions, uncertainty bands, EDA evidence, and deployment monitoring.
                </div>
            </div>
            <div class="ops-status-row">{status_row}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
if artifact_country and artifact_country != country_code:
    st.warning(f"Loaded LSTM artifacts are for {artifact_country}; selected country is {country_code}.")

top_cols = st.columns(5)
mape_metric = metrics.get("mape") if isinstance(metrics, dict) else None
mape_tone = "neutral"
if isinstance(mape_metric, (int, float)):
    mape_tone = tone_for_reliability("High" if mape_metric <= 5 else "Good" if mape_metric <= 10 else "Moderate" if mape_metric <= 15 else "Low")
with top_cols[0]:
    ops_card("Selected country", country_code, str(country_info.get("name", country_name)).split(" - ")[0], "info")
with top_cols[1]:
    ops_card("Forecast engine", "LSTM" if model_loaded else "Baseline", model_status_label, tone_for_ready(model_loaded))
with top_cols[2]:
    ops_card("Saved MAPE", fmt_pct(mape_metric) if isinstance(mape_metric, (int, float)) else "n/a", "latest model metric", mape_tone)
with top_cols[3]:
    ops_card("Primary model", "Ready" if primary_ready else "Pending", f"primary: {primary_country or 'n/a'}", tone_for_ready(primary_ready))
with top_cols[4]:
    ops_card("Source profile", source_label(country_info.get("source")), short_source_profile, "ok" if country_info.get("live_capable") else "warn")

if country_info:
    source_status_card(
        source=country_info.get("source"),
        trust="Selected profile",
        detail=str(country_info.get("description", "")),
        freshness=assignment_label,
    )

tabs = st.tabs(["Forecast", "EDA", "Monitoring", "About"])
current_forecast_payload = {
    "country_code": country_code,
    "start": start_date.isoformat(),
    "end": end_date.isoformat(),
    "horizon": horizon,
}
saved_forecast_payload = st.session_state.get("last_forecast_payload", {})
if (
    isinstance(saved_forecast_payload, dict)
    and saved_forecast_payload.get("country_code")
    and saved_forecast_payload.get("country_code") != country_code
):
    reset_saved_forecast()
    saved_forecast_payload = {}
saved_forecast_matches_current = bool(st.session_state.get("last_forecast_result")) and saved_forecast_payload == current_forecast_payload

with tabs[0]:
    section_header("Forecast workspace", "Run a live-source forecast, inspect uncertainty, and translate the result into operator action.")
    if not model_loaded:
        with st.expander("Enable LSTM forecasts", expanded=False):
            st.warning(
                "LSTM model artifacts are not available yet, so this chart uses a seasonal naive fallback. "
                "Train the model to enable neural-network forecasts."
            )
            st.write("Missing artifacts:")
            st.write(f"- Model: `{model_status.get('model_path', 'models/load_lstm.keras')}`")
            st.write(f"- Scaler: `{model_status.get('scaler_path', 'models/load_scaler.joblib')}`")
            st.write("Recommended training command:")
            st.code(
                model_status.get(
                    "training_command",
                    "py -3.10 -m src.train --country-code PK --years 2 --horizon 24 --epochs 30",
                ),
                language="powershell",
            )
            st.caption("TensorFlow training should use Python 3.10. Restart FastAPI after training so the artifacts load.")

    if run_forecast or saved_forecast_matches_current:
        payload = current_forecast_payload
        spinner_text = "Fetching recent load and generating forecast..." if run_forecast else "Restoring the last forecast..."
        with st.spinner(spinner_text):
            try:
                if run_forecast:
                    result = api_post("/forecast", payload)
                    st.session_state["last_forecast_result"] = result
                    st.session_state["last_forecast_payload"] = payload
                    st.session_state["last_forecast_result_version"] = UI_RESULT_VERSION
                else:
                    result = st.session_state["last_forecast_result"]
                    if forecast_result_is_stale(result):
                        reset_saved_forecast()
                        st.warning("The saved forecast used older uncertainty-band logic. It was cleared; run a new forecast.")
                        st.stop()
                forecast_df = pd.DataFrame(result["forecast"])
                if forecast_df.empty:
                    st.warning("No forecast points were returned.")
                else:
                    history_df = pd.DataFrame(result.get("history", []))
                    if not history_df.empty:
                        history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
                    forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"])
                    history_window = result.get("history_window", {})
                    method = result.get("forecast_method", "unknown")
                    result_metadata = result.get("model_metadata", {})
                    result_metrics = result_metadata.get("metrics", {}) if isinstance(result_metadata, dict) else {}
                    data_source = result.get("data_source", {})
                    active_source = data_source.get("source", "unknown")
                    trust, trust_note = trust_label(active_source)
                    forecast_summary = result.get("forecast_summary", {})
                    hourly_profile = result.get("hourly_profile", {})
                    data_freshness = result.get("data_freshness", {})
                    recent_accuracy = result.get("recent_accuracy", {})
                    forecast_peak = forecast_df.loc[forecast_df["predicted_load_mw"].idxmax()]
                    forecast_low = forecast_df.loc[forecast_df["predicted_load_mw"].idxmin()]
                    average_prediction = forecast_summary.get("average_load_mw", float(forecast_df["predicted_load_mw"].mean()))
                    average_band = float((forecast_df["upper_bound_mw"] - forecast_df["lower_bound_mw"]).mean())
                    baseline_gap = forecast_summary.get("baseline_gap_mw", float((forecast_df["predicted_load_mw"] - forecast_df["baseline_load_mw"]).mean()))
                    trend_direction = forecast_summary.get("trend_direction", "stable")
                    trend_change_pct = forecast_summary.get("trend_change_pct", 0.0)
                    trend_label = trend_direction.capitalize()
                    trend_note = f"Load trending {trend_direction} ({trend_change_pct:+.1f}% over horizon)"
                    mape_value = result_metrics.get("mape") if result_metrics else None
                    backtest_mape = recent_accuracy.get("mape_pct") if recent_accuracy.get("available") else None
                    composite_mape = backtest_mape if isinstance(backtest_mape, (int, float)) else mape_value
                    if isinstance(composite_mape, (int, float)):
                        reliability = "High" if composite_mape <= 5 else "Good" if composite_mape <= 10 else "Moderate" if composite_mape <= 15 else "Low"
                    else:
                        reliability = "Unknown"
                    action_text = "Prepare for the peak hour" if trend_direction == "rising" else "Watch the peak hour"
                    if trend_direction == "falling":
                        action_text = "Avoid over-commitment"
                    staleness = data_freshness.get("staleness", "unknown")
                    hours_behind = data_freshness.get("hours_behind_now")
                    freshness_label = {"fresh": "Fresh data", "stale": "Stale", "very_stale": "Very stale", "unknown": "Unknown"}.get(staleness, "Unknown")
                    freshness_detail = f"{hours_behind:.0f}h behind now" if isinstance(hours_behind, (int, float)) else ""
                    peak_to_avg = forecast_summary.get("peak_to_avg_ratio")
                    load_change_rate = forecast_summary.get("load_change_rate_mw_per_hour")

                    summary_cols = st.columns(5)
                    with summary_cols[0]:
                        ops_card("Forecast method", "LSTM" if method == "lstm" else "Baseline", method, tone_for_ready(method == "lstm"))
                    with summary_cols[1]:
                        ops_card("Data source", source_label(active_source), trust, tone_for_source(active_source))
                    with summary_cols[2]:
                        ops_card("Average load", fmt_mw(average_prediction), f"{result['horizon']} hour horizon", "info")
                    with summary_cols[3]:
                        ops_card("Accuracy", reliability, f"Composite MAPE: {fmt_pct(composite_mape) if isinstance(composite_mape, (int, float)) else 'n/a'}", tone_for_reliability(reliability))
                    with summary_cols[4]:
                        ops_card("Data freshness", freshness_label, freshness_detail, "ok" if staleness == "fresh" else "warn" if staleness in {"stale", "very_stale"} else "neutral")

                    section_header("Decision strip", "The four signals below convert the forecast into dispatch and reserve planning cues.")
                    st.markdown(
                        f"""
                        <div class="decision-grid">
                            <div class="decision-card">
                                <div class="decision-title">What is happening?</div>
                                <div class="decision-main">{html.escape(str(trend_label))}</div>
                                <div class="decision-note">{html.escape(str(trend_note))}</div>
                            </div>
                            <div class="decision-card">
                                <div class="decision-title">When is the peak?</div>
                                <div class="decision-main">{html.escape(str(fmt_mw(forecast_peak["predicted_load_mw"])))}</div>
                                <div class="decision-note">{html.escape(str(fmt_time(forecast_peak["timestamp"])))}</div>
                            </div>
                            <div class="decision-card">
                                <div class="decision-title">Can I trust it?</div>
                                <div class="decision-main">{html.escape(str(reliability))}</div>
                                <div class="decision-note">Composite MAPE: {fmt_pct(composite_mape) if isinstance(composite_mape, (int, float)) else "n/a"} | Backtest: {fmt_pct(backtest_mape) if isinstance(backtest_mape, (int, float)) else "n/a"}</div>
                            </div>
                            <div class="decision-card">
                                <div class="decision-title">Recommended action</div>
                                <div class="decision-main">{html.escape(str(action_text))}</div>
                                <div class="decision-note">Load changing {html.escape(str(round(load_change_rate, 1)))} MW/hr | Peak/avg: {html.escape(str(round(peak_to_avg, 2))) if isinstance(peak_to_avg, (int, float)) else "n/a"}</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    source_status_card(
                        source=active_source,
                        trust=trust,
                        detail=f"{trust_note} Model trained for {result_metadata.get('country_code', 'n/a') if isinstance(result_metadata, dict) else 'n/a'} using {source_label(result_metadata.get('data_source')) if isinstance(result_metadata, dict) else 'n/a'}.",
                        freshness=f"{freshness_label} {f'({freshness_detail})' if freshness_detail else ''}",
                        rows=data_source.get("rows", ""),
                        latest=data_source.get("latest_timestamp", ""),
                    )
                    st.markdown(
                        f"""
                        <div class="range-strip">
                            <div class="range-item">
                                <div class="range-label">Selected history window</div>
                                <div class="range-value">{html.escape(str(fmt_selected_range(history_window, start_date, end_date)))}</div>
                            </div>
                            <div class="range-item">
                                <div class="range-label">Actual history shown</div>
                                <div class="range-value">{html.escape(str(fmt_date_range(history_window.get('actual_start'), history_window.get('actual_end'))))}</div>
                            </div>
                            <div class="range-item">
                                <div class="range-label">LSTM input window</div>
                                <div class="range-value">{html.escape(str(fmt_date_range(history_window.get('model_input_start'), history_window.get('model_input_end'))))}</div>
                            </div>
                            <div class="range-item">
                                <div class="range-label">Forecast period</div>
                                <div class="range-value">{html.escape(str(fmt_date_range(history_window.get('forecast_start'), history_window.get('forecast_end'))))}</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    
                    st.caption("The start/end controls choose the actual history shown on the graph. The forecast line starts after the latest actual data point.")

                    section_header("Load forecast chart", "Actual history, LSTM forecast, seasonal baseline, confidence band, and forecast-start marker.")
                    render_forecast_chart(forecast_df, history_df, history_window)

                    section_header("Operational details", "Use these numbers for reserve and comparison checks.")
                    detail_cols = st.columns(6)
                    detail_cols[0].metric("Peak", fmt_mw(forecast_peak["predicted_load_mw"]))
                    detail_cols[1].metric("Lowest", fmt_mw(forecast_low["predicted_load_mw"]))
                    detail_cols[2].metric("Avg. Uncertainty", f"+/- {fmt_mw(average_band / 2)}")
                    detail_cols[3].metric("Baseline Gap", fmt_mw(baseline_gap))
                    detail_cols[4].metric("Peak/Avg", f"{round(peak_to_avg, 2)}x" if isinstance(peak_to_avg, (int, float)) else "n/a")
                    backtest_label = fmt_pct(backtest_mape) if isinstance(backtest_mape, (int, float)) else "n/a"
                    detail_cols[5].metric("Backtest MAPE", backtest_label)

                    hourly_profile = result.get("hourly_profile", {})
                    if hourly_profile and any(v > 0 for v in hourly_profile.values() if isinstance(v, (int, float))):
                        st.markdown(
                            f"""
                            <div class="source-card">
                                <strong>Hourly profile:</strong>
                                Night (00-05): {fmt_mw(hourly_profile.get('night_00_05', 0))} |
                                Morning (06-11): {fmt_mw(hourly_profile.get('morning_06_11', 0))} |
                                Afternoon (12-17): {fmt_mw(hourly_profile.get('afternoon_12_17', 0))} |
                                Evening (18-23): {fmt_mw(hourly_profile.get('evening_18_23', 0))}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    explanation = result.get("explanation", {})
                    if explanation:
                        section_header("Operator brief", "Plain-language AI-style explanation generated from the forecast result.")
                        brief_html = format_operator_brief(explanation.get("text", "No explanation available."))
                        st.markdown(
                            f"""
                            <div class="explain-box">
                                {brief_html}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        
                        if explanation.get("note"):
                            st.warning(str(explanation["note"]))
                    for warning in result.get("warnings", []):
                        st.warning(warning)
                    if method != "lstm":
                        st.warning(
                            "LSTM model artifacts are not available yet, so this chart uses a seasonal naive fallback. "
                            "Train the model to enable neural-network forecasts."
                        )
            except requests.RequestException as exc:
                show_api_error(exc)
            except Exception as exc:
                show_dashboard_error(exc)
    else:
        if st.session_state.get("last_forecast_result") and st.session_state.get("last_forecast_payload"):
            st.info(
                "A saved forecast exists for "
                f"{forecast_payload_label(st.session_state.get('last_forecast_payload'))}. "
                "Run forecast to generate a fresh result for the current sidebar selection."
            )
        else:
            st.info("Run a forecast to see the active data source, uncertainty band, model quality, and plain-language explanation.")

with tabs[1]:
    section_header("Exploratory data analysis", "Validate the selected history window before trusting a forecast.")
    params = {
        "country_code": country_code,
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
    }
    if st.button("Load EDA"):
        with st.spinner("Fetching load data and generating EDA..."):
            try:
                summary = api_get("/eda/summary", params=params)
                plots = api_get("/eda/plots", params=params)
                stats = summary.get("summary_statistics", {})
                weekly = summary.get("weekly_seasonality", [])
                plot_payload = plots.get("plots", {})
                if not stats:
                    st.markdown(
                        '<div class="empty-state">No EDA statistics were returned for this range. Try a wider history window or confirm the data source is available.</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    stat_cols = st.columns(5)
                    stat_map = [
                        ("Mean", stats.get("mean")),
                        ("Median", stats.get("median")),
                        ("Minimum", stats.get("min")),
                        ("Maximum", stats.get("max")),
                        ("Std dev", stats.get("std")),
                    ]
                    for col, (label, value) in zip(stat_cols, stat_map):
                        with col:
                            ops_card(label, fmt_mw(value) if isinstance(value, (int, float)) else "n/a", "load statistic", "info")
                    with st.expander("Full summary statistics", expanded=False):
                        st.dataframe(pd.DataFrame([stats]), use_container_width=True)
                section_header("Weekly seasonality", "Average demand shape by weekday and hour.")
                if weekly:
                    st.markdown('<div class="table-panel">', unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame(weekly), use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.markdown(
                        '<div class="empty-state">No weekly pattern is available yet. Select at least several days of hourly data.</div>',
                        unsafe_allow_html=True,
                    )

                if not plot_payload:
                    st.markdown('<div class="empty-state">No EDA plots were returned for this range.</div>', unsafe_allow_html=True)
                for title, image_b64 in plot_payload.items():
                    if not image_b64:
                        continue
                    section_header(title.replace("_", " ").title(), "Base64-rendered PNG from the EDA module.")
                    st.image(base64.b64decode(image_b64), use_container_width=True)
            except Exception as exc:
                show_api_error(exc)

with tabs[2]:
    section_header("Monitoring", "Runtime health, model readiness, source warnings, and deployment checks.")
    monitoring = load_monitoring_status()
    if not monitoring:
        st.warning("Monitoring status is unavailable. Check the FastAPI service.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            ops_card("API status", str(monitoring.get("status", "unknown")).upper(), "FastAPI health", "ok" if monitoring.get("status") == "ok" else "bad")
        with col2:
            ops_card("Primary country", str(monitoring.get("primary_country_code", "unknown")), "assignment route", "info")
        with col3:
            ops_card("Primary model", "Ready" if monitoring.get("model_loaded") else "Pending", "loaded artifacts", tone_for_ready(bool(monitoring.get("model_loaded"))))
        with col4:
            error_count = len(monitoring.get("recent_errors", []))
            ops_card("Recent errors", str(error_count), "last API errors", "ok" if error_count == 0 else "warn")

        section_header("Deployment readiness", "Checklist for live demo and grading readiness.")
        checks = pd.DataFrame(monitoring.get("deployment_checks", []))
        if not checks.empty:
            checks["status"] = checks["ok"].apply(lambda value: "OK" if value else "Action needed")
            checks["tone"] = checks["ok"].apply(lambda value: "Ready" if value else "Review")
            st.dataframe(checks, use_container_width=True)

        warnings = monitoring.get("warnings", [])
        if warnings:
            section_header("Operational notices", "Non-blocking items that may affect source selection or deployment posture.")
            for warning in warnings:
                st.warning(warning)

        section_header("Last forecast", "Most recent request recorded by the API process.")
        last_forecast = monitoring.get("last_forecast")
        if last_forecast:
            st.dataframe(pd.DataFrame([last_forecast]), use_container_width=True)
        else:
            st.markdown('<div class="empty-state">No forecast has been requested since the API started.</div>', unsafe_allow_html=True)

        section_header("Recent errors", "Rolling API error buffer.")
        recent_errors = monitoring.get("recent_errors", [])
        if recent_errors:
            st.dataframe(pd.DataFrame(recent_errors), use_container_width=True)
        else:
            st.success("No recent API errors recorded.")

        section_header("Primary model metadata", "Training metrics and artifact details for the selected primary country.")
        primary_metadata = monitoring.get("model_metadata", {})
        if primary_metadata:
            meta_cols = st.columns(4)
            primary_metrics = primary_metadata.get("metrics", {})
            with meta_cols[0]:
                ops_card("Train rows", f"{primary_metadata.get('rows', 'n/a'):,}" if isinstance(primary_metadata.get("rows"), int) else "n/a", "hourly samples", "info")
            with meta_cols[1]:
                ops_card("MAPE", fmt_pct(primary_metrics.get("mape")) if isinstance(primary_metrics, dict) else "n/a", "test metric", "ok")
            with meta_cols[2]:
                ops_card("Data source", source_label(primary_metadata.get("data_source")), "training source", tone_for_source(primary_metadata.get("data_source")))
            with meta_cols[3]:
                ops_card("Epochs", str(primary_metadata.get("epochs_ran", "n/a")), "training epochs", "neutral")
            with st.expander("Raw metadata", expanded=False):
                st.json(primary_metadata)

        country_models = monitoring.get("country_models", {})
        if country_models:
            section_header("Country model readiness", "Per-country artifact status and training source.")
            readiness = [
                {
                    "country": country,
                    "status": status_text(bool(payload.get("model_ready"))),
                    "loaded": bool(payload.get("model_loaded")),
                    "artifact_country": payload.get("artifact_country_code"),
                    "method": payload.get("forecast_method"),
                    "data_class": catalog.get(country, {}).get("data_class", "unknown"),
                    "mape": payload.get("metadata", {}).get("metrics", {}).get("mape")
                    if isinstance(payload.get("metadata"), dict)
                    else None,
                    "training_source": payload.get("metadata", {}).get("data_source")
                    if isinstance(payload.get("metadata"), dict)
                    else None,
                }
                for country, payload in country_models.items()
            ]
            st.dataframe(pd.DataFrame(readiness), use_container_width=True)

with tabs[3]:
    section_header("About", "Architecture, data-source policy, assignment compliance, and runtime configuration.")
    about_cols = st.columns(4)
    with about_cols[0]:
        ops_card("Frontend", "Streamlit", "forecast-first operator console", "info")
    with about_cols[1]:
        ops_card("Backend", "FastAPI", "forecast, EDA, monitoring APIs", "info")
    with about_cols[2]:
        ops_card("Model", "LSTM", "168-hour sequence window", "ok")
    with about_cols[3]:
        ops_card("Primary source", "DE_LU", "ENTSO-E path with SMARD live support", "ok")

    source_status_card(
        source="entsoe_or_smard",
        trust="Assignment path",
        detail="DE_LU remains the primary grading country. ENTSO-E API/export is assignment-compliant; SMARD live DE-LU is a real Germany/Luxembourg operational source shown clearly when active.",
        freshness="transparent source labeling",
    )
    source_rows = [
        {"country": "DE_LU", "source": "ENTSO-E API", "status": "Best assignment source when token is configured", "type": "Real live"},
        {"country": "DE_LU", "source": "ENTSO-E export", "status": "Real assignment data from manual export", "type": "Real imported"},
        {"country": "DE_LU", "source": "SMARD", "status": "Real Germany/Luxembourg hourly load when active", "type": "Real dynamic"},
        {"country": "US48", "source": "EIA", "status": "Live U.S. demand with EIA_API_KEY", "type": "Real live"},
        {"country": "PK", "source": "CSV", "status": "Hidden by default; only historical demo unless official live Pakistan API is added", "type": "Demo"},
    ]
    section_header("Data-source matrix", "Every source is labeled honestly so real/live and demo paths are not confused.")
    st.dataframe(pd.DataFrame(source_rows), use_container_width=True)
    st.info(
        "For the most accurate real setup, use DE_LU with ENTSO-E API/export or US48 with EIA live data, "
        "then train a matching country LSTM. The app will not use a model from the wrong country."
    )
    section_header("Selected runtime", "Current model status and dashboard API routing.")
    st.json(model_status)
    config_cols = st.columns(3)
    with config_cols[0]:
        ops_card("Configured API", API_BASE_URL, "dashboard setting", "neutral")
    with config_cols[1]:
        ops_card("Active API", st.session_state.get("active_api_base_url", "not connected yet"), "last successful API URL", "ok" if st.session_state.get("active_api_base_url") else "warn")
    with config_cols[2]:
        ops_card("Fallback URLs", str(len(api_candidates())), "connection candidates", "info")
    with st.expander("API fallback order", expanded=False):
        st.code(", ".join(api_candidates()))
