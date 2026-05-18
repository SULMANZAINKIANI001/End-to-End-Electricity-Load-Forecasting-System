import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.data import (
    SUPPORTED_COUNTRIES,
    DataFetchError,
    fetch_eia_us48_load,
    fetch_load_data,
    fetch_smard_de_lu_load,
    load_pakistan_csv_dataset,
    normalize_entsoe_export_file,
    public_supported_countries,
    _normalize_load_response,
    _to_timestamp,
)
from src.eda import build_eda_payload, build_plot_payload
from src.explain import build_deterministic_explanation
from src.modeling import SEQUENCE_LENGTH, artifact_paths_for_country, confidence_bounds, create_sequences, iterative_forecast, seasonal_naive_forecast
from src.monitoring import deployment_checks
from src.safety import sanitize_for_json
from src.api import _compute_forecast_summary, _compute_recent_accuracy, app, app_settings


def test_create_sequences_uses_weekly_window() -> None:
    values = np.arange(220, dtype=np.float32).reshape(-1, 1)
    x_values, y_values = create_sequences(values, sequence_length=SEQUENCE_LENGTH, horizon=24)
    assert x_values.shape == (29, SEQUENCE_LENGTH, 1)
    assert y_values.shape == (29, 24)


def test_normalize_load_response_returns_hourly_timezone_aware_frame() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="30min", tz="Europe/Brussels")
    series = pd.Series([1.0, 3.0, np.nan, 5.0], index=index)
    frame = _normalize_load_response(series)
    assert frame.index.tz is not None
    assert frame.index.freqstr == "h"
    assert frame["load_mw"].isna().sum() == 0


def test_normalize_load_response_rejects_empty_data() -> None:
    with pytest.raises(DataFetchError):
        _normalize_load_response(pd.Series(dtype=float))


def test_eda_payloads() -> None:
    index = pd.date_range("2024-01-01", periods=48, freq="h", tz="Europe/Brussels")
    load_df = pd.DataFrame({"load_mw": np.linspace(100, 200, len(index))}, index=index)
    payload = build_eda_payload(load_df)
    plots = build_plot_payload(load_df)
    assert payload["summary_statistics"]["count"] == 48
    assert len(payload["weekly_seasonality"]) > 0
    assert set(plots) == {"load_curve", "weekly_seasonality", "daily_boxplot"}
    assert all(isinstance(value, str) and len(value) > 100 for value in plots.values())


def test_fetch_load_data_supports_pakistan_csv(tmp_path, monkeypatch) -> None:
    csv_path = tmp_path / "pakistan_load.csv"
    csv_path.write_text(
        "timestamp,load_mw\n"
        "2024-01-01 00:00:00,100\n"
        "2024-01-01 01:00:00,110\n"
        "2024-01-01 02:00:00,120\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PAKISTAN_LOAD_CSV", str(csv_path))
    frame = fetch_load_data("PK", "2024-01-01", "2024-01-01 03:00:00")
    assert len(frame) == 3
    assert str(frame.index.tz) == "Asia/Karachi"
    assert frame["load_mw"].tolist() == [100, 110, 120]
    full_frame = load_pakistan_csv_dataset()
    assert len(full_frame) == 3


def test_supported_countries_include_pakistan_and_entsoe_regions() -> None:
    assert SUPPORTED_COUNTRIES["PK"]["source"] == "csv"
    assert SUPPORTED_COUNTRIES["DE_LU"]["source"] == "entsoe_or_smard"
    assert SUPPORTED_COUNTRIES["US48"]["source"] == "eia"
    assert "PK" not in public_supported_countries(include_demo=False)
    assert "PK" in public_supported_countries(include_demo=True)


def test_entsoe_export_normalization_handles_mtu_and_actual_load(tmp_path) -> None:
    raw_path = tmp_path / "de_lu_load_raw.csv"
    output_path = tmp_path / "de_lu_load.csv"
    raw_path.write_text(
        "MTU,Actual Total Load [MW] - BZN|DE-LU,Day-ahead Total Load Forecast [MW]\n"
        "01.01.2024 00:00 - 01.01.2024 01:00,50000,51000\n"
        "01.01.2024 01:00 - 01.01.2024 02:00,49000,50000\n",
        encoding="utf-8",
    )
    frame = normalize_entsoe_export_file(raw_path, output_path)
    assert output_path.exists()
    assert frame.index.tz is not None
    assert frame["load_mw"].tolist() == [50000, 49000]


def test_smard_connector_returns_timezone_aware_hourly_frame(monkeypatch) -> None:
    start = pd.Timestamp("2024-01-01", tz="Europe/Brussels")
    start_ms = int(start.tz_convert("UTC").timestamp() * 1000)

    def fake_smard_json(path: str) -> dict:
        if path.endswith("index_hour.json"):
            return {"timestamps": [start_ms]}
        return {
            "series": [
                [start_ms, 50000.0],
                [start_ms + 3600_000, 51000.0],
            ]
        }

    monkeypatch.setattr("src.data._smard_json", fake_smard_json)
    frame = fetch_smard_de_lu_load(start, start + pd.Timedelta(hours=2))
    assert frame.attrs["source"] == "smard_dynamic_fallback"
    assert str(frame.index.tz) == "Europe/Brussels"
    assert frame["load_mw"].tolist() == [50000.0, 51000.0]


def test_eia_connector_paginates_hourly_rows(monkeypatch) -> None:
    from dataclasses import replace

    from src.config import settings

    class FakeResponse:
        def __init__(self, offset: int) -> None:
            self.offset = offset

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            if self.offset >= 5000:
                rows = [
                    {"period": "2024-07-27T08", "value": 150.0},
                    {"period": "2024-07-27T09", "value": 160.0},
                ]
            else:
                start = pd.Timestamp("2024-01-01T00:00:00Z")
                rows = [
                    {"period": (start + pd.Timedelta(hours=index)).strftime("%Y-%m-%dT%H"), "value": 100.0 + index}
                    for index in range(5000)
                ]
            return {"response": {"total": 5002, "data": rows}}

    def fake_get(_url: str, params: list[tuple[str, str]], timeout: int) -> FakeResponse:
        offset = int(dict(params).get("offset", 0))
        return FakeResponse(offset)

    monkeypatch.setattr("src.data.settings", replace(settings, eia_api_key="test-key"))
    monkeypatch.setattr("src.data.requests.get", fake_get)
    frame = fetch_eia_us48_load(pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2024-08-01T00:00:00Z"))
    assert len(frame) == 5002
    assert frame.attrs["source"] == "eia_live"


def test_per_country_artifact_paths_are_isolated() -> None:
    de_model, _, de_metadata = artifact_paths_for_country("DE_LU")
    pk_model, _, pk_metadata = artifact_paths_for_country("PK")
    assert de_model != pk_model
    assert de_metadata.parent.name == "DE_LU"
    assert pk_metadata.parent.name == "PK"


def test_confidence_bounds_and_baseline_forecast() -> None:
    predictions = np.asarray([100.0, 110.0, 120.0])
    lower, upper = confidence_bounds(predictions, {"residuals": {"p90_abs_residual_mw": 10.0}})
    assert lower[0] == 90.0
    assert upper[0] == 110.0
    assert np.isfinite(lower).all()
    assert np.isfinite(upper).all()
    assert (upper >= lower).all()
    long_predictions = np.ones(24) * 100_000.0
    long_lower, long_upper = confidence_bounds(long_predictions, {"residuals": {"p90_abs_residual_mw": 39_077.0}})
    avg_half_width = float(np.mean(long_upper - long_lower) / 2)
    assert avg_half_width < 50_000.0
    baseline = seasonal_naive_forecast(np.asarray([1.0, 2.0, 3.0]), horizon=5, season_length=3)
    assert baseline.tolist() == [1.0, 2.0, 3.0, 1.0, 2.0]


def test_deterministic_explanation_and_monitoring_checks() -> None:
    payload = {
        "forecast_method": "lstm",
        "forecast": [
            {"timestamp": "2024-01-01T00:00:00", "predicted_load_mw": 100.0, "lower_bound_mw": 90.0, "upper_bound_mw": 110.0},
            {"timestamp": "2024-01-01T01:00:00", "predicted_load_mw": 120.0, "lower_bound_mw": 105.0, "upper_bound_mw": 135.0},
        ],
        "data_source": {"source": "eia_live", "rows": 48, "latest_timestamp": "2024-01-01T01:00:00Z"},
        "history_window": {
            "selected_start": "2023-12-01",
            "selected_end": "2024-01-01",
            "actual_start": "2023-12-01T00:00:00Z",
            "actual_end": "2024-01-01T01:00:00Z",
            "model_input_start": "2023-12-18T00:00:00Z",
            "model_input_end": "2024-01-01T01:00:00Z",
            "model_input_rows": 337,
            "history_capped": True,
        },
        "model_metadata": {"metrics": {"mape": 6.0}, "rows": 1000},
    }
    explanation = build_deterministic_explanation(payload)
    assert "forecast" in explanation.lower()
    assert "Graph history shown" in explanation
    assert "LSTM input window" in explanation
    checks = deployment_checks(model_loaded=True, metadata={"country_code": "DE_LU"})
    assert any(check["name"] == "LSTM model loaded" and check["ok"] for check in checks)


def test_settings_endpoint_payload_has_ui_limits() -> None:
    payload = app_settings()
    assert payload["max_history_days"] >= payload["forecast_history_days"]
    assert payload["default_history_days"] <= payload["max_history_days"]
    assert payload["max_forecast_horizon"] >= payload["default_forecast_horizon"]


def test_json_sanitizer_converts_non_finite_and_pandas_values() -> None:
    payload = {
        "nan": float("nan"),
        "inf": float("inf"),
        "np": np.float64("-inf"),
        "ts": pd.Timestamp("2024-01-01T00:00:00Z"),
        "items": [np.int64(2), pd.NaT],
    }
    clean = sanitize_for_json(payload)
    assert clean["nan"] is None
    assert clean["inf"] is None
    assert clean["np"] is None
    assert clean["ts"] == "2024-01-01T00:00:00+00:00"
    assert clean["items"] == [2, None]


def test_to_timestamp_handles_dst_ambiguous_and_nonexistent_times() -> None:
    ambiguous = _to_timestamp("2024-10-27 02:30:00", "Europe/Brussels")
    nonexistent = _to_timestamp("2024-03-31 02:30:00", "Europe/Brussels")
    assert ambiguous.tzinfo is not None
    assert nonexistent.tzinfo is not None
    assert nonexistent.hour == 3


def test_modeling_guards_nonfinite_and_negative_forecasts() -> None:
    class FakeScaler:
        def transform(self, values):
            return values / 100.0

        def inverse_transform(self, values):
            return values * 100.0

    class FakeModel:
        def predict(self, _window, verbose=0):
            return np.asarray([[np.nan, -0.5, 0.2]])

    history = np.linspace(100, 200, SEQUENCE_LENGTH)
    history[3] = np.nan
    forecast = iterative_forecast(FakeModel(), FakeScaler(), history, horizon=3, sequence_length=SEQUENCE_LENGTH)
    assert np.isfinite(forecast).all()
    assert (forecast >= 0).all()
    lower, upper = confidence_bounds(np.asarray([np.nan, -5.0, 20.0]), {"residuals": {"p90_abs_residual_mw": float("inf")}})
    assert np.isfinite(lower).all()
    assert np.isfinite(upper).all()
    assert (lower >= 0).all()
    assert (upper >= lower).all()


def test_forecast_summary_and_recent_accuracy_have_stable_keys() -> None:
    summary = _compute_forecast_summary(np.asarray([np.nan, float("inf")]), [], np.asarray([0, 0]))
    assert summary["available"] is False
    assert "average_load_mw" in summary
    accuracy = _compute_recent_accuracy(pd.DataFrame({"load_mw": [np.nan] * 10}))
    assert accuracy["available"] is False
    assert "mape_pct" in accuracy


def test_api_explain_schema_and_train_security() -> None:
    client = TestClient(app)
    disabled = client.post("/train", json={"country_code": "DE_LU"}, headers={"X-Admin-Token": "bad"})
    assert disabled.status_code == 403
    too_large = [{"timestamp": "2024-01-01T00:00:00Z", "predicted_load_mw": 1.0}] * 400
    invalid = client.post("/explain", json={"forecast": too_large})
    assert invalid.status_code == 422


def test_request_size_middleware_rejects_large_body() -> None:
    client = TestClient(app)
    response = client.post("/explain", content=b"x" * 1_100_000, headers={"content-type": "application/json"})
    assert response.status_code == 413


def test_health_endpoint_shape() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_global_error_handler_returns_safe_json(monkeypatch) -> None:
    def explode(_country_code: str):
        raise RuntimeError("private stack detail")

    monkeypatch.setattr("src.api.get_available_data_range", explode)
    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/data/range", params={"country_code": "DE_LU"})
    assert response.status_code == 500
    body = response.json()
    assert body["error_type"] == "internal_error"
    assert "private stack detail" not in body["detail"]
