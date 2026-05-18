# Electricity Load Forecasting System

An end-to-end web application for forecasting hourly electricity load with real public power-system data, FastAPI, Streamlit, TensorFlow/Keras LSTM models, EDA, monitoring, Docker, and plain-language AI-style forecast explanations.

The project is built for the assignment requirement around ENTSO-E ActualTotalLoad data. `DE_LU` is the primary assignment country. The app can use ENTSO-E live API data, a normalized ENTSO-E export, or SMARD real Germany/Luxembourg hourly load data. `US48` is also supported through EIA Open Data as a second real live data source. Pakistan CSV support exists only as an optional historical demo and is hidden by default.

## What This App Does

The system lets a user select a country, date range, and forecast horizon, then returns:

- actual historical load data used by the model
- future load forecast
- LSTM forecast line
- seasonal baseline comparison
- confidence band from model residuals
- source and freshness information
- model metrics: MAE, RMSE, MAPE
- simple explanation of what the forecast means
- monitoring status for deployment readiness

The dashboard is designed so non-technical users can understand:

- what is happening to demand
- when the peak is expected
- whether the forecast is reliable
- what data source was used
- whether the data is live, dynamic, imported, or demo
- what operational action to take

## Current Model Results

These models are already trained and available in `models/`.

| Country | Runtime Source | Training Rows | Forecast Method | LSTM MAPE | Baseline MAPE | Status |
| --- | --- | ---: | --- | ---: | ---: | --- |
| `DE_LU` | SMARD live DE-LU | 17,520 | LSTM | 5.698% | 21.066% | Ready |
| `US48` | EIA live API | 17,521 | LSTM | 3.968% | 12.570% | Ready |
| `PK` | Historical CSV demo | local CSV | LSTM/demo | 11.361% | varies | Hidden by default |

Important: For maximum assignment compliance, add an `ENTSOE_API_KEY` or import a manual ENTSO-E ActualTotalLoad export for `DE_LU`, then retrain the `DE_LU` model.

## How It Works

```text
User selects country/date/horizon in Streamlit
        |
        v
Streamlit calls FastAPI /forecast
        |
        v
FastAPI fetches real hourly load data
        |
        v
Data is normalized to timezone-aware hourly load_mw
        |
        v
Country-specific model artifacts are loaded
        |
        v
LSTM predicts future hourly load
        |
        v
Seasonal baseline and confidence bands are calculated
        |
        v
API returns history, forecast, metrics, source info, and explanation
        |
        v
Dashboard shows actual history + future forecast + monitoring
```

## Data Sources

| Country | Source Priority | Type | Notes |
| --- | --- | --- | --- |
| `DE_LU` | ENTSO-E API -> ENTSO-E export -> SMARD live DE-LU | Assignment/real dynamic | Primary assignment path |
| `FR`, `NL`, `BE` | ENTSO-E API | Assignment/live | Requires `ENTSOE_API_KEY` and training |
| `US48` | EIA Open Data API | Real live | Extra real live source |
| `PK` | Local CSV | Historical demo | Not live ENTSO-E/EIA data |

Runtime mode:

- Default dashboard mode shows only countries with ready LSTM models.
- Historical demo countries are hidden unless `ENABLE_DEMO_COUNTRIES=true`.
- Countries without model artifacts are hidden unless the user enables "Show countries needing training".

## Model Design

The forecasting model follows the assignment requirements:

- scaling: `MinMaxScaler`
- sequence length: `168` hours, one week of hourly history
- forecast horizon: configurable, default `24` hours
- architecture: two LSTM layers with dropout
- split: chronological train, validation, test
- callbacks: early stopping and learning-rate reduction
- metrics: MAE, RMSE, MAPE
- benchmark: seasonal naive baseline
- confidence band: residual-based uncertainty interval
- artifacts saved per country

Artifact paths:

```text
models/<COUNTRY_CODE>/load_lstm.keras
models/<COUNTRY_CODE>/load_scaler.joblib
models/<COUNTRY_CODE>/metadata.json
```

Example:

```text
models/DE_LU/load_lstm.keras
models/DE_LU/load_scaler.joblib
models/DE_LU/metadata.json
```

The API will not use a model trained for the wrong country. If matching artifacts are missing, it returns a clearly labeled `seasonal_naive_fallback` instead of silently giving invalid LSTM output.

## Dashboard Screens

### Forecast

Shows:

- model readiness cards
- selected source profile
- actual history window
- actual data used
- future forecast period
- actual load history line
- LSTM forecast line
- seasonal baseline line
- confidence band
- peak, low, average load, baseline gap
- easy forecast explanation

The selected start and end dates are the historical input window. The forecast starts after the latest actual load row.

The sidebar uses a **History Window** control for safer forecasting:

- `30 days`, `90 days`, `1 year`, and `2 years` presets automatically calculate the start date.
- `Custom` unlocks manual start/end dates.
- The UI blocks ranges longer than `MAX_HISTORY_DAYS`; the graph shows the selected history window, while the LSTM engine uses the latest `FORECAST_HISTORY_DAYS` days from that window for runtime speed.
- Live countries default to the latest safe date (`today - 2 days`), while CSV/imported sources use their actual available min/max dates.

### EDA

Shows:

- summary statistics
- weekly seasonality table
- load curve plot
- weekly seasonality plot
- daily boxplot

### Monitoring

Shows:

- API status
- primary country
- model readiness
- recent errors
- deployment checklist
- last forecast request
- model metadata
- readiness per country

### About

Explains:

- architecture
- source types
- ENTSO-E compliance
- EIA support
- Pakistan caveat
- selected model status

## AI Forecast Explanation

The app supports two explanation modes.

### Mode 1: AI Model Explanation

When `OPENCODE_API_KEY` is configured, the API sends a compact forecast JSON to an OpenAI-compatible chat model:

```env
OPENCODE_API_KEY=your_key_here
OPENCODE_BASE_URL=https://opencode.ai/zen/v1
OPENCODE_MODEL=nemotron-3-super-free
AI_EXPLANATION_TIMEOUT_SECONDS=5
AI_EXPLANATION_MAX_TOKENS=900
AI_EXPLANATION_RETRIES=0
AI_EXPLANATION_ON_FORECAST=false
AI_EXPLANATION_COOLDOWN_SECONDS=60
```

The JSON includes the selected country, data source, selected graph history, LSTM input window, forecast peak/low/average, uncertainty band, model metrics, seasonal baseline comparison, and runtime warnings. The model is instructed to use only those values and not invent weather, outages, or grid events.

The external AI call is separate from the main forecast so the dashboard does not get stuck on `Fetching recent load and generating forecast...`. In the sidebar, keep **Try external AI explanation** off for the fastest demo. Turn it on only when you want the OpenCode model to rewrite the explanation. If the provider is slow, unsupported, or rate-limited, the API falls back to the built-in explanation, shows the reason, and cools down before trying again.

### Mode 2: Built-In Fallback

If the model key is missing or the provider fails, the app uses a built-in deterministic explanation engine. It works without external AI keys and keeps the dashboard reliable.

Each forecast explanation includes:

- Forecast brief
- Operator guidance / how to read the forecast
- Data used
- Model confidence
- Caveats

Example:

```text
Forecast brief
Status: demand is rising over the next 24 hours.
Peak watch: 500.6 GW around May 16, 2026 22:00 UTC.

Data used
Graph history shown: May 16, 2025 to May 16, 2026.
LSTM input window: latest 45 days from that selected range.

Model confidence
Accuracy check: MAPE is 3.97%, and the LSTM beats the seasonal baseline.
```

Never commit real API keys.

## API Endpoints

Base URL locally:

```text
http://localhost:8000
```

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | API health and primary model metadata |
| `GET` | `/settings` | Frontend limits such as max history days and forecast horizon |
| `GET` | `/countries` | Supported countries and source metadata |
| `GET` | `/countries?forecast_ready_only=true` | Only countries with ready LSTM models |
| `GET` | `/model/status?country_code=DE_LU` | Country model artifact status |
| `GET` | `/monitoring/status` | Deployment, model, and runtime monitoring |
| `GET` | `/data/range?country_code=PK` | Available local CSV data range |
| `POST` | `/forecast` | Forecast load with model/source metadata |
| `POST` | `/explain` | Generate explanation for a forecast payload |
| `GET` | `/eda/summary` | Summary statistics and seasonality |
| `GET` | `/eda/plots` | Base64 PNG EDA plots |

### Forecast Request

```json
{
  "country_code": "US48",
  "start": "2025-04-15",
  "end": "2026-05-15",
  "horizon": 24
}
```

`start` and `end` define the historical input window. Date-only `end` values are treated as inclusive user dates. Internally, the API converts the end date to an exclusive timestamp.

### Forecast Response Includes

```json
{
  "country_code": "US48",
  "horizon": 24,
  "forecast_method": "lstm",
  "data_source": {
    "source": "eia_live",
    "source_detail": "EIA hourly RTO demand, respondent US48, type D",
    "latest_timestamp": "2026-05-16T05:00:00+05:00",
    "rows": 9505
  },
  "history_window": {
    "selected_start": "2025-04-15",
    "selected_end": "2026-05-15",
    "selected_end_inclusive": true,
    "actual_start": "2025-04-15T05:00:00+05:00",
    "actual_end": "2026-05-16T05:00:00+05:00",
    "forecast_start": "2026-05-16T06:00:00+05:00",
    "forecast_end": "2026-05-17T05:00:00+05:00"
  },
  "history": [],
  "forecast": [],
  "model_metadata": {},
  "explanation": {},
  "warnings": []
}
```

## Project Structure

```text
.
├── app.py
├── Dockerfile
├── docker-compose.yml
├── Procfile
├── runtime.txt
├── requirements.txt
├── requirements-dashboard.txt
├── README.md
├── report.md
├── data/
│   └── pakistan_load.csv
├── models/
│   ├── DE_LU/
│   ├── US48/
│   └── PK/
├── notebooks/
├── scripts/
│   ├── normalize_entsoe_export.py
│   └── normalize_ntdc.py
├── src/
│   ├── api.py
│   ├── config.py
│   ├── data.py
│   ├── eda.py
│   ├── explain.py
│   ├── logging_config.py
│   ├── modeling.py
│   ├── monitoring.py
│   └── train.py
└── tests/
```

## Important Files

| File | Purpose |
| --- | --- |
| `src/data.py` | Fetches and normalizes ENTSO-E, SMARD, EIA, and CSV load data |
| `src/eda.py` | Summary stats, weekly seasonality, base64 PNG plots |
| `src/modeling.py` | Sequence creation, LSTM model, evaluation, artifact loading |
| `src/train.py` | Country-specific LSTM training pipeline |
| `src/api.py` | FastAPI backend and public endpoints |
| `src/explain.py` | Built-in and optional AI forecast explanations |
| `src/monitoring.py` | Readiness checks, warnings, last forecast, recent errors |
| `app.py` | Streamlit dashboard |
| `Dockerfile` | Container image for API or dashboard |
| `docker-compose.yml` | Runs API and dashboard together |
| `report.md` | Two-page report draft |

## Environment Variables

Copy `.env.example` to `.env`:

```powershell
copy .env.example .env
```

Main variables:

| Variable | Required | Purpose |
| --- | --- | --- |
| `ENTSOE_API_KEY` | Recommended | Live ENTSO-E Transparency Platform access |
| `EIA_API_KEY` | Required for `US48` live | EIA Open Data key |
| `API_BASE_URL` | Dashboard only | FastAPI URL for Streamlit |
| `PRIMARY_COUNTRY_CODE` | Optional | Default primary country, usually `DE_LU` |
| `ENABLE_DEMO_COUNTRIES` | Optional | Set `true` to show Pakistan CSV demo |
| `PAKISTAN_LOAD_CSV` | Optional | Path to Pakistan CSV demo data |
| `DE_LU_LOAD_CSV` | Optional | Normalized ENTSO-E export CSV path |
| `MODELS_DIR` | Optional | Model artifact directory |
| `OPENCODE_API_KEY` | Optional | External LLM explanation provider |
| `CORS_ORIGINS` | Production | Allowed frontend origins |
| `MAX_REQUEST_BYTES` | Optional | Reject oversized API requests, default `1000000` |
| `ENABLE_TRAIN_ENDPOINT` | Optional | Enables protected `/train`; keep `false` for demos |
| `TRAIN_API_TOKEN` | Required only if `/train` enabled | Admin token for training endpoint |
| `MAX_TRAIN_EPOCHS` | Optional | Upper bound for API-triggered training jobs |
| `API_RETRY_ATTEMPTS` | Optional | Retry count for ENTSO-E, SMARD, and EIA calls |
| `API_RETRY_BACKOFF_SECONDS` | Optional | Base retry backoff for public API calls |
| `MAX_EXPLAIN_FORECAST_POINTS` | Optional | Maximum forecast points accepted by `/explain` |

Do not commit `.env`.

## Setup

Create and activate a Python environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install API and training dependencies:

```powershell
python -m pip install -r requirements.txt
```

Install dashboard dependencies:

```powershell
python -m pip install -r requirements-dashboard.txt
```

TensorFlow is most reliable with Python 3.10 on Windows:

```powershell
py -3.10 -m venv .venv310
.\.venv310\Scripts\python.exe -m pip install -r requirements.txt
```

## Run Locally Without Docker

Terminal 1:

```powershell
.\.venv310\Scripts\python.exe -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

Terminal 2:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

Open:

```text
http://localhost:8501
```

## Run With Docker

```powershell
docker compose up -d --build
```

Services:

```text
FastAPI:   http://localhost:8000
Streamlit: http://localhost:8501
```

The container runs as a non-root user and includes health checks. The dashboard waits for the API health check before starting.

Stop services:

```powershell
docker compose down
```

## Training Commands

Train the primary assignment country:

```powershell
.\.venv310\Scripts\python.exe -m src.train --country-code DE_LU --years 2 --horizon 24 --epochs 30 --verbose 0
```

Train or refresh the EIA live model:

```powershell
.\.venv310\Scripts\python.exe -m src.train --country-code US48 --years 2 --horizon 24 --epochs 5 --verbose 0
```

Train Pakistan demo data:

```powershell
.\.venv310\Scripts\python.exe -m src.train --country-code PK --years 2 --horizon 24 --epochs 30 --verbose 0
```

Train a fixed date range:

```powershell
.\.venv310\Scripts\python.exe -m src.train --country-code DE_LU --start 2024-01-01 --end 2026-01-01 --horizon 24 --epochs 30 --verbose 0
```

Protected API training is available for production completeness but is disabled by default:

```env
ENABLE_TRAIN_ENDPOINT=false
TRAIN_API_TOKEN=replace-with-long-random-secret
```

When enabled, call `POST /train` with header `X-Admin-Token: <TRAIN_API_TOKEN>`. Check progress with `GET /train/status/{job_id}`. Do not expose this endpoint without authentication because training is expensive.

## ENTSO-E Manual Export

If ENTSO-E API token approval is pending:

1. Go to `https://transparency.entsoe.eu`.
2. Open Load -> Total Load - Day Ahead / Actual.
3. Select bidding zone `DE_LU`.
4. Select at least two years of hourly data.
5. Export CSV/XLSX.
6. Save it as `data/de_lu_load_raw.csv` or `data/de_lu_load_raw.xlsx`.
7. Normalize it:

```powershell
.\.venv310\Scripts\python.exe scripts\normalize_entsoe_export.py --input data\de_lu_load_raw.csv --output data\de_lu_load.csv
```

Then retrain:

```powershell
.\.venv310\Scripts\python.exe -m src.train --country-code DE_LU --years 2 --horizon 24 --epochs 30 --verbose 0
```

## Pakistan Demo Data

Pakistan is not covered by ENTSO-E or EIA. In this project, Pakistan is supported only as a historical CSV demo.

Expected CSV format:

```csv
timestamp,load_mw
2024-01-01 00:00:00,19250
2024-01-01 01:00:00,18740
```

Enable it in `.env` only when needed:

```env
ENABLE_DEMO_COUNTRIES=true
PAKISTAN_LOAD_CSV=data/pakistan_load.csv
```

## Testing

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Current status:

```text
12 passed
```

Compile/import check:

```powershell
.\.venv310\Scripts\python.exe -m compileall app.py src scripts tests
```

## Deployment

### API on Railway

1. Push repository to GitHub.
2. Create a Railway project.
3. Use Docker deployment.
4. Add environment variables:
   - `ENTSOE_API_KEY`
   - `EIA_API_KEY`
   - `CORS_ORIGINS`
   - optional `OPENCODE_API_KEY`
5. Deploy and copy the API URL.

### Dashboard on Streamlit Cloud

1. Deploy `app.py`.
2. Add `API_BASE_URL` pointing to the Railway API URL.
3. Add any dashboard-specific environment variables.

## Submission Checklist

- GitHub repository link
- live application URL
- `README.md`
- `report.md`
- Dockerfile
- trained model artifacts or training instructions
- no hardcoded API keys
- ENTSO-E token or ENTSO-E export for strongest assignment compliance

## Notes And Caveats

- `DE_LU` is the primary assignment country.
- `SMARD` is real Germany/Luxembourg hourly load data, but it is not the same endpoint as the ENTSO-E Transparency API.
- For maximum marks, configure `ENTSOE_API_KEY` or use manual ENTSO-E export before final submission.
- `US48` uses real EIA live data and is included as an additional real source.
- `PK` is historical demo only.
- Date-only `end` values are treated as inclusive user dates.
- The graph shows actual history first, then the future forecast after the latest actual row.
- Rotate any API key that was ever pasted in chat or screenshots before public deployment.
