# Electricity Load Forecasting System Report

## Architecture

The system is split into five layers. `src/data.py` ingests ENTSO-E ActualTotalLoad time-series data, normalized ENTSO-E exports, SMARD dynamic Germany/Luxembourg data, optional EIA US48 demand, or optional Pakistan CSV data. `src/eda.py` creates reusable analysis outputs such as summary statistics, weekly seasonality, load curves, and daily boxplots. `src/train.py` trains and saves country-specific LSTM model and scaler artifacts into `models/<COUNTRY_CODE>/`. `src/api.py` exposes FastAPI endpoints consumed by the Streamlit dashboard in `app.py`. `src/explain.py` and `src/monitoring.py` add forecast explanations, readiness checks, and runtime status.

The assignment-compliant live dataset is Germany/Luxembourg (`DE_LU`) because ENTSO-E is a European electricity transparency platform and does not provide Pakistan load data. `DE_LU` uses ENTSO-E API first, a normalized manual ENTSO-E export second, and SMARD dynamic fallback third while REST token access is pending. To make the web application useful in Pakistan, the backend also supports `PK` through a local CSV dataset configured with `PAKISTAN_LOAD_CSV`. The application is Docker-ready with separate API and dashboard services, and can be deployed on Railway or Streamlit Cloud plus an API host.

## Preprocessing and EDA

Raw load data is converted into a single `load_mw` time series with timezone-aware timestamps. ENTSO-E and SMARD countries use European time zones, EIA US48 data uses UTC, and Pakistan CSV data uses `Asia/Karachi`. The pipeline sorts timestamps, removes duplicates, resamples to hourly frequency, and fills missing values using forward fill followed by backward fill for leading gaps. This creates a regular hourly series suitable for LSTM training.

EDA includes mean, median, minimum, maximum, standard deviation, quartiles, missing values, and row count. Weekly seasonality is calculated from day-of-week and hour-of-day averages. Visual outputs are returned as base64 PNG strings so that the API can serve them directly to the frontend.

## Model Details and Evaluation

The model uses a sliding window length of `168` hours, representing one full week of hourly observations. Load values are scaled with `MinMaxScaler`. The neural network contains two LSTM layers with dropout and a dense forecast head. Training uses chronological train, validation, and test splits to avoid leakage, and early stopping restores the best validation weights. The API also computes a seasonal naive baseline and residual-based confidence bands so users can compare model output with a simple operational benchmark.

Evaluation reports MAE, RMSE, and MAPE on the held-out test set. Saved artifacts include the Keras model, the scaler, and metadata such as country code, active data source, sequence length, horizon, training period, baseline metrics, residual summary, and evaluation metrics. Country-specific artifacts prevent a Pakistan model from being used for Germany/Luxembourg or any other region.

## Challenges and Future Improvements

The main challenge is that live ENTSO-E data requires an API token and may have temporary availability gaps. The implemented fallback chain keeps Germany/Luxembourg dynamic through SMARD and supports manual ENTSO-E exports until token approval. Pakistan is a separate challenge because the assignment source does not provide Pakistan load, so the implemented Pakistan mode depends on a reliable historical CSV export. Forecast quality can be improved by adding weather, calendar holidays, prices, and generation mix features. Future work should include scheduled retraining, persistent feature storage, alerting when live data is stale, and comparison against more statistical baselines.
