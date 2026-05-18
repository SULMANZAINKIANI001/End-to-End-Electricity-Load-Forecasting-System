from __future__ import annotations

import argparse
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.data import DEFAULT_COUNTRY_CODE, fetch_load_data, load_pakistan_csv_dataset
from src.modeling import (
    SEQUENCE_LENGTH,
    artifact_paths_for_country,
    build_lstm_model,
    create_sequences,
    evaluate_forecast,
    residual_summary,
    save_metadata,
    seasonal_naive_forecast,
    split_sequences,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an LSTM electricity load forecasting model.")
    parser.add_argument("--country-code", default=DEFAULT_COUNTRY_CODE)
    parser.add_argument("--years", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--verbose", type=int, default=0, choices=[0, 1, 2], help="Keras training verbosity.")
    parser.add_argument("--start", default=None, help="Optional inclusive training start date.")
    parser.add_argument("--end", default=None, help="Optional exclusive training end date.")
    return parser.parse_args()


def main() -> None:
    from tensorflow import keras

    args = parse_args()
    country_code = args.country_code.upper()
    if args.start or args.end:
        end = pd.Timestamp(args.end) if args.end else pd.Timestamp(datetime.utcnow().date()) - pd.Timedelta(days=2)
        start = pd.Timestamp(args.start) if args.start else end - pd.Timedelta(days=365 * args.years)
        load_df = fetch_load_data(country_code, start, end, use_cache=False)
    elif country_code == "PK":
        load_df = load_pakistan_csv_dataset()
        cutoff = load_df.index.max() - pd.Timedelta(days=365 * args.years)
        load_df = load_df.loc[load_df.index >= cutoff]
    else:
        end = pd.Timestamp(datetime.utcnow().date()) - pd.Timedelta(days=2)
        start = end - pd.Timedelta(days=365 * args.years)
        load_df = fetch_load_data(country_code, start, end, use_cache=False)
    values = load_df["load_mw"].to_numpy(dtype=np.float32).reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(values)
    x_values, y_values = create_sequences(scaled_values, sequence_length=SEQUENCE_LENGTH, horizon=args.horizon)
    if len(x_values) < 100:
        raise RuntimeError("Not enough sequence samples for training. Select a longer date range.")

    x_train, y_train, x_val, y_val, x_test, y_test = split_sequences(x_values, y_values)
    model = build_lstm_model(sequence_length=SEQUENCE_LENGTH, horizon=args.horizon)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5),
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=args.verbose,
    )

    predictions = model.predict(x_test, verbose=0)
    metrics = evaluate_forecast(y_test, predictions, scaler)
    residuals = residual_summary(y_test, predictions, scaler)
    test_true = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    benchmark = seasonal_naive_forecast(values.ravel()[-(len(test_true) + 24) :], len(test_true), season_length=24)
    benchmark_eval = {
        "mae": round(float(np.mean(np.abs(test_true - benchmark[: len(test_true)]))), 3),
        "rmse": round(float(np.sqrt(np.mean((test_true - benchmark[: len(test_true)]) ** 2))), 3),
        "mape": round(float(np.nanmean(np.abs((test_true - benchmark[: len(test_true)]) / np.where(test_true == 0, np.nan, test_true))) * 100), 3),
    }

    model_path, scaler_path, metadata_path = artifact_paths_for_country(country_code)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    save_metadata(
        {
            "country_code": country_code,
            "data_source": load_df.attrs.get("source", "unknown"),
            "data_source_detail": load_df.attrs.get("source_detail", ""),
            "sequence_length": SEQUENCE_LENGTH,
            "horizon": args.horizon,
            "train_start": load_df.index.min().isoformat(),
            "train_end": load_df.index.max().isoformat(),
            "rows": int(len(load_df)),
            "samples": int(len(x_values)),
            "epochs_ran": int(len(history.history.get("loss", []))),
            "metrics": metrics,
            "baseline_metrics": benchmark_eval,
            "residuals": residuals,
            "trained_at": pd.Timestamp.utcnow().isoformat(),
        },
        path=metadata_path,
    )

    print(f"Saved model to {model_path}")
    print(f"Saved scaler to {scaler_path}")
    print(f"Test metrics: {metrics}")


if __name__ == "__main__":
    main()
