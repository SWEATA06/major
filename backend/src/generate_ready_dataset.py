"""
Generate a ready-to-run synthetic dataset for AutoScale Intelligence AI.

A fresh checkout of this repo has no data (the data/ folder is gitignored), so
the backend has nothing to simulate on. This script produces
`data/final/final_dataset.csv` with EXACTLY the columns the application and
`backend/src/feature_engineering.py` expect, so the app can run end-to-end in
either Mock AI mode or (once TensorFlow is available) with trained models.

Columns produced:
  cpu_usage, memory_usage, disk_io, network_usage, request_rate, queue_length,
  latency, error_rate, cpu_change_rate, moving_average_cpu,
  hour_of_day, day_of_week, minute_bucket, failure_label

Usage:
  PYTHONPATH=$(pwd) python backend/src/generate_ready_dataset.py
"""
import os
import numpy as np
import pandas as pd


def generate(num_samples=5000, output_path=None):
    if output_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        out_dir = os.path.join(base_dir, "data", "final")
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "final_dataset.csv")
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    np.random.seed(42)

    # 5-minute cadence timestamps drive realistic daily seasonality.
    timestamps = pd.date_range(start="2026-01-01", periods=num_samples, freq="5min")
    hours = timestamps.hour.to_numpy()

    # Daily CPU cycle (low at night, higher midday) + noise.
    base_cpu = 40 + 25 * np.sin(np.pi * (hours - 6) / 12)
    cpu_usage = base_cpu + np.random.normal(0, 5, num_samples)

    # Inject ~5% high-load anomalies so failure/overload events actually occur.
    anomaly_idx = np.random.choice(num_samples, int(num_samples * 0.05), replace=False)
    cpu_usage[anomaly_idx] += np.random.normal(30, 10, len(anomaly_idx))
    cpu_usage = np.clip(cpu_usage, 0, 100)

    # Correlated metrics.
    memory_usage = np.clip(cpu_usage * 0.7 + np.random.normal(10, 5, num_samples), 0, 100)
    request_rate = np.maximum(cpu_usage * 10 + np.random.normal(50, 20, num_samples), 10)
    network_usage = np.maximum(request_rate * 2.5 + np.random.normal(10, 5, num_samples), 0)
    disk_io = np.maximum(request_rate * 0.8 + np.random.normal(5, 2, num_samples), 0)

    queue_length = np.where(
        cpu_usage > 80,
        np.random.poisson(50, num_samples),
        np.random.poisson(5, num_samples),
    )

    latency = np.where(
        cpu_usage > 85,
        np.random.normal(500, 100, num_samples),
        np.random.normal(50, 10, num_samples),
    )
    latency = np.clip(latency, 10, None)

    error_rate = np.where(
        cpu_usage > 90,
        np.random.uniform(2, 6, num_samples),
        np.random.uniform(0, 0.5, num_samples),
    )

    df = pd.DataFrame({
        "cpu_usage": cpu_usage,
        "memory_usage": memory_usage,
        "disk_io": disk_io,
        "network_usage": network_usage,
        "request_rate": np.round(request_rate).astype(int),
        "queue_length": np.round(queue_length).astype(int),
        "latency": latency,
        "error_rate": error_rate,
    })

    # Derived features expected by the models.
    df["cpu_change_rate"] = df["cpu_usage"].diff().fillna(0).astype(float)
    df["moving_average_cpu"] = df["cpu_usage"].rolling(5).mean().bfill().astype(float)

    # Time features.
    df["hour_of_day"] = hours.astype(int)
    df["day_of_week"] = timestamps.dayofweek.to_numpy().astype(int)
    df["minute_bucket"] = (timestamps.minute.to_numpy() // 5).astype(int)

    # Failure label: high CPU sustained with high latency/errors.
    cpu_crit = df["cpu_usage"].quantile(0.90)
    lat_high = df["latency"].quantile(0.80)
    df["failure_label"] = np.where(
        ((df["cpu_usage"] >= cpu_crit) & (df["latency"] >= lat_high))
        | (df["error_rate"] > 3.0),
        1, 0,
    ).astype(int)

    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")
    print(f"Failure label balance: {df['failure_label'].value_counts(normalize=True).to_dict()}")
    return output_path


if __name__ == "__main__":
    generate()
