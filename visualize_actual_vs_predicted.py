"""
Plot Actual vs Predicted values using real data from this repo.

Data sources (auto-detected):
1) SQLite simulation history (backend/db/autoscaling.db) if it exists and has rows.
2) If available, trained workload models (backend/models/workload_tcn_model_{i}.h5)
   + engineered dataset (data/final/final_dataset_ready.csv) to plot per-model predictions.

Outputs:
- comparison_line.png
- comparison_bar.png
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")  # ensure running headless
import matplotlib.pyplot as plt  # noqa: E402


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def load_from_sqlite(db_path: Path):
    """
    Returns:
      x_steps: np.ndarray[int] (0..n-1)
      x_time_ms: np.ndarray[int] timestamps from DB (ms since epoch)
      actual: np.ndarray[float]
      predicted_mean: np.ndarray[float]
    """
    if not db_path.exists():
        return None

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, timestamp, actual_cpu, predicted_cpu
            FROM metrics_history
            ORDER BY id ASC
            """
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return None

    ids = np.array([r[0] for r in rows], dtype=np.int64)
    # DB stores "timestamp" as milliseconds since epoch; use int64 to avoid Windows long overflow.
    time_ms = np.array([r[1] for r in rows], dtype=np.int64)
    actual = np.array([r[2] for r in rows], dtype=float)
    predicted_mean = np.array([r[3] for r in rows], dtype=float)
    x_steps = np.arange(len(rows), dtype=int)
    return x_steps, time_ms, actual, predicted_mean


def load_predictions_from_models_and_dataset(
    dataset_csv: Path,
    model_paths: list[Path],
):
    """
    Returns:
      x_steps: np.ndarray[int] (0..n-1) aligned to the created sequences
      actual: np.ndarray[float] (y_seq[:, 0])
      preds_all_models: np.ndarray[float] with shape (num_models, n_points)
      preds_ensemble_mean: np.ndarray[float] with shape (n_points,)
    """
    # Import ML stack only if needed.
    import pandas as pd
    import joblib

    from tcn import TCN
    from tensorflow.keras.models import load_model
    from tensorflow.keras.utils import get_custom_objects

    from backend.src.feature_engineering import preprocess_and_engineer_features, create_sequences
    from backend.src.models.workload_predictor import AttentionLayer

    get_custom_objects().update({"TCN": TCN, "AttentionLayer": AttentionLayer})

    df = pd.read_csv(dataset_csv)
    # Matches backend preprocessing: create future labels, then drop NaNs.
    df = preprocess_and_engineer_features(df)

    feature_cols_tcn = [
        "cpu_usage",
        "memory_usage",
        "disk_io",
        "network_usage",
        "request_rate",
        "queue_length",
        "cpu_change_rate",
        "moving_average_cpu",
        "hour_of_day",
        "day_of_week",
        "minute_bucket",
    ]
    target_cols = ["future_cpu_usage", "future_request_rate"]

    backend_models_dir = Path(__file__).resolve().parent / "backend" / "models"
    feature_scaler = joblib.load(backend_models_dir / 'feature_scaler.joblib')
    target_scaler = joblib.load(backend_models_dir / 'target_scaler.joblib')

    df_scaled = df.copy()
    df_scaled[feature_cols_tcn] = feature_scaler.transform(df_scaled[feature_cols_tcn])

    X, _ = create_sequences(df_scaled, feature_cols_tcn, target_cols, seq_length=12)
    _, y = create_sequences(df, feature_cols_tcn, target_cols, seq_length=12)
    
    actual = y[:, 0].astype(float)
    x_steps = np.arange(len(actual), dtype=int)

    preds_all_models = []
    for mp in model_paths:
        model = load_model(str(mp))
        pred_scaled = model.predict(X, verbose=0)  # shape (n_points, 2)
        pred_inv = target_scaler.inverse_transform(pred_scaled)
        preds_all_models.append(pred_inv[:, 0].astype(float))

    preds_all_models = np.array(preds_all_models)  # (num_models, n_points)
    preds_ensemble_mean = np.median(preds_all_models, axis=0)
    return x_steps, actual, preds_all_models, preds_ensemble_mean


def save_line_plot(
    out_path: Path,
    x_steps: np.ndarray,
    actual: np.ndarray,
    preds_ensemble_mean: np.ndarray,
    preds_all_models: np.ndarray | None = None,
):
    plt.figure(figsize=(12, 6))

    # Actual
    plt.plot(x_steps, actual, color="#2563EB", linewidth=2.5, label="Actual")

    # Ensemble mean
    plt.plot(
        x_steps,
        preds_ensemble_mean,
        color="#F59E0B",
        linewidth=2.5,
        label="Predicted (ensemble mean)",
    )

    # Individual model curves (if provided)
    if preds_all_models is not None and preds_all_models.size > 0:
        colors = ["#7C3AED", "#DB2777", "#0EA5E9", "#16A34A", "#64748B"]
        for i in range(preds_all_models.shape[0]):
            plt.plot(
                x_steps,
                preds_all_models[i],
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                color=colors[i % len(colors)],
                label=f"Predicted (model {i})",
            )

    plt.title("Actual vs Predicted Comparison")
    plt.xlabel("Time / Index")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plt.savefig(str(out_path), dpi=200)
    plt.close()


def save_bar_plot(
    out_path: Path,
    actual: np.ndarray,
    preds_ensemble_mean: np.ndarray,
    preds_all_models: np.ndarray | None = None,
    bar_points: int = 10,
):
    n = len(actual)
    k = min(bar_points, n)

    # Compare the last k points.
    actual_k = actual[-k:]
    pred_mean_k = preds_ensemble_mean[-k:]
    x = np.arange(k, dtype=float)

    if preds_all_models is None or preds_all_models.size == 0:
        # Only actual vs ensemble mean
        width = 0.42
        plt.figure(figsize=(12, 6))
        plt.bar(x - width / 2, actual_k, width=width, color="#2563EB", label="Actual")
        plt.bar(x + width / 2, pred_mean_k, width=width, color="#F59E0B", label="Predicted (ensemble mean)")
    else:
        # Show actual plus each model prediction.
        num_models = preds_all_models.shape[0]
        all_models_k = preds_all_models[:, -k:]  # (num_models, k)

        # total bars per group = actual + models
        total_bars = 1 + num_models
        group_width = 0.95
        width = group_width / total_bars

        plt.figure(figsize=(14, 6))
        # Colors
        model_colors = ["#7C3AED", "#DB2777", "#0EA5E9", "#16A34A", "#64748B"]

        # Actual bar at first position
        for j in range(1):
            plt.bar(
                x - group_width / 2 + j * width,
                actual_k,
                width=width,
                color="#2563EB",
                label="Actual",
            )

        # Model bars
        for i in range(num_models):
            plt.bar(
                x - group_width / 2 + (i + 1) * width,
                all_models_k[i],
                width=width,
                color=model_colors[i % len(model_colors)],
                alpha=0.85,
                label=f"Predicted (model {i})",
            )

        # Also overlay ensemble mean as a line for clarity.
        plt.plot(x + 0.5 * width, pred_mean_k, color="#F59E0B", linewidth=2.0, label="Predicted (ensemble mean)")

    plt.title("Actual vs Predicted Comparison")
    plt.xlabel("Time / Index (last window)")
    plt.ylabel("Value")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=None, help="Path to autoscaling.db")
    parser.add_argument("--dataset", default=None, help="Path to final_dataset_ready.csv")
    parser.add_argument("--models-dir", default=None, help="Directory containing workload_tcn_model_{i}.h5")
    parser.add_argument("--bar-points", type=int, default=10, help="How many last points for the bar plot")
    parser.add_argument("--max-points", type=int, default=3000, help="Max points to plot from computed sequences")
    parser.add_argument("--zoom-points", type=int, default=0, help="If > 0, also save zoomed-in plots for the last N points")
    args = parser.parse_args()

    root = _project_root()

    db_path = Path(args.db) if args.db else root / "backend" / "db" / "autoscaling.db"
    dataset_csv = Path(args.dataset) if args.dataset else root / "data" / "final" / "final_dataset_ready.csv"
    models_dir = Path(args.models_dir) if args.models_dir else root / "backend" / "models"

    # 1) Try SQLite history first (always available if you ran simulation).
    sqlite_data = load_from_sqlite(db_path)

    # 2) If model files + dataset exist, compute per-model predictions too.
    model_paths = [models_dir / f"workload_tcn_model_{i}.h5" for i in range(5)]
    have_all_models = all(p.exists() for p in model_paths)
    have_dataset = dataset_csv.exists()

    preds_all_models = None
    preds_ensemble_mean = None
    actual = None
    x_steps = None

    if have_all_models and have_dataset:
        try:
            computed = load_predictions_from_models_and_dataset(dataset_csv, model_paths)
            x_steps_all, actual_all, preds_all_models_all, preds_mean_all = computed

            if sqlite_data is not None:
                # Align computed sequences to the number of DB rows.
                n = min(len(sqlite_data[0]), len(actual_all))
                x_steps = x_steps_all[:n]
                actual = actual_all[:n]
                preds_all_models = preds_all_models_all[:, :n]
                preds_ensemble_mean = preds_mean_all[:n]
            else:
                n = min(len(actual_all), args.max_points)
                x_steps = x_steps_all[:n]
                actual = actual_all[:n]
                preds_all_models = preds_all_models_all[:, :n]
                preds_ensemble_mean = preds_mean_all[:n]
        except Exception as e:
            # If ML stack isn't available (e.g., missing 'tcn'), fall back to SQLite.
            print(f"Warning: Could not compute per-model predictions from ML stack: {e}")
            print("Falling back to SQLite simulation history (ensemble mean only).")
            sqlite_data = sqlite_data  # keep for clarity
    else:
        # Fall back to SQLite (ensemble mean only).
        if sqlite_data is None:
            raise FileNotFoundError(
                "No simulation history found (backend/db/autoscaling.db missing or empty), "
                "and model+dataset computation is not possible (missing models/*.h5 or dataset csv)."
            )
        x_steps_db, _time_ms, actual_db, pred_mean_db = sqlite_data
        n = len(actual_db)
        if n > args.max_points:
            # Keep only the tail for plotting readability
            keep_from = n - args.max_points
            x_steps = x_steps_db[keep_from:]
            actual = actual_db[keep_from:]
            preds_ensemble_mean = pred_mean_db[keep_from:]
        else:
            x_steps = x_steps_db
            actual = actual_db
            preds_ensemble_mean = pred_mean_db
        preds_all_models = None

    # If we attempted ML computation but fell back, we still need to populate from sqlite_data.
    if preds_ensemble_mean is None:
        if sqlite_data is None:
            raise RuntimeError("No data available to plot (SQLite empty and ML computation failed).")
        x_steps_db, _time_ms, actual_db, pred_mean_db = sqlite_data
        n = len(actual_db)
        if n > args.max_points:
            keep_from = n - args.max_points
            x_steps = x_steps_db[keep_from:]
            actual = actual_db[keep_from:]
            preds_ensemble_mean = pred_mean_db[keep_from:]
        else:
            x_steps = x_steps_db
            actual = actual_db
            preds_ensemble_mean = pred_mean_db
        preds_all_models = None

    out_line = root / "comparison_line.png"
    out_bar = root / "comparison_bar.png"

    save_line_plot(out_line, x_steps, actual, preds_ensemble_mean, preds_all_models)
    save_bar_plot(out_bar, actual, preds_ensemble_mean, preds_all_models, bar_points=args.bar_points)

    if args.zoom_points and args.zoom_points > 0:
        z = min(args.zoom_points, len(actual))
        x_zoom = x_steps[-z:]
        actual_zoom = actual[-z:]
        pred_mean_zoom = preds_ensemble_mean[-z:]
        preds_all_models_zoom = preds_all_models[:, -z:] if preds_all_models is not None else None

        out_line_zoom = root / "comparison_line_zoom.png"
        out_bar_zoom = root / "comparison_bar_zoom.png"
        save_line_plot(out_line_zoom, x_zoom, actual_zoom, pred_mean_zoom, preds_all_models_zoom)
        save_bar_plot(out_bar_zoom, actual_zoom, pred_mean_zoom, preds_all_models_zoom, bar_points=min(args.bar_points, z))

        print(f"Saved: {out_line_zoom}")
        print(f"Saved: {out_bar_zoom}")

    print(f"Saved: {out_line}")
    print(f"Saved: {out_bar}")


if __name__ == "__main__":
    main()

