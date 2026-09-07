"""
Train and Compare Base Paper (Filter-KD Bi-LSTM) vs Current Project Model (TCN + Attention Ensemble).

Generates comparative charts matching the base paper's visualization style:
1. Learning Curves (Epochs vs Loss for Baseline Bi-LSTM, Filter-KD Bi-LSTM, and TCN+Attention Ensemble)
2. Actual vs Predicted CPU Usage Overlay
3. Performance Metrics Bar Charts (MSE, Latency ms, Model Size MB)
4. Exports metrics JSON for API/Frontend integration
"""

import os
import json
import time
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import load_model
from tcn import TCN

from backend.src.feature_engineering import preprocess_and_engineer_features, create_sequences
from backend.src.models.workload_predictor import build_workload_model, AttentionLayer
from backend.src.models.base_paper_filter_kd import build_bilstm_model, FilterKDStudentModel

get_custom_objects().update({'TCN': TCN, 'AttentionLayer': AttentionLayer})


def measure_model_memory_mb(model: tf.keras.Model) -> float:
    """Estimates parameter size in MB for Keras model."""
    total_params = model.count_params()
    # Assuming float32 (4 bytes per param)
    return round((total_params * 4) / (1024 * 1024), 2)


def main():
    print("==========================================================================")
    print(" Base Paper (Filter-KD Bi-LSTM) vs Current Improved Model Benchmark")
    print("==========================================================================")

    # Directories setup
    project_root = Path(__file__).resolve().parent
    chart_dir_data = project_root / "data" / "charts"
    chart_dir_public = project_root / "frontend" / "public" / "charts"
    chart_dir_data.mkdir(parents=True, exist_ok=True)
    chart_dir_public.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset
    ready_csv = project_root / "data" / "final" / "final_dataset_ready.csv"
    raw_csv = project_root / "data" / "final" / "final_dataset.csv"

    if ready_csv.exists():
        df = pd.read_csv(ready_csv)
    elif raw_csv.exists():
        raw_df = pd.read_csv(raw_csv)
        df = preprocess_and_engineer_features(raw_df)
        df.to_csv(ready_csv, index=False)
    else:
        raise FileNotFoundError("Dataset file not found in data/final/")

    print(f"Dataset loaded: {len(df)} rows.")

    feature_cols_tcn = [
        'cpu_usage', 'memory_usage', 'disk_io', 'network_usage',
        'request_rate', 'queue_length', 'cpu_change_rate', 'moving_average_cpu',
        'hour_of_day', 'day_of_week', 'minute_bucket'
    ]
    target_cols = ['future_cpu_usage', 'future_request_rate']

    # Scalers
    from sklearn.preprocessing import MinMaxScaler
    models_dir = project_root / "backend" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    feat_scaler_path = models_dir / "feature_scaler.joblib"
    targ_scaler_path = models_dir / "target_scaler.joblib"

    if feat_scaler_path.exists() and targ_scaler_path.exists():
        feature_scaler = joblib.load(feat_scaler_path)
        target_scaler = joblib.load(targ_scaler_path)
    else:
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
        feature_scaler.fit(df[feature_cols_tcn])
        target_scaler.fit(df[target_cols])
        joblib.dump(feature_scaler, feat_scaler_path)
        joblib.dump(target_scaler, targ_scaler_path)

    df_scaled = df.copy()
    df_scaled[feature_cols_tcn] = feature_scaler.transform(df_scaled[feature_cols_tcn])
    df_scaled[target_cols] = target_scaler.transform(df_scaled[target_cols])

    X_seq, y_seq = create_sequences(df_scaled, feature_cols_tcn, target_cols, seq_length=12)
    _, y_seq_unscaled = create_sequences(df, feature_cols_tcn, target_cols, seq_length=12)

    # Train / Test split (80% train, 20% test, no shuffle for time series)
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    actual_cpu_test = y_seq_unscaled[split_idx:, 0]

    epochs = 3
    batch_size = 64


    # ---------------------------------------------------------
    # Model 1: Base Paper Baseline Student (Bi-LSTM-128x2)
    # ---------------------------------------------------------
    print("\n--- 1/3 Training Base Paper Baseline Student (Bi-LSTM-128x2) ---")
    bilstm_baseline = build_bilstm_model(seq_length=12, num_features=len(feature_cols_tcn), hidden_units=128, num_layers=2)
    history_baseline = bilstm_baseline.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # ---------------------------------------------------------
    # Model 2: Base Paper Filter-KD Student (Bi-LSTM-128x2 with Teacher 512x2)
    # ---------------------------------------------------------
    print("\n--- 2/3 Training Base Paper Teacher (Bi-LSTM-512x2) & Distilled Student (Filter-KD) ---")
    bilstm_teacher = build_bilstm_model(seq_length=12, num_features=len(feature_cols_tcn), hidden_units=512, num_layers=2)
    bilstm_teacher.fit(X_train, y_train, epochs=5, batch_size=batch_size, verbose=0)

    bilstm_student_kd = build_bilstm_model(seq_length=12, num_features=len(feature_cols_tcn), hidden_units=128, num_layers=2)
    kd_model = FilterKDStudentModel(student_network=bilstm_student_kd, teacher_network=bilstm_teacher, gamma=0.5, epsilon=0.02)
    kd_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.003))

    history_kd_train_loss = []
    history_kd_val_loss = []

    for epoch in range(epochs):
        hist = kd_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=batch_size, verbose=0)
        tr_l = float(hist.history['loss'][0])
        val_l = float(hist.history['val_loss'][0])
        history_kd_train_loss.append(tr_l)
        history_kd_val_loss.append(val_l)
        print(f"Epoch {epoch+1}/{epochs} - Filter-KD Loss: {tr_l:.4f} - Val Loss: {val_l:.4f}")

    # ---------------------------------------------------------
    # Model 3: Current Improved Model (TCN + Attention Ensemble)
    # ---------------------------------------------------------
    print("\n--- 3/3 Training Current Improved Model (TCN + Attention Ensemble) ---")
    tcn_model = build_workload_model(seq_length=12, num_features=len(feature_cols_tcn))
    history_tcn = tcn_model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Load full 5-model ensemble if available to measure true ensemble inference, else use single TCN
    ensemble_models = []
    for i in range(5):
        m_path = models_dir / f"workload_tcn_model_{i}.keras"
        if not m_path.exists():
            m_path = models_dir / f"workload_tcn_model_{i}.h5"
        if m_path.exists():
            ensemble_models.append(load_model(str(m_path)))

    if len(ensemble_models) < 5:
        ensemble_models = [tcn_model]

    # ---------------------------------------------------------
    # Benchmark Inference Latency & Model Sizes
    # ---------------------------------------------------------
    print("\nBenchmarking Inference Latency...")
    # Baseline Bi-LSTM inference time
    t0 = time.perf_counter()
    pred_base_scaled = bilstm_baseline.predict(X_test, verbose=0)
    lat_base = ((time.perf_counter() - t0) / len(X_test)) * 1000.0  # ms per sample

    # Distilled Filter-KD Student inference time
    t0 = time.perf_counter()
    pred_kd_scaled = bilstm_student_kd.predict(X_test, verbose=0)
    lat_kd = ((time.perf_counter() - t0) / len(X_test)) * 1000.0

    # TCN + Attention Ensemble inference time
    t0 = time.perf_counter()
    ens_preds_scaled = [m.predict(X_test, verbose=0) for m in ensemble_models]
    lat_improved = ((time.perf_counter() - t0) / len(X_test)) * 1000.0
    pred_tcn_scaled = np.median(ens_preds_scaled, axis=0)

    # Inverse transform predictions to CPU % scale
    pred_base_cpu = target_scaler.inverse_transform(pred_base_scaled)[:, 0]
    pred_kd_cpu = target_scaler.inverse_transform(pred_kd_scaled)[:, 0]
    pred_improved_cpu = target_scaler.inverse_transform(pred_tcn_scaled)[:, 0]

    # Un-scaled MSE & MAE in CPU percentage
    mse_base = float(np.mean((pred_base_cpu - actual_cpu_test) ** 2))
    mae_base = float(np.mean(np.abs(pred_base_cpu - actual_cpu_test)))

    mse_kd = float(np.mean((pred_kd_cpu - actual_cpu_test) ** 2))
    mae_kd = float(np.mean(np.abs(pred_kd_cpu - actual_cpu_test)))

    mse_improved = float(np.mean((pred_improved_cpu - actual_cpu_test) ** 2))
    mae_improved = float(np.mean(np.abs(pred_improved_cpu - actual_cpu_test)))

    size_base = measure_model_memory_mb(bilstm_baseline)
    size_kd = measure_model_memory_mb(bilstm_student_kd)
    size_improved = measure_model_memory_mb(tcn_model) * len(ensemble_models)

    print("\n=== Benchmark Summary ===")
    print(f"Base Paper Baseline (Bi-LSTM):    MSE = {mse_base:.4f}, MAE = {mae_base:.4f}, Latency = {lat_base:.3f} ms/sample, Size = {size_base} MB")
    print(f"Base Paper Distilled (Filter-KD): MSE = {mse_kd:.4f}, MAE = {mae_kd:.4f}, Latency = {lat_kd:.3f} ms/sample, Size = {size_kd} MB")
    print(f"Current Improved (TCN+Attention): MSE = {mse_improved:.4f}, MAE = {mae_improved:.4f}, Latency = {lat_improved:.3f} ms/sample, Size = {size_improved} MB")

    # ---------------------------------------------------------
    # Graph 1: Learning Curves (Epochs vs Loss)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 5))
    ep_axis = np.arange(1, epochs + 1)
    plt.plot(ep_axis, history_baseline.history['val_loss'], label="Base Paper Baseline Student (Bi-LSTM)", color="#EF4444", linewidth=2, linestyle="--")
    plt.plot(ep_axis, history_kd_val_loss, label="Base Paper Distilled Student (Filter-KD)", color="#F59E0B", linewidth=2)
    plt.plot(ep_axis, history_tcn.history['val_loss'], label="Current Improved Model (TCN + Attention)", color="#3B82F6", linewidth=2.5)

    plt.title("Learning Curves (Validation Loss across Epochs)", fontsize=14, fontweight="bold")
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Validation Loss (MSE)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()

    fig1_path_data = chart_dir_data / "base_paper_vs_improved_learning_curves.png"
    fig1_path_public = chart_dir_public / "base_paper_vs_improved_learning_curves.png"
    plt.savefig(str(fig1_path_data), dpi=200)
    plt.savefig(str(fig1_path_public), dpi=200)
    plt.close()

    # ---------------------------------------------------------
    # Graph 2: Actual vs Predicted CPU Overlay
    # ---------------------------------------------------------
    sample_points = min(120, len(actual_cpu_test))
    x_steps = np.arange(sample_points)

    plt.figure(figsize=(12, 6))
    plt.plot(x_steps, actual_cpu_test[:sample_points], label="Actual Ground Truth CPU", color="#10B981", linewidth=2.5)
    plt.plot(x_steps, pred_base_cpu[:sample_points], label="Base Paper Baseline (Bi-LSTM)", color="#EF4444", linewidth=1.5, linestyle="--")
    plt.plot(x_steps, pred_kd_cpu[:sample_points], label="Base Paper Distilled (Filter-KD Bi-LSTM)", color="#F59E0B", linewidth=1.8, linestyle=":")
    plt.plot(x_steps, pred_improved_cpu[:sample_points], label="Current Improved (TCN + Attention)", color="#6366F1", linewidth=2.2)

    plt.title("Workload Prediction Comparison (Actual vs Base Paper vs Improved Model)", fontsize=14, fontweight="bold")
    plt.xlabel("Time Step (Sample Index)", fontsize=12)
    plt.ylabel("CPU Usage (%)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()

    fig2_path_data = chart_dir_data / "base_paper_vs_improved_predictions.png"
    fig2_path_public = chart_dir_public / "base_paper_vs_improved_predictions.png"
    plt.savefig(str(fig2_path_data), dpi=200)
    plt.savefig(str(fig2_path_public), dpi=200)
    plt.close()

    # ---------------------------------------------------------
    # Graph 3: Comparative Bar Metrics Chart
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    models_labels = ["Base Baseline\n(Bi-LSTM)", "Base Filter-KD\n(Distilled)", "Current Improved\n(TCN+Attention)"]
    colors = ["#EF4444", "#F59E0B", "#3B82F6"]

    # Subplot 1: MSE Error
    mse_vals = [mse_base, mse_kd, mse_improved]
    bars1 = axes[0].bar(models_labels, mse_vals, color=colors, width=0.55)
    axes[0].set_title("Prediction Error (MSE)", fontweight="bold")
    axes[0].set_ylabel("MSE (Lower is Better)")
    axes[0].grid(axis="y", alpha=0.3)
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, yval + (yval * 0.02), f"{yval:.4f}", ha='center', va='bottom', fontweight='bold')

    # Subplot 2: Latency per Sample
    lat_vals = [lat_base, lat_kd, lat_improved]
    bars2 = axes[1].bar(models_labels, lat_vals, color=colors, width=0.55)
    axes[1].set_title("Inference Latency (ms / sample)", fontweight="bold")
    axes[1].set_ylabel("Latency in ms (Lower is Better)")
    axes[1].grid(axis="y", alpha=0.3)
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, yval + (yval * 0.02), f"{yval:.3f} ms", ha='center', va='bottom', fontweight='bold')

    # Subplot 3: Model Footprint (MB)
    size_vals = [size_base, size_kd, size_improved]
    bars3 = axes[2].bar(models_labels, size_vals, color=colors, width=0.55)
    axes[2].set_title("Model Memory Size (MB)", fontweight="bold")
    axes[2].set_ylabel("Memory in MB")
    axes[2].grid(axis="y", alpha=0.3)
    for bar in bars3:
        yval = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2, yval + (yval * 0.02), f"{yval:.2f} MB", ha='center', va='bottom', fontweight='bold')

    plt.suptitle("Base Paper vs Current Improved Model Performance Comparison", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    fig3_path_data = chart_dir_data / "base_paper_vs_improved_metrics.png"
    fig3_path_public = chart_dir_public / "base_paper_vs_improved_metrics.png"
    plt.savefig(str(fig3_path_data), dpi=200, bbox_inches='tight')
    plt.savefig(str(fig3_path_public), dpi=200, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # Metrics JSON Export
    # ---------------------------------------------------------
    metrics_summary = {
        "dataset_points": len(df),
        "test_points": len(actual_cpu_test),
        "base_paper_baseline": {
            "name": "Bi-LSTM Baseline (Base Paper)",
            "mse": round(mse_base, 5),
            "mae": round(mae_base, 5),
            "latency_ms": round(lat_base, 4),
            "model_size_mb": size_base
        },
        "base_paper_filter_kd": {
            "name": "Filter-KD Distilled Bi-LSTM (Base Paper Proposed)",
            "mse": round(mse_kd, 5),
            "mae": round(mae_kd, 5),
            "latency_ms": round(lat_kd, 4),
            "model_size_mb": size_kd
        },
        "current_improved_model": {
            "name": "TCN + Attention Ensemble (Current Project)",
            "mse": round(mse_improved, 5),
            "mae": round(mae_improved, 5),
            "latency_ms": round(lat_improved, 4),
            "model_size_mb": size_improved
        },
        "improvement_pct": {
            "mse_reduction_vs_baseline_pct": round(((mse_base - mse_improved) / mse_base) * 100, 2),
            "mse_reduction_vs_filter_kd_pct": round(((mse_kd - mse_improved) / mse_kd) * 100, 2),
        },
        "charts": {
            "learning_curves": "/charts/base_paper_vs_improved_learning_curves.png",
            "predictions_overlay": "/charts/base_paper_vs_improved_predictions.png",
            "metrics_bars": "/charts/base_paper_vs_improved_metrics.png"
        }
    }

    json_path_data = chart_dir_data / "base_paper_comparison_metrics.json"
    json_path_public = chart_dir_public / "base_paper_comparison_metrics.json"

    with open(json_path_data, "w") as f:
        json.dump(metrics_summary, f, indent=2)

    with open(json_path_public, "w") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"\nAll comparison charts and metrics JSON successfully saved to {chart_dir_public}")

if __name__ == "__main__":
    main()
