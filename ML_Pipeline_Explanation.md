# ML Pipeline Explanation (Code-Based)

This document describes the **complete machine learning pipeline implemented in this repository**, based strictly on the code under `backend/` and the datasets checked into `data/`.

---

## Project Overview

### Problem statement
- Cloud systems need to **scale compute instances** to handle changing workload.
- Over-scaling wastes money; under-scaling risks **SLA issues** (high latency / overload) and **failures**.

### Goal of the project
- Build an ML-assisted system that can:
  - **Predict near-future workload** (CPU + request rate),
  - Estimate **prediction uncertainty**,
  - Predict **failure risk** (binary probability),
  - Use these signals inside a **cost-aware, SLA-aware scaling decision engine**,
  - Run a **step-by-step autoscaling simulation** and log results for visualization.

Where this is implemented:
- Backend API + simulation: `backend/main.py`, `backend/src/simulator.py`, `backend/src/decision_engine.py`
- Training: `backend/train.py`
- Feature engineering / labeling: `backend/src/build_final_dataset.py`, `backend/src/feature_engineering.py`
- Dataset preparation: `backend/src/merge_dataset.py`, `backend/src/clean_dataset.py`
- Validation utilities: `backend/src/validate_dataset.py`

---

## Dataset Details

### Source of data (as per repo + code)
- Raw input files are CSVs located in `data/raw/*.csv`.
  - The merge script reads these files using `sep=';'`:
    - `backend/src/merge_dataset.py`
- The repository includes example raw files:
  - `data/raw/1.csv`
  - `data/raw/3.csv`

Intermediate and final datasets:
- **Merged**: `data/processed/merged_dataset.csv` (created by `backend/src/merge_dataset.py`)
- **Cleaned**: `data/processed/cleaned_dataset.csv` (created by `backend/src/clean_dataset.py`)
- **Final engineered dataset**: `data/final/final_dataset.csv` (created by `backend/src/build_final_dataset.py`)
- **Final dataset with future targets**: `data/final/final_dataset_ready.csv` (created by `backend/train.py` via `preprocess_and_engineer_features`)

Note:
- There is also a synthetic telemetry generator in `backend/src/data_generator.py` that can create a standalone dataset (default `data/telemetry.csv`), but it is **not referenced by the training or API pipeline**.

### Features used (actual feature lists in code)

#### Workload prediction model (TCN ensemble) input features
Used in training and inference (`backend/train.py`, `backend/main.py`):
- `cpu_usage`
- `memory_usage`
- `disk_io`
- `network_usage`
- `request_rate`
- `queue_length`
- `cpu_change_rate`
- `moving_average_cpu`
- `hour_of_day`
- `day_of_week`
- `minute_bucket`

Targets (supervised regression, 2 outputs):
- `future_cpu_usage`
- `future_request_rate`

#### Failure prediction model (XGBoost) input features
Used in training and inference (`backend/train.py`, `backend/main.py`):
- `cpu_usage`
- `memory_usage`
- `disk_io`
- `network_usage`
- `request_rate`
- `queue_length`
- `latency`
- `error_rate`
- `cpu_change_rate`
- `moving_average_cpu`
- `hour_of_day`
- `day_of_week`
- `minute_bucket`

Target (binary classification):
- `future_failure` (constructed from `failure_label`, shifted to “next step”)

### Data preprocessing steps (pipeline in code)

#### 1) Merge raw CSVs
Script: `backend/src/merge_dataset.py`
- Reads: `data/raw/*.csv` using `pd.read_csv(file, sep=';')`
- Strips whitespace from column names
- Concatenates into `data/processed/merged_dataset.csv`

#### 2) Clean merged dataset
Script: `backend/src/clean_dataset.py`
- Reads: `data/processed/merged_dataset.csv`
- Strips column name whitespace
- Drops duplicates and drops rows with null values
- Sorts by `Timestamp [ms]` (if present)
- Writes: `data/processed/cleaned_dataset.csv`

#### 3) Build final ML-ready dataset (feature creation + labels)
Script: `backend/src/build_final_dataset.py`
- Reads: `data/processed/cleaned_dataset.csv`
- Creates base numeric features from raw telemetry-style columns:
  - `cpu_usage` from `CPU usage [%]`
  - `memory_usage` as percent from `Memory usage [KB] / Memory capacity provisioned [KB]`
  - `disk_io` from `Disk read throughput [KB/s] + Disk write throughput [KB/s]`
  - `network_usage` from `Network received throughput [KB/s] + Network transmitted throughput [KB/s]`
- Replaces `inf/-inf` with NaN and fills with column mean (for these base metrics)
- Creates engineered features:
  - `cpu_change_rate` = diff of `cpu_usage`
  - `moving_average_cpu` = rolling mean of `cpu_usage` (window=5)
  - `request_rate` = `(network_usage * 0.1) + Normal(50, 20)` then clipped to minimum 10 and rounded to int
  - `queue_length` = conditional function of request spikes vs rolling mean (window=10), rounded to int
  - `latency` = function of `disk_io`, `queue_length`, `cpu_usage` + noise, clipped at 0
  - `error_rate` = higher random range when latency is high or queue is high, else small random range
- Creates labels using quantile-based rules:
  - `overload_label` based on CPU/memory/latency thresholds
  - `failure_label` based on more severe CPU/latency/error combinations
- Creates time features (if `Timestamp [ms]` exists):
  - Attempts to detect seconds vs milliseconds by median magnitude
  - Derives:
    - `hour_of_day`
    - `day_of_week`
    - `minute_bucket` (minute // 5)
  - Adds a safe monotonically increasing numeric `timestamp` column as `np.arange(len(df))`
- Drops raw source columns (if present) such as:
  - `CPU usage [%]`, `Memory usage [KB]`, `Disk read throughput [KB/s]`, `Timestamp [ms]`, etc.
- Writes: `data/final/final_dataset.csv`

#### 4) Create “future” targets and sequences
File: `backend/src/feature_engineering.py`
- `preprocess_and_engineer_features(df)`:
  - If `Timestamp [ms]` exists, attempts to convert to datetime into `timestamp` (note: this is separate from the incremental `timestamp` created in `build_final_dataset.py`)
  - Creates next-step supervised targets:
    - `future_cpu_usage` = `cpu_usage.shift(-1)`
    - `future_request_rate` = `request_rate.shift(-1)`
    - `future_failure` = `failure_label.shift(-1)` (only if `failure_label` exists)
  - Drops rows with NaN from shifting
- `create_sequences(df, feature_cols, target_cols, seq_length=12)`:
  - Builds sliding windows of length 12 for TCN input
  - Target is the value at `i + seq_length`

Where this is executed:
- Training: `backend/train.py` reads `data/final/final_dataset.csv`, runs `preprocess_and_engineer_features`, writes `data/final/final_dataset_ready.csv`, and builds sequences.
- API: `backend/main.py` loads `data/final/final_dataset_ready.csv` and builds sequences on startup (when ML is available).

---

## Model Architecture

This project uses **two ML model families** (plus an ensemble wrapper) and combines them in a single decision-making pipeline.

### 1) Workload predictor: TCN + Attention (regression, 2 outputs)
Implementation: `backend/src/models/workload_predictor.py`
- Architecture:
  - Input: sequence \((12, \#features)\)
  - `TCN` layer:
    - `nb_filters=64`
    - `kernel_size=3`
    - `dilations=[1, 2, 4, 8, 16]`
    - `return_sequences=True`
    - `activation='relu'`
  - Custom `AttentionLayer` on top of the TCN time dimension
  - Dense output:
    - `Dense(2, activation='linear')` predicting:
      - `future_cpu_usage`
      - `future_request_rate`
- Training config:
  - `optimizer='adam'`
  - `loss='mse'`
  - metric: `mae`

Why this model is selected (supported by code intent)
- It is built specifically for **time-series sequence modeling** (`create_sequences` and `TCN` usage).
- Attention is used to learn **which time steps matter most** for the prediction (custom `AttentionLayer`).

Advantages (relative to what the code is trying to do)
- TCNs handle temporal patterns with **dilated convolutions**, giving a wider effective receptive field over the last 12 steps.
- The model outputs **two related continuous targets** jointly (CPU and request rate), matching the project’s predictive analytics goal.

### 2) Uncertainty estimation: Ensemble of 5 workload models (mean + std)
Implementation: `backend/src/models/uncertainty_estimator.py`
- `EnsemblePredictor`:
  - Takes a list of trained Keras models
  - Runs inference across all models
  - Returns:
    - mean prediction
    - standard deviation across models (used as an uncertainty score)

Where it is used:
- Training: `backend/train.py` trains 5 workload models and saves them as `backend/models/workload_tcn_model_{i}.h5`
- Inference: `backend/main.py` loads exactly 5 models (if present) and uses std dev as `uncertainty`

Advantages
- Gives a practical uncertainty signal without changing the base architecture (uncertainty = ensemble disagreement).

### 3) Failure predictor: XGBoost classifier (binary probability)
Implementation: `backend/src/models/failure_predictor.py`
- Uses `xgboost.XGBClassifier` with:
  - `n_estimators=100`
  - `max_depth=5`
  - `learning_rate=0.1`
  - `eval_metric='logloss'`
  - `scale_pos_weight` to address class imbalance
  - `random_state=42`

Where it is used:
- Training: `backend/train.py` trains on `future_failure` and saves `backend/models/failure_predictor.joblib`
- Inference: `backend/main.py` calls `predict_proba(...)` and uses the probability as `failure_prob`

Why this model is selected (supported by code intent)
- The pipeline needs a **probability-like risk score** (`failure_prob`) to drive scaling decisions.
- Training explicitly addresses **class imbalance** via `scale_pos_weight`.

Advantages
- Strong baseline for tabular engineered features.
- Fast inference, easy to integrate into an API.

---

## Step-by-Step Workflow (End-to-End)

### Data collection
From code, data is provided as:
- Local raw CSV files in `data/raw/*.csv` (merged by `backend/src/merge_dataset.py`)

### Data preprocessing
Implemented by:
- `backend/src/merge_dataset.py` → `data/processed/merged_dataset.csv`
- `backend/src/clean_dataset.py` → `data/processed/cleaned_dataset.csv`

### Feature engineering
Implemented by:
- `backend/src/build_final_dataset.py` → `data/final/final_dataset.csv`
  - Converts raw telemetry columns into ML features
  - Creates additional synthetic/derived features (`request_rate`, `latency`, `error_rate`, etc.)
  - Creates labels (`overload_label`, `failure_label`)
  - Extracts time features (`hour_of_day`, `day_of_week`, `minute_bucket`)
- `backend/src/feature_engineering.py`
  - Creates next-step (“future”) targets by shifting
  - Builds TCN sequences of length 12

### Model training
Primary training script: `backend/train.py`
- Reads: `data/final/final_dataset.csv`
- Produces: `data/final/final_dataset_ready.csv`
- Trains:
  - **Workload predictor ensemble**:
    - 5 models
    - `epochs=5`, `batch_size=64`, `validation_split=0.2`
    - Saves: `backend/models/workload_tcn_model_{0..4}.h5`
  - **Failure predictor**:
    - Splits with `train_test_split(..., shuffle=False)` (explicitly to avoid time-series leakage)
    - Saves: `backend/models/failure_predictor.joblib`

### Validation and testing
Validation script: `backend/src/validate_dataset.py`
- Runs dataset sanity checks:
  - shape, dtypes, missing values, duplicates
  - class balance (`failure_label`, `overload_label`)
  - correlation inspection (`df.corr()`) and prints top correlations to `failure_label`
  - baseline feature importance using `RandomForestClassifier`
- Generates a heatmap image:
  - `data/correlation_heatmap.png`

Evaluation printed during failure model training: `backend/train.py`
- `accuracy`
- `precision`
- `recall`
- `f1`
- `roc_auc`
- `confusion_matrix`

Workload model training metrics: `backend/src/models/workload_predictor.py`
- Trains with:
  - loss: MSE
  - metric: MAE

### Deployment (what exists in this repo)
There is a local deployment via an API server:
- **FastAPI backend** in `backend/main.py`
- Frontend UI calls the API at `http://localhost:8000/api` (see `frontend/src/api.js`)

Backend behavior on startup:
- Loads models if present under `backend/models/`
- Loads `data/final/final_dataset_ready.csv` (or builds it if possible from `data/final/final_dataset.csv`)

Endpoints used by the UI:
- `GET /api/system/status`
- `POST /api/model/train` (starts training in the background)
- `POST /api/scale/run` (runs one simulation step)
- `GET /api/metrics/current`
- `GET /api/timeline`

Persistence:
- Metrics are stored in SQLite:
  - DB: `backend/db/autoscaling.db` (configured in `backend/database.py`)
  - Table: `metrics_history` (defined in `backend/db_models.py`)

---

## Model Flow (How models are connected)

This project is a **multi-model pipeline** feeding a decision engine and simulator.

### Pipeline structure (as implemented)
- **Input at step \(i\)**: latest window of engineered features from the dataset.
- **Workload predictor ensemble (TCN+Attention)**:
  - Takes a 12-step sequence of features
  - Outputs:
    - `predicted_cpu` (from mean prediction’s CPU component)
    - `uncertainty` (std dev across ensemble for CPU)
- **Failure predictor (XGBoost)**:
  - Takes a single-row tabular feature vector at the corresponding time index
  - Outputs:
    - `failure_prob`
- **Drift detector**:
  - Compares `actual_cpu` vs `predicted_cpu` over a rolling window
  - Output:
    - `drift_detected` (boolean) + MAE drift score
- **Decision engine** (`backend/src/decision_engine.py`):
  - Consumes `predicted_cpu`, `failure_prob`, `uncertainty`, plus runtime signals (latency, queue, budget, peak_hours)
  - Outputs:
    - scaling `action`
    - `target_instances`
    - `reason`
- **Simulator** (`backend/src/simulator.py`):
  - Applies the decision and computes:
    - adjusted CPU (scaled by instance count)
    - SLA violation flag
    - adjusted latency
    - cost per interval
- **Database logger**:
  - Persists each step for the UI timeline charts

### Ensemble vs sequential vs parallel
- The pipeline runs **two predictors in parallel** at each step:
  - TCN ensemble (regression)
  - XGBoost classifier (risk)
- Their outputs are combined **sequentially** into:
  - drift detection → decision engine → simulator → persistence

---

## Evaluation Metrics

### Workload prediction (regression)
Defined in `backend/src/models/workload_predictor.py`:
- **Loss**: Mean Squared Error (MSE)
- **Metric**: Mean Absolute Error (MAE)

Why these are used (as reflected by training setup)
- The workload model predicts continuous values, so MSE/MAE are standard regression objectives.
- MAE provides an interpretable “average absolute error” alongside MSE optimization.

### Failure prediction (classification)
From `backend/train.py` and `backend/src/models/failure_predictor.py`:
- XGBoost training uses `eval_metric='logloss'`
- Reported metrics (printed in training):
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1 Score**
  - **ROC-AUC**
  - **Confusion Matrix**

Why these are used (supported by the printed emphasis)
- The script explicitly prints a “Performance” section and includes recall-focused metrics; it also handles imbalance via `scale_pos_weight`.

### Drift detection metric (monitoring)
From `backend/src/monitoring/drift_detector.py`:
- Computes **MAE** between actual and predicted over a rolling window.
- Triggers drift when MAE exceeds a configured threshold.

---

## Improvements & Optimization (Implemented in code)

### Data leakage prevention (time-series split)
In `backend/train.py`:
- Failure model uses `train_test_split(..., shuffle=False)`
- The file explicitly flags this as a “data leakage fix” for time-series data.

### Handling class imbalance (failure prediction)
In `backend/train.py`:
- Computes:
  - `weight = (#negatives) / max(#positives, 1)`
- Passes this as `scale_pos_weight` to XGBoost.

### Uncertainty estimation via ensemble
In `backend/src/models/uncertainty_estimator.py`:
- Uses standard deviation across 5 independently trained TCN models as `uncertainty`.

### Robustness / fallback behavior for deployment environments
In `backend/main.py`:
- If ML imports fail (e.g., TensorFlow limitations), the system switches to a **mock inference path**:
  - predicted CPU = actual + noise
  - uncertainty = random range
  - failure probability = high when CPU is high, else random low

This ensures the API + UI remain functional even when ML libraries cannot run.

---

## Conclusion

### Final outcome (what the implemented pipeline produces)
- A working end-to-end pipeline that:
  - builds a structured dataset from raw telemetry CSVs,
  - engineers time-series + derived operational features,
  - trains:
    - a **TCN + Attention** workload predictor ensemble (with uncertainty),
    - an **XGBoost** failure-risk classifier,
  - serves predictions via **FastAPI**,
  - feeds them into a **cost-aware/SLA-aware autoscaling decision engine**,
  - simulates scaling outcomes and logs time-series metrics to **SQLite** for dashboard visualization.

### Future improvements (only what is suggested by gaps in current code)
- **Persist drift score** (MAE value) in the DB alongside `drift_detected` to make drift analysis explainable over time (currently only boolean is stored).
- **Add explicit train/validation split for the workload model** based on time ordering (current workload training uses `validation_split=0.2`, which splits by array order but is not explicitly documented as a time-based holdout).
- **Centralize feature definitions** in one module to avoid duplication between training (`backend/train.py`) and serving (`backend/main.py`).

