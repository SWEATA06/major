# Project Features Explanation (Code-Based, Simple Terms)

This file explains what your project does in **simple, recruiter-friendly language**, strictly based on the code in `backend/` + `frontend/` and the datasets under `data/`.

---

## 1. Core Functionality

### What exactly does this project do?
- This is an **AI-assisted autoscaling simulator + dashboard**.
- It repeatedly takes cloud workload telemetry (CPU/latency/traffic-like metrics), then:
  - **predicts near-future workload** (CPU + request rate),
  - estimates **uncertainty** (how confident the workload prediction is),
  - predicts **failure risk** (probability of a failure/critical event),
  - makes an **autoscaling decision** (scale up/down/hold) using SLA + cost-aware rules,
  - simulates the result (instances, adjusted CPU, latency, cost),
  - stores each step in a **SQLite database**,
  - shows everything on a React dashboard (charts + KPIs).

Where this happens in code:
- API + orchestration: `backend/main.py`
- Scaling rules: `backend/src/decision_engine.py`
- Simulation + cost/SLA calculations: `backend/src/simulator.py`
- Dashboard polling + charts: `frontend/src/App.jsx`, `frontend/src/api.js`

### What real-world problem is it solving?
- In real cloud operations, teams want to **keep performance stable** (avoid overload/high latency/failures) while **controlling infrastructure cost**.
- This project demonstrates that idea by combining:
  - ML predictions (workload + failure probability),
  - monitoring (drift flag),
  - a cost/SLA-aware decision engine,
  - and an autoscaling simulation loop.

Important: This repo is a **simulation** (it changes a simulated `instances` count and computes simulated cost/latency), not a live cloud provisioning tool.

---

## 2. Key Features (What “features” means here)

When you ask “features means? like are we doing cost/energy optimization?”, here is what your **actual code implements**:

### ML-driven workload forecasting
- **What it does**: Predicts future CPU usage (and request rate) from the last 12 time steps of metrics.
- **Where**: trained in `backend/train.py`, model defined in `backend/src/models/workload_predictor.py`.

### Uncertainty estimation (confidence signal)
- **What it does**: Runs an ensemble of 5 workload models and uses the **standard deviation** across models as “uncertainty”.
- **Where**: `backend/src/models/uncertainty_estimator.py`, used in `backend/main.py`.

### Failure-risk prediction
- **What it does**: Predicts a probability (`failure_prob`) using an XGBoost classifier.
- **Where**: `backend/src/models/failure_predictor.py` + `backend/train.py` + inference in `backend/main.py`.

### Cost-aware + SLA-aware autoscaling decisions
- **What it does**: Converts predicted CPU + failure risk + uncertainty + latency/queue into actions like:
  - `urgent_scale_up`, `scale_up`, `conservative_scaling`, `scale_down`, `hold`
- **Where**: `backend/src/decision_engine.py`
- **Cost optimization?** Yes—**cost is explicitly modeled**:
  - decision logic checks remaining budget (daily budget vs budget used)
  - simulator computes a **per-interval cost** = `instances * cost_per_instance`
- **Energy optimization?** No—there is **no energy/power/carbon metric** in code.

### Drift detection (monitoring)
- **What it does**: Flags drift when the rolling MAE between actual CPU and predicted CPU exceeds a threshold.
- **Where**: `backend/src/monitoring/drift_detector.py`, used in `backend/main.py`.

### Step-by-step simulation engine
- **What it does**:
  - Applies scaling decision (changes instance count)
  - Adjusts CPU based on instance count
  - Flags SLA violations when adjusted CPU is too high
  - Adjusts latency based on load
  - Computes cost per step
- **Where**: `backend/src/simulator.py`

### Persisted history (time series logging)
- **What it does**: Stores every simulation step as a row in SQLite (`metrics_history` table).
- **Where**: `backend/database.py`, `backend/db_models.py`, writes in `backend/main.py`.

### Interactive dashboard (train + simulate + visualize)
- **What it does**:
  - “Train Models” button triggers backend training
  - “Step Forward” runs one simulation step
  - “Auto Simulate” runs steps continuously (every 2 seconds)
  - Charts: predicted vs actual CPU, instances, failure risk, uncertainty
- **Where**: `frontend/src/App.jsx`, calls `frontend/src/api.js`

### “Mock AI mode” fallback
- **What it does**: If ML libraries can’t load, backend generates mock predictions so the app still works.
- **Where**: `backend/main.py` (the `ML_AVAILABLE` branch).

---

## 3. Input to the System

### What kind of data is given as input?
The system uses **time-series-like telemetry rows** (CPU, memory, disk I/O, network, latency, etc.).

There are two “input” sources in this repo:
- **Primary source used by the app**: CSV dataset files under `data/`
  - Backend loads: `data/final/final_dataset_ready.csv` (or builds it from `data/final/final_dataset.csv`)
- **Raw source used to create the dataset**: `data/raw/*.csv` merged/cleaned into final files

### Format of input
- **CSV time-series tabular data**, where each row is a time point and models use:
  - a **12-row window** for workload forecasting (TCN model)
  - a **single-row feature vector** for failure prediction (XGBoost model)

### Example of input (from your real dataset file)
From `data/final/final_dataset_ready.csv` header + first rows, columns include:
- inputs: `cpu_usage`, `memory_usage`, `disk_io`, `network_usage`, `request_rate`, `queue_length`, `latency`, `error_rate`,
  plus time features `hour_of_day`, `day_of_week`, `minute_bucket`, etc.
- targets: `future_cpu_usage`, `future_request_rate`, `future_failure`

Example row (truncated to key fields):
- `cpu_usage=93.23`, `request_rate=60`, `latency=834.32`, `failure_label=1`, `future_cpu_usage=3.36`, `future_failure=0.0`

---

## 4. Output from the System

### What does the system produce as output?
- The backend produces **API responses** and **database rows** that contain the system’s latest decision and metrics:
  - current (simulated) instance count
  - predicted vs actual CPU
  - failure risk probability
  - uncertainty score
  - scaling action (recommendation)
  - latency and cost
  - drift detected flag

Where the output shape comes from in code:
- **API schema**: `backend/schemas.py` (`MetricOut`)
- **SQLite table**: `backend/db_models.py` (`MetricsHistory`)

### Format of output (what types of outputs are produced?)
- **Prediction (regression)**: `predicted_cpu` (float)
- **Risk score (probability)**: `failure_prob` (float in \([0, 1]\))
- **Uncertainty estimate**: `uncertainty` (float; ensemble disagreement)
- **Decision / recommendation**: `action` (string) + `instances` (int)
- **Monitoring signal**: `drift_detected` (boolean)
- **Cost output**: `cost` (float; per interval, computed in simulator)

### Example of output (fields actually returned)
From the backend response model (`MetricOut`), a typical output object has:
- `timestamp` (int)
- `instances` (int)
- `actual_cpu` (float)
- `predicted_cpu` (float)
- `failure_prob` (float)
- `uncertainty` (float)
- `action` (str)
- `latency` (float)
- `cost` (float)
- `drift_detected` (bool)

---

## 5. End-to-End Flow (Step-by-step: input → output)

### Step 1: Backend loads models + data
- File: `backend/main.py`
- On server startup:
  - creates DB tables if needed
  - attempts to load trained model files from `backend/models/`
  - loads `data/final/final_dataset_ready.csv` (or prepares it from `final_dataset.csv` if needed)

### Step 2: Frontend loads and polls the backend
- File: `frontend/src/App.jsx`
- Every ~2 seconds:
  - calls `GET /api/system/status`
  - if models are loaded, also calls:
    - `GET /api/metrics/current`
    - `GET /api/timeline`

### Step 3: (Optional) user triggers training
- UI action: “Train Models”
- Frontend calls: `POST /api/model/train` (see `frontend/src/api.js`)
- Backend runs training in the background by executing `python -m backend.train` (see `backend/main.py`).

### Step 4: User runs a simulation step (or auto-simulates)
- UI actions:
  - “Step Forward” (one step)
  - “Auto Simulate” (repeats every 2 seconds)
- Frontend calls: `POST /api/scale/run`

### Step 5: Backend produces predictions for the current time step
- File: `backend/main.py`
- For time index \(i\), backend computes:
  - `predicted_cpu` + `uncertainty` using the **TCN ensemble** (if ML is available)
  - `failure_prob` using the **XGBoost** model (if ML is available)
- If ML isn’t available, it uses the “Mock AI Mode” branch to generate these values.

### Step 6: Drift detection runs
- File: `backend/src/monitoring/drift_detector.py`
- Backend compares actual CPU vs predicted CPU over a rolling window (MAE) and sets `drift_detected`.

### Step 7: Decision engine chooses scaling action (cost + SLA aware)
- File: `backend/src/decision_engine.py`
- Uses predicted CPU, failure risk, uncertainty, latency, queue length, and budget settings to output:
  - `action`
  - `target_instances`

### Step 8: Simulator applies the action and computes results
- File: `backend/src/simulator.py`
- Applies the new instance count and computes:
  - adjusted CPU based on instance count
  - SLA violation flag (internally)
  - adjusted latency
  - step cost = `instances * 0.10`

### Step 9: Results are stored and returned to the UI
- File: `backend/main.py`
- Writes a row to SQLite table `metrics_history`
- Returns the metrics object to the frontend
- Frontend renders KPIs + charts from `/metrics/current` and `/timeline`

---

## Quick answer to your “features” question

- **Cost optimization**: **Yes** (budget-aware scaling decisions + cost computed each interval and displayed on the dashboard).
- **Energy optimization**: **No** (no energy/power/carbon data is used or produced).
- **Reliability/SLA optimization**: **Yes** (latency/queue/failure-risk triggers aggressive scale-up; simulator tracks overload via adjusted CPU).
- **Monitoring**: **Yes** (drift detection flag based on rolling MAE).
