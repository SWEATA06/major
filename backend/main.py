import os
import time
import asyncio
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager

from backend import database, db_models, schemas
from backend.src.monitoring.drift_detector import DriftDetector
from backend.src.simulator import CloudSimulator
from backend.src.decision_engine import make_scaling_decision

# ML Imports
import pandas as pd
import numpy as np

try:
    from tensorflow.keras.models import load_model    
    from tensorflow.keras.utils import get_custom_objects
    from tcn import TCN
    from backend.src.models.workload_predictor import AttentionLayer
    from backend.src.models.uncertainty_estimator import EnsemblePredictor
    from backend.src.models.failure_predictor import load_xgboost_model
    from backend.src.feature_engineering import preprocess_and_engineer_features, create_sequences
    get_custom_objects().update({'TCN': TCN, 'AttentionLayer': AttentionLayer})
    ML_AVAILABLE = True
except ImportError as e:
    ML_AVAILABLE = False
    print(f"ML imports failed: {e}")

database.Base.metadata.create_all(bind=database.engine)


def _ensure_energy_columns():
    """
    Idempotent lightweight migration: add energy/carbon columns to an existing
    metrics_history table if they are missing. SQLAlchemy's create_all does not
    ALTER existing tables, so older databases need these columns added so new
    inserts don't fail.
    """
    from sqlalchemy import text
    new_columns = {
        "energy_kwh": "FLOAT DEFAULT 0.0",
        "carbon_g": "FLOAT DEFAULT 0.0",
        "carbon_intensity": "FLOAT DEFAULT 0.0",
    }
    try:
        with database.engine.connect() as conn:
            existing = conn.execute(text("PRAGMA table_info(metrics_history)")).fetchall()
            existing_cols = {row[1] for row in existing}
            for col, ddl in new_columns.items():
                if col not in existing_cols:
                    conn.execute(text(f"ALTER TABLE metrics_history ADD COLUMN {col} {ddl}"))
            conn.commit()
    except Exception as e:
        print(f"WARNING: energy column migration skipped: {e}")


_ensure_energy_columns()

class AppState:
    simulator = CloudSimulator(initial_instances=5)
    drift_detector = DriftDetector(window_size=50, threshold=10.0)
    models_loaded = False
    ensemble = None
    fail_model = None
    feature_scaler = None
    target_scaler = None
    data_df = None
    X_seq = None
    y_seq = None
    current_step = 0
    max_steps = 0
    is_training = False

state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load Models if exist
    load_models()
    load_data()
    yield
    # Shutdown
    pass

from fastapi.staticfiles import StaticFiles

app = FastAPI(lifespan=lifespan)

charts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "public", "charts"))
if not os.path.exists(charts_dir):
    os.makedirs(charts_dir, exist_ok=True)
app.mount("/charts", StaticFiles(directory=charts_dir), name="charts")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

def load_models():
    try:
        if ML_AVAILABLE:
            models = []
            for i in range(5):
                # Prefer native Keras format; fall back to legacy HDF5 if needed
                path_keras = f'backend/models/workload_tcn_model_{i}.keras'
                path_h5 = f'backend/models/workload_tcn_model_{i}.h5'
                if os.path.exists(path_keras):
                    models.append(load_model(path_keras))
                elif os.path.exists(path_h5):
                    models.append(load_model(path_h5))
            if len(models) == 5:
                state.ensemble = EnsemblePredictor(models)
                state.fail_model = load_xgboost_model('backend/models/failure_predictor.joblib')
                try:
                    import joblib
                    fs_path = 'backend/models/feature_scaler.joblib'
                    ts_path = 'backend/models/target_scaler.joblib'
                    if os.path.exists(fs_path) and os.path.exists(ts_path):
                        state.feature_scaler = joblib.load(fs_path)
                        state.target_scaler = joblib.load(ts_path)
                        print("Scalers loaded successfully")
                    else:
                        state.feature_scaler = None
                        state.target_scaler = None
                        print("WARNING: Scaler files not found; predictions will use raw scale.")
                except Exception as se:
                    state.feature_scaler = None
                    state.target_scaler = None
                    print(f"WARNING: Failed to load scalers: {se}")
                state.models_loaded = True
                print("Models loaded successfully")
                return
            else:
                print("No models found in backend/models/ directory. Awaiting training trigger.")
    except Exception as e:
        print(f"Error loading real models: {e}")

    if not ML_AVAILABLE:
        print("Hardware limitation detected (AVX/Tensorflow). Enabling Mock AI Mode...")
    state.models_loaded = False if ML_AVAILABLE else True

def load_data():
    try:
        path = 'data/final/final_dataset_ready.csv'
        if not os.path.exists(path):
            print("Ready dataset not found. Generating from raw...")
            raw_path = 'data/final/final_dataset.csv'
            if os.path.exists(raw_path):
                from backend.src.feature_engineering import preprocess_and_engineer_features
                raw_df = pd.read_csv(raw_path)
                processed = preprocess_and_engineer_features(raw_df)
                processed.to_csv(path, index=False)
            else:
                print("No datasets found at all!")
                return
                
        if os.path.exists(path):
            df = pd.read_csv(path)
            if len(df) > 500:
                df = df.tail(500).reset_index(drop=True)
            state.data_df = df
            
            if ML_AVAILABLE:
                feature_cols_tcn = [
                    'cpu_usage', 'memory_usage', 'disk_io', 'network_usage', 
                    'request_rate', 'queue_length', 'cpu_change_rate', 'moving_average_cpu',
                    'hour_of_day', 'day_of_week', 'minute_bucket'
                ]
                target_cols = ['future_cpu_usage', 'future_request_rate']
                if state.feature_scaler is not None and state.target_scaler is not None:
                    df_scaled = df.copy()
                    df_scaled[feature_cols_tcn] = state.feature_scaler.transform(df_scaled[feature_cols_tcn])
                    X, _ = create_sequences(df_scaled, feature_cols_tcn, target_cols, seq_length=12)
                    _, y = create_sequences(df, feature_cols_tcn, target_cols, seq_length=12)
                else:
                    X, y = create_sequences(df, feature_cols_tcn, target_cols, seq_length=12)
                state.X_seq = X
                state.y_seq = y
                state.max_steps = len(X)
            else:
                state.max_steps = len(df) - 15
            state.current_step = 0
    except Exception as e:
        print(f"Error loading data: {e}")

@app.get("/api/system/status")
def get_status(db: Session = Depends(get_db)):
    recent_drift = False
    try:
        latest = db.query(db_models.MetricsHistory).order_by(db_models.MetricsHistory.id.desc()).first()
        if latest is not None:
            recent_drift = bool(latest.drift_detected)
    except Exception:
        recent_drift = False
    return {
        "status": "training" if state.is_training else "online",
        "models_loaded": state.models_loaded,
        "current_instances": state.simulator.current_instances,
        "recent_drift": recent_drift
    }

@app.post("/api/scale/run")
def trigger_simulation_step(db: Session = Depends(get_db)):
    if not state.models_loaded or state.current_step >= state.max_steps:
        if state.is_training:
            raise HTTPException(status_code=400, detail="Models are currently training.")
        raise HTTPException(status_code=400, detail="Models not loaded or end of data.")
        
    i = state.current_step
    df = state.data_df
    
    if ML_AVAILABLE:
        current_seq = state.X_seq[i:i+1]
        preds_mean, preds_std = state.ensemble.predict_with_uncertainty(current_seq)
        if state.target_scaler is not None:
            preds_mean_real = state.target_scaler.inverse_transform(preds_mean)
            pred_cpu = float(preds_mean_real[0][0])
            cpu_scale = float(state.target_scaler.data_max_[0] - state.target_scaler.data_min_[0])
            uncertainty = float(preds_std[0][0]) * cpu_scale
        else:
            pred_cpu = float(preds_mean[0][0])
            uncertainty = float(preds_std[0][0])
        
        feature_cols_xgb = [
            'cpu_usage', 'memory_usage', 'disk_io', 'network_usage', 
            'request_rate', 'queue_length', 'latency', 'error_rate', 
            'cpu_change_rate', 'moving_average_cpu',
            'hour_of_day', 'day_of_week', 'minute_bucket'
        ]
        current_xgb_slice = df[feature_cols_xgb].iloc[i + 11].values.reshape(1, -1)
        
        try:
            fail_prob = float(state.fail_model.predict_proba(current_xgb_slice)[0][1])
        except IndexError:
            fail_prob = float(state.fail_model.predict_proba(current_xgb_slice)[0][0])
            
        actual_cpu = float(state.y_seq[i][0])
    else:
        # MOCK DATA INJECTION (AVX CPU Fallback)
        actual_cpu = float(df.iloc[i + 11]['cpu_usage'])
        pred_cpu = actual_cpu + float(np.random.normal(0, 5.0))
        pred_cpu = min(max(pred_cpu, 0.0), 100.0)
        uncertainty = float(np.random.uniform(1.0, 15.0))
        fail_prob = 0.8 if pred_cpu > 85 else float(np.random.uniform(0.01, 0.3))
        
    latency = float(df.iloc[i + 11]['latency'])
    queue_length = float(df.iloc[i + 11]['queue_length'])
    hour = int(df.iloc[i + 11]['hour_of_day'])
    timestamp = int(time.time() * 1000)
    
    # Drift
    state.drift_detector.add_record(actual_cpu, pred_cpu)
    is_drift, drift_value = state.drift_detector.check_drift()
    
    # Cost logic. budget_used reflects the simulator's real accumulated spend
    # so the cost-aware tier responds to actual usage instead of a constant.
    _summary = state.simulator.get_summary()
    budget_used = float(_summary.get('total_cost', 0.0))
    from backend.src.energy import carbon_intensity as _carbon_intensity
    current_carbon = _carbon_intensity(hour)
    action, target_instances, reason = make_scaling_decision(
        predicted_cpu=pred_cpu,
        failure_prob=fail_prob,
        uncertainty_score=uncertainty,
        current_instances=state.simulator.current_instances,
        latency=latency,
        queue_length=queue_length,
        cost_per_instance=0.10,
        budget_used=budget_used,
        daily_budget=10.0,
        peak_hours=(9 <= hour <= 17),
        carbon_intensity=current_carbon
    )
    
    record = state.simulator.step(
        timestamp=timestamp,
        actual_cpu=actual_cpu,
        predicted_cpu=pred_cpu,
        failure_prob=fail_prob,
        uncertainty=uncertainty,
        action=action,
        target_instances=target_instances,
        latency=latency,
        hour=hour
    )
    
    # Save to DB
    db_metric = db_models.MetricsHistory(
        timestamp=int(record['timestamp']),
        instances=int(record['instances']),
        actual_cpu=float(record['adjusted_cpu']),
        predicted_cpu=float(record['predicted_cpu']),
        failure_prob=float(record['failure_prob']),
        uncertainty=float(record['uncertainty']),
        action=action,
        latency=float(record['latency']),
        cost=float(record['cost']),
        energy_kwh=float(record.get('energy_kwh', 0.0)),
        carbon_g=float(record.get('carbon_g', 0.0)),
        carbon_intensity=float(record.get('carbon_intensity', 0.0)),
        drift_detected=is_drift
    )
    db.add(db_metric)
    db.commit()
    db.refresh(db_metric)
    
    state.current_step += 1
    
    return db_metric

@app.get("/api/metrics/current", response_model=schemas.MetricOut)
def current_metrics(db: Session = Depends(get_db)):
    metric = db.query(db_models.MetricsHistory).order_by(db_models.MetricsHistory.id.desc()).first()
    if not metric:
        return schemas.MetricOut(
            id=0, timestamp=0, instances=0, actual_cpu=0, predicted_cpu=0,
            failure_prob=0, uncertainty=0, action="hold", latency=0, cost=0, drift_detected=False
        )
    return metric

@app.get("/api/timeline")
def timeline(db: Session = Depends(get_db), limit: int = 100):
    metrics = db.query(db_models.MetricsHistory).order_by(db_models.MetricsHistory.id.desc()).limit(limit).all()
    return list(reversed(metrics))

@app.get("/api/energy/summary")
def energy_summary(db: Session = Depends(get_db)):
    """
    Aggregated energy and carbon metrics across all recorded simulation steps.
    Powers the sustainability KPI cards on the dashboard.
    """
    from sqlalchemy import func
    q = db.query(
        func.coalesce(func.sum(db_models.MetricsHistory.energy_kwh), 0.0),
        func.coalesce(func.sum(db_models.MetricsHistory.carbon_g), 0.0),
        func.coalesce(func.avg(db_models.MetricsHistory.carbon_intensity), 0.0),
        func.coalesce(func.sum(db_models.MetricsHistory.cost), 0.0),
        func.count(db_models.MetricsHistory.id),
    ).one()
    total_energy, total_carbon, avg_intensity, total_cost, steps = q
    return {
        "total_energy_kwh": round(float(total_energy), 4),
        "total_carbon_g": round(float(total_carbon), 2),
        "total_carbon_kg": round(float(total_carbon) / 1000.0, 4),
        "avg_carbon_intensity": round(float(avg_intensity), 2),
        "total_cost": round(float(total_cost), 2),
        "steps": int(steps),
    }

@app.get("/api/predictions/latest")
def latest_predictions(db: Session = Depends(get_db)):
    """
    Lightweight endpoint for real-time charts.
    Returns the latest simulation outputs in the shape:
      { "actual": number, "predicted": number, "timestamp": number }
    """
    metric = db.query(db_models.MetricsHistory).order_by(db_models.MetricsHistory.id.desc()).first()
    if not metric:
        return {"actual": 0.0, "predicted": 0.0, "timestamp": 0}
    return {"actual": float(metric.actual_cpu), "predicted": float(metric.predicted_cpu), "timestamp": int(metric.timestamp)}

def train_models_background():
    state.is_training = True
    try:
        if ML_AVAILABLE:
            import subprocess
            import sys
            import os

            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            train_py = os.path.join(repo_root, "backend", "train.py")
            env = os.environ.copy()
            # Ensure project root is on sys.path for imports like `from backend...`
            env["PYTHONPATH"] = repo_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

            subprocess.run([sys.executable, train_py], cwd=repo_root, env=env, check=True)
        else:
            time.sleep(2)  # Simulate slight delay for mock

        load_models()
        load_data()
    except Exception as e:
        print(f"Training error: {e}")
    finally:
        state.is_training = False

@app.post("/api/model/train")
def train_model(background_tasks: BackgroundTasks):
    if state.is_training:
        return {"status": "training already in progress"}
    
    background_tasks.add_task(train_models_background)
    return {"status": "training started", "message": "Using data/final/final_dataset.csv"}

@app.get("/api/comparison/base-paper")
def get_base_paper_comparison():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    metrics_path = os.path.join(repo_root, "frontend", "public", "charts", "base_paper_comparison_metrics.json")
    if not os.path.exists(metrics_path):
        metrics_path = os.path.join(repo_root, "data", "charts", "base_paper_comparison_metrics.json")
    
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path, "r") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Comparison metrics not generated yet. Please run train_and_compare_base_paper.py.")

def run_base_paper_comparison_background():
    try:
        import subprocess
        import sys
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        script = os.path.join(repo_root, "train_and_compare_base_paper.py")
        env = os.environ.copy()
        env["PYTHONPATH"] = repo_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        subprocess.run([sys.executable, script], cwd=repo_root, env=env, check=True)
    except Exception as e:
        print(f"Error running base paper comparison: {e}")

@app.post("/api/comparison/base-paper/run")
def run_base_paper_comparison_endpoint(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_base_paper_comparison_background)
    return {"status": "comparison benchmark started in background"}

