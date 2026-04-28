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

class AppState:
    simulator = CloudSimulator(initial_instances=5)
    drift_detector = DriftDetector(window_size=50, threshold=10.0)
    models_loaded = False
    ensemble = None
    fail_model = None
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

app = FastAPI(lifespan=lifespan)

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
                path = f'backend/models/workload_tcn_model_{i}.h5'
                if os.path.exists(path):
                    models.append(load_model(path))
            if len(models) == 5:
                state.ensemble = EnsemblePredictor(models)
                state.fail_model = load_xgboost_model('backend/models/failure_predictor.joblib')
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
def get_status():
    return {
        "status": "training" if state.is_training else "online",
        "models_loaded": state.models_loaded,
        "current_instances": state.simulator.current_instances,
        "recent_drift": False
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
    
    # Cost logic
    action, target_instances, reason = make_scaling_decision(
        predicted_cpu=pred_cpu,
        failure_prob=fail_prob,
        uncertainty_score=uncertainty,
        current_instances=state.simulator.current_instances,
        latency=latency,
        queue_length=queue_length,
        cost_per_instance=0.10,
        budget_used=5.0,
        daily_budget=10.0,
        peak_hours=(9 <= hour <= 17)
    )
    
    record = state.simulator.step(
        timestamp=timestamp,
        actual_cpu=actual_cpu,
        predicted_cpu=pred_cpu,
        failure_prob=fail_prob,
        uncertainty=uncertainty,
        action=action,
        target_instances=target_instances,
        latency=latency
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

def train_models_background():
    state.is_training = True
    try:
        if ML_AVAILABLE:
            import subprocess
            import sys
            subprocess.run([sys.executable, "-m", "backend.train"], cwd="d:/major_project", check=True)
        else:
            time.sleep(2) # Simulate slight delay for mock
            
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
