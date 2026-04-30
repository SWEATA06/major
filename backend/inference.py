import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
from tcn import TCN

from backend.src.feature_engineering import preprocess_and_engineer_features, create_sequences
from backend.src.models.workload_predictor import AttentionLayer
from backend.src.models.uncertainty_estimator import EnsemblePredictor
from backend.src.models.failure_predictor import load_xgboost_model
from backend.src.decision_engine import make_scaling_decision
from backend.src.simulator import CloudSimulator
import joblib

get_custom_objects().update({'TCN': TCN, 'AttentionLayer': AttentionLayer})

def main():
    print("Loading models...")
    models = []
    for i in range(5):
        if not os.path.exists(f'backend/models/workload_tcn_model_{i}.h5'):
            print("Error: Models not found. Run train.py first.")
            return
        m = load_model(f'backend/models/workload_tcn_model_{i}.h5')
        models.append(m)
        
    ensemble = EnsemblePredictor(models)
    fail_model = load_xgboost_model('backend/models/failure_predictor.joblib')
    
    print("Loading test data...")
    df = pd.read_csv('data/final/final_dataset_ready.csv')
    
    if len(df) > 500:
        df = df.tail(500).reset_index(drop=True)
    
    # Feature inputs aligned with time extraction
    feature_cols_tcn = [
        'cpu_usage', 'memory_usage', 'disk_io', 'network_usage', 
        'request_rate', 'queue_length', 'cpu_change_rate', 'moving_average_cpu',
        'hour_of_day', 'day_of_week', 'minute_bucket'
    ]
    feature_cols_xgb = [
        'cpu_usage', 'memory_usage', 'disk_io', 'network_usage', 
        'request_rate', 'queue_length', 'latency', 'error_rate', 
        'cpu_change_rate', 'moving_average_cpu',
        'hour_of_day', 'day_of_week', 'minute_bucket'
    ]
    target_cols = ['future_cpu_usage', 'future_request_rate']
    
    print("Loading scalers...")
    feature_scaler = joblib.load('backend/models/feature_scaler.joblib')
    target_scaler = joblib.load('backend/models/target_scaler.joblib')
    
    df_scaled = df.copy()
    df_scaled[feature_cols_tcn] = feature_scaler.transform(df_scaled[feature_cols_tcn])
    df_scaled[target_cols] = target_scaler.transform(df_scaled[target_cols])
    
    X, _ = create_sequences(df_scaled, feature_cols_tcn, target_cols, seq_length=12)
    _, y_actual = create_sequences(df, feature_cols_tcn, target_cols, seq_length=12)
    
    simulator = CloudSimulator(initial_instances=5)
    
    print("Running Simulation...")
    for i in range(len(X)):
        current_seq = X[i:i+1] # shape (1, 12, features)
        
        preds_mean_scaled, preds_std_scaled = ensemble.predict_with_uncertainty(current_seq)
        preds_mean = target_scaler.inverse_transform(preds_mean_scaled)
        
        pred_cpu = preds_mean[0][0]
        # Approximate uncertainty scaling for CPU
        cpu_scale = target_scaler.data_max_[0] - target_scaler.data_min_[0]
        uncertainty = preds_std_scaled[0][0] * cpu_scale
        
        current_xgb_slice = df[feature_cols_xgb].iloc[i + 11].values.reshape(1, -1)
        
        try:
            fail_prob = fail_model.predict_proba(current_xgb_slice)[0][1]
        except IndexError:
            fail_prob = fail_model.predict_proba(current_xgb_slice)[0][0]
        
        action, target_instances = make_scaling_decision(
            predicted_cpu=pred_cpu,
            failure_prob=fail_prob,
            uncertainty_score=uncertainty,
            current_instances=simulator.current_instances
        )
        
        actual_cpu = y_actual[i][0]
        latency = df.iloc[i + 11]['latency']
        
        timestamp = df.iloc[12+i]['timestamp'] if 'timestamp' in df.columns else i
        
        simulator.step(
            timestamp=timestamp,
            actual_cpu=actual_cpu,
            predicted_cpu=pred_cpu,
            failure_prob=fail_prob,
            uncertainty=uncertainty,
            action=action,
            target_instances=target_instances,
            latency=latency
        )
        
    summary = simulator.get_summary()
    print("\nSimulation metrics:")
    print(f"Total Cost: ${summary['total_cost']:.2f}")
    print(f"Total SLA Violations (intervals >95% Load): {summary['total_sla_violations']}")
    print(f"Average Latency: {summary['average_latency']:.2f}ms")
    
    summary['history_df'].to_csv('data/simulation_results.csv', index=False)
    print("Saved simulation_results.csv")

if __name__ == "__main__":
    main()
