import os
import pandas as pd
import numpy as np
from tensorflow.keras.utils import get_custom_objects

from backend.src.feature_engineering import preprocess_and_engineer_features, create_sequences
from backend.src.models.workload_predictor import build_workload_model, AttentionLayer
from backend.src.models.failure_predictor import train_failure_predictor, save_xgboost_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tcn import TCN

get_custom_objects().update({'TCN': TCN, 'AttentionLayer': AttentionLayer})

def main():
    print("1. Loading optimized user dataset...")
    df_raw = pd.read_csv('data/final/final_dataset.csv')
    
    print("2. Preparing targets...")
    df_processed = preprocess_and_engineer_features(df_raw)
    df_processed.to_csv('data/final/final_dataset_ready.csv', index=False)
    
    # 3. Prepare sequences for Workload Prediction (TCN)
    feature_cols_tcn = [
        'cpu_usage', 'memory_usage', 'disk_io', 'network_usage', 
        'request_rate', 'queue_length', 'cpu_change_rate', 'moving_average_cpu',
        'hour_of_day', 'day_of_week', 'minute_bucket'
    ]
    target_cols = ['future_cpu_usage', 'future_request_rate']
    
    X, y = create_sequences(df_processed, feature_cols_tcn, target_cols, seq_length=12)
    
    print("4. Training Workload Predictor Ensemble (5 models)...")
    if not os.path.exists('backend/models'):
        os.makedirs('backend/models')
        
    num_models = 5
    for i in range(num_models):
        print(f"   Training Model {i+1}/{num_models}...")
        model = build_workload_model(seq_length=12, num_features=len(feature_cols_tcn))
        model.fit(X, y, epochs=5, batch_size=64, validation_split=0.2, verbose=1)
        model.save(f'backend/models/workload_tcn_model_{i}.h5')
        
    print("\n5. Training Failure Predictor (XGBoost)...")
    feature_cols_xgb = [
        'cpu_usage', 'memory_usage', 'disk_io', 'network_usage', 
        'request_rate', 'queue_length', 'latency', 'error_rate', 
        'cpu_change_rate', 'moving_average_cpu',
        'hour_of_day', 'day_of_week', 'minute_bucket'
    ]
    
    X_fail = df_processed[feature_cols_xgb].values
    y_fail = df_processed['future_failure'].values
    
    # 💥 DATA LEAKAGE FIX: Added shuffle=False. Randomly shuffling time-series data leaks future observations into training metrics.
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_fail, y_fail, test_size=0.2, shuffle=False)
    
    # Class Imbalance Scaling (Stage 2 requirements)
    weight = float(np.sum(y_train_f == 0)) / max(np.sum(y_train_f == 1), 1)
    
    xgb_model = train_failure_predictor(X_train_f, y_train_f, scale_pos_weight=weight)
    
    # Detailed Evaluation Metrics emphasizing recall
    y_pred = xgb_model.predict(X_test_f)
    y_prob = xgb_model.predict_proba(X_test_f)[:, 1]
    
    print("\n=== XGBoost Failure Predictor Performance ===")
    print(f"Accuracy:  {accuracy_score(y_test_f, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test_f, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test_f, y_pred, zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(y_test_f, y_pred, zero_division=0):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test_f, y_prob):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test_f, y_pred))
    
    save_xgboost_model(xgb_model, 'backend/models/failure_predictor.joblib')
    print("\nTraining Complete. All models saved in backend/models")

if __name__ == "__main__":
    main()
