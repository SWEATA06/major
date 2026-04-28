import pandas as pd
import numpy as np

def preprocess_and_engineer_features(df):
    """
    Apply final feature engineering necessary for the model targets.
    The user dataset already contains the core engineered features.
    """
    df = df.copy()
    
    # Process timestamp if available as 'Timestamp [ms]'
    if 'Timestamp [ms]' in df.columns:
        # Some timestamp columns are Unix ms, others might be string format. 
        # Attempt conversion if it's numeric, otherwise regular datetime
        try:
            df['timestamp'] = pd.to_datetime(df['Timestamp [ms]'], unit='ms')
        except:
            df['timestamp'] = pd.to_datetime(df['Timestamp [ms]'], errors='coerce')
    
    # We still need to create future labels for Model 1 (Workload Prediction)
    df['future_cpu_usage'] = df['cpu_usage'].shift(-1)
    df['future_request_rate'] = df['request_rate'].shift(-1)
    
    # User provided 'failure_label', we shift it to predict future failure
    if 'failure_label' in df.columns:
        df['future_failure'] = df['failure_label'].shift(-1)
        
    # Drop NaN created by shifting
    df = df.dropna().reset_index(drop=True)
    
    return df

def create_sequences(df, feature_cols, target_cols, seq_length=12):
    X, y = [], []
    data_X = df[feature_cols].values
    data_y = df[target_cols].values
    
    for i in range(len(df) - seq_length):
        X.append(data_X[i:i+seq_length])
        y.append(data_y[i+seq_length])
        
    return np.array(X), np.array(y)
