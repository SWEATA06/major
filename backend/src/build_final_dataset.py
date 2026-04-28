import pandas as pd
import numpy as np
import os

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    in_path = os.path.join(base_dir, 'data', 'processed', 'cleaned_dataset.csv')
    out_dir = os.path.join(base_dir, 'data', 'final')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'final_dataset.csv')
    
    df = pd.read_csv(in_path)

    # Base Metrics ensuring positive floats
    df["cpu_usage"] = np.maximum(df["CPU usage [%]"].astype(float), 0)
    df["memory_usage"] = np.maximum(((df["Memory usage [KB]"] / df["Memory capacity provisioned [KB]"]) * 100).astype(float), 0)
    df["disk_io"] = np.maximum((df["Disk read throughput [KB/s]"] + df["Disk write throughput [KB/s]"]).astype(float), 0)
    df["network_usage"] = np.maximum((df["Network received throughput [KB/s]"] + df["Network transmitted throughput [KB/s]"]).astype(float), 0)
    
    # Handle infinities defensively
    for col in ["cpu_usage", "memory_usage", "disk_io", "network_usage"]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(df[col].mean())

    if 'CPU cores' in df.columns:
        df['CPU cores'] = df['CPU cores'].astype(int)

    df["cpu_change_rate"] = df["cpu_usage"].diff().fillna(0).astype(float)
    df["moving_average_cpu"] = df["cpu_usage"].rolling(5).mean().bfill().astype(float)

    np.random.seed(42)
    
    # Request rate (ensure integer and no negatives)
    df["request_rate"] = (df["network_usage"] * 0.1) + np.random.normal(50, 20, len(df))
    df["request_rate"] = np.maximum(df["request_rate"], 10)
    df["request_rate"] = np.round(df["request_rate"]).astype(int)

    # Queue length (ensure integer)
    req_rolling = df["request_rate"].rolling(10).mean().bfill()
    df["queue_length"] = np.where(df["request_rate"] > req_rolling * 1.2, 
                                  df["request_rate"] * 0.4, 0)
    df["queue_length"] = np.round(df["queue_length"]).astype(int)

    # Latency & Error Rate
    df["latency"] = 10 + (df["disk_io"] * 0.05) + (df["queue_length"] * 2.0) + (df["cpu_usage"] * 0.2) + np.random.normal(5, 2, len(df))
    df["latency"] = np.maximum(df["latency"], 0.0).astype(float)

    lat_q80 = df["latency"].quantile(0.80)
    df["error_rate"] = np.where((df["latency"] > lat_q80) | (df["queue_length"] > 15),
                                np.random.uniform(2, 6, len(df)),
                                np.random.uniform(0, 0.5, len(df)))
    df["error_rate"] = np.maximum(df["error_rate"], 0.0).astype(float)

    # Labels (Integer forced)
    cpu_high = df["cpu_usage"].quantile(0.75)
    mem_high = df["memory_usage"].quantile(0.80)
    lat_high = df["latency"].quantile(0.80)
    
    cpu_critical = df["cpu_usage"].quantile(0.90)
    lat_critical = df["latency"].quantile(0.92)

    df["overload_label"] = np.where(
        (df["cpu_usage"] >= cpu_high) |
        (df["memory_usage"] >= mem_high) |
        (df["latency"] >= lat_high),
        1, 0
    ).astype(int)

    df["failure_label"] = np.where(
        ((df["cpu_usage"] >= cpu_critical) & (df["latency"] >= lat_high)) |
        (df["latency"] >= lat_critical) |
        ((df["cpu_usage"] >= cpu_high) & (df["error_rate"] > 1.0)),
        1, 0
    ).astype(int)
    
    # Timestamp conversions and dropping
    if 'Timestamp [ms]' in df.columns:
        # Detect unit dynamically (seconds vs ms Unix)
        try:
            raw_ts = df['Timestamp [ms]']
            if raw_ts.median() > 1e11: # usually means ms
                timestamp_dt = pd.to_datetime(raw_ts, unit='ms')
            else:
                timestamp_dt = pd.to_datetime(raw_ts, unit='s')
        except:
            timestamp_dt = pd.to_datetime(df['Timestamp [ms]'], errors='coerce')
        
        # We also create a raw 'timestamp' if simulator needs to track indices chronologically later
        df['timestamp'] = np.arange(len(df)) # safe incremental index
        
        df['hour_of_day'] = timestamp_dt.dt.hour.fillna(0).astype(int)
        df['day_of_week'] = timestamp_dt.dt.dayofweek.fillna(0).astype(int)
        df['minute_bucket'] = (timestamp_dt.dt.minute // 5).fillna(0).astype(int)

    # Purge redundant/duplicate leaky metadata
    cols_to_drop = [
        "CPU usage [%]", "Memory usage [KB]", "Memory capacity provisioned [KB]",
        "Disk read throughput [KB/s]", "Disk write throughput [KB/s]",
        "Network received throughput [KB/s]", "Network transmitted throughput [KB/s]",
        "CPU capacity provisioned [MHZ]", "CPU usage [MHZ]", "Timestamp [ms]"
    ]
    
    dropped_list = []
    for c in cols_to_drop:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)
            dropped_list.append(c)
            
    df = df.dropna()

    df.to_csv(out_path, index=False)
    
    # Print formatted report
    print("\n=== DATASET AUDIT REPORT ===")
    print(f"Final shape: {df.shape}")
    print(f"\n[Removed Redundancies]: {dropped_list}")
    print(f"\n[Fixed Values]: Zeroed all negative counts. Replaced NaNs/Inf. Forced categorical integer distributions.")
    
    int_cols = df.select_dtypes(include=['int32', 'int64']).columns.tolist()
    float_cols = df.select_dtypes(include=['float32', 'float64']).columns.tolist()
    print(f"\n[Integer Columns] (Time features + identifiers + classes):\n{int_cols}")
    print(f"\n[Float Columns] (Continuous variables):\n{float_cols}")
    
    print("\n[Target Balance]")
    print(f"Overload Label: {df['overload_label'].value_counts(normalize=True).to_dict()}")
    print(f"Failure Label: {df['failure_label'].value_counts(normalize=True).to_dict()}")

if __name__ == "__main__":
    main()