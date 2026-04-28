import pandas as pd
import numpy as np

def generate_telemetry_data(num_samples=10000, output_path='data/telemetry.csv'):
    """Generate synthetic cloud telemetry data."""
    np.random.seed(42)
    
    # Time generation
    timestamps = pd.date_range(start='2026-01-01', periods=num_samples, freq='5min')
    
    # Base CPU usage with day/night cycles
    hours = timestamps.hour
    base_cpu = 30 + 20 * np.sin(np.pi * (hours - 6) / 12) 
    
    # Add random noise and spikes
    cpu_usage = base_cpu + np.random.normal(0, 5, num_samples)
    
    # Introduce some anomalies / high load periods
    anomaly_indices = np.random.choice(num_samples, int(num_samples * 0.05), replace=False)
    cpu_usage[anomaly_indices] += np.random.normal(30, 10, len(anomaly_indices))
    
    cpu_usage = np.clip(cpu_usage, 0, 100)
    
    # Correlated metrics
    memory_usage = cpu_usage * 0.7 + np.random.normal(10, 5, num_samples)
    memory_usage = np.clip(memory_usage, 0, 100)
    
    request_rate = cpu_usage * 10 + np.random.normal(50, 20, num_samples)
    request_rate = np.clip(request_rate, 0, None)
    
    # Metrics influenced by high CPU/Traffic (representing load)
    network_usage = request_rate * 2.5 + np.random.normal(10, 5, num_samples)
    
    disk_io = request_rate * 0.8 + np.random.normal(5, 2, num_samples)
    
    queue_length = np.where(cpu_usage > 80, np.random.poisson(50, num_samples), np.random.poisson(5, num_samples))
    
    latency = np.where(cpu_usage > 85, np.random.normal(500, 100, num_samples), np.random.normal(50, 10, num_samples))
    latency = np.clip(latency, 10, None) # Min 10ms
    
    error_rate = np.where(cpu_usage > 90, np.random.uniform(0.05, 0.2, num_samples), np.random.uniform(0, 0.01, num_samples))
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'cpu_usage': cpu_usage,
        'memory_usage': memory_usage,
        'disk_io': disk_io,
        'network_usage': network_usage,
        'request_rate': request_rate,
        'queue_length': queue_length,
        'latency': latency,
        'error_rate': error_rate
    })
    
    df.to_csv(output_path, index=False)
    print(f"Generated {num_samples} records and saved to {output_path}")
    return df

if __name__ == "__main__":
    import os
    os.makedirs('data', exist_ok=True)
    generate_telemetry_data()
