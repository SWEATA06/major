import pandas as pd

class CloudSimulator:
    def __init__(self, initial_instances=5):
        self.current_instances = initial_instances
        self.history = []
        self.cost_per_instance = 0.10 # $0.10 per interval
        
    def step(self, timestamp, actual_cpu, predicted_cpu, failure_prob, uncertainty, action, target_instances, latency):
        # Apply scaling decision
        self.current_instances = target_instances
        
        # Calculate real CPU based on instances
        # If we didn't scale up enough and traffic hit, actual CPU goes higher
        # Assuming actual_cpu in the dataset assumes a baseline of 5 instances
        adjusted_cpu = actual_cpu * (5 / self.current_instances)
        
        # SLA violation if adjusted CPU > 95%
        sla_violation = 1 if adjusted_cpu > 95 else 0
        
        # Adjust latency based on load
        if adjusted_cpu > 90:
            adjusted_latency = latency * (adjusted_cpu / 90)
        else:
            adjusted_latency = latency
            
        step_cost = self.current_instances * self.cost_per_instance
        
        record = {
            'timestamp': timestamp,
            'instances': self.current_instances,
            'adjusted_cpu': adjusted_cpu,
            'predicted_cpu': predicted_cpu,
            'failure_prob': failure_prob,
            'uncertainty': uncertainty,
            'action': action,
            'sla_violation': sla_violation,
            'latency': adjusted_latency,
            'cost': step_cost
        }
        
        self.history.append(record)
        return record
        
    def get_summary(self):
        df = pd.DataFrame(self.history)
        total_cost = df['cost'].sum() if not df.empty else 0
        total_violations = df['sla_violation'].sum() if not df.empty else 0
        avg_latency = df['latency'].mean() if not df.empty else 0
        
        return {
            'total_cost': total_cost,
            'total_sla_violations': total_violations,
            'average_latency': avg_latency,
            'history_df': df
        }
