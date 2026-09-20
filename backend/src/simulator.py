import pandas as pd

from backend.src.energy import energy_kwh, carbon_grams, carbon_intensity


class CloudSimulator:
    def __init__(self, initial_instances=5):
        self.current_instances = initial_instances
        self.history = []
        self.cost_per_instance = 0.10 # $0.10 per interval
        
    def step(self, timestamp, actual_cpu, predicted_cpu, failure_prob, uncertainty, action, target_instances, latency, hour=None):
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

        # Energy & carbon for this interval. Power scales with the actual load
        # the running instances are carrying (adjusted_cpu), not just the count.
        step_energy_kwh = energy_kwh(self.current_instances, adjusted_cpu)
        step_carbon_g = carbon_grams(step_energy_kwh, hour) if hour is not None else 0.0
        step_carbon_intensity = carbon_intensity(hour) if hour is not None else 0.0
        
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
            'cost': step_cost,
            'energy_kwh': step_energy_kwh,
            'carbon_g': step_carbon_g,
            'carbon_intensity': step_carbon_intensity
        }
        
        self.history.append(record)
        return record
        
    def get_summary(self):
        df = pd.DataFrame(self.history)
        total_cost = df['cost'].sum() if not df.empty else 0
        total_violations = df['sla_violation'].sum() if not df.empty else 0
        avg_latency = df['latency'].mean() if not df.empty else 0
        total_energy = df['energy_kwh'].sum() if not df.empty and 'energy_kwh' in df else 0
        total_carbon = df['carbon_g'].sum() if not df.empty and 'carbon_g' in df else 0
        
        return {
            'total_cost': total_cost,
            'total_sla_violations': total_violations,
            'average_latency': avg_latency,
            'total_energy_kwh': total_energy,
            'total_carbon_g': total_carbon,
            'history_df': df
        }
