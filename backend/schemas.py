from pydantic import BaseModel
from typing import Optional, List

class MetricOut(BaseModel):
    id: int
    timestamp: int
    instances: int
    actual_cpu: float
    predicted_cpu: float
    failure_prob: float
    uncertainty: float
    action: str
    latency: float
    cost: float
    drift_detected: bool

    class Config:
        from_attributes = True

class SystemStatus(BaseModel):
    status: str
    models_loaded: bool
    current_instances: int
    recent_drift: bool
