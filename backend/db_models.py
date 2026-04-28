from sqlalchemy import Column, Integer, Float, String, Boolean
from backend.database import Base

class MetricsHistory(Base):
    __tablename__ = "metrics_history"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(Integer, index=True)
    instances = Column(Integer)
    actual_cpu = Column(Float)
    predicted_cpu = Column(Float)
    failure_prob = Column(Float)
    uncertainty = Column(Float)
    action = Column(String)
    latency = Column(Float)
    cost = Column(Float)
    drift_detected = Column(Boolean, default=False)
