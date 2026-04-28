import numpy as np
from collections import deque

class DriftDetector:
    def __init__(self, window_size=50, threshold=15.0):
        self.window_size = window_size
        self.threshold = threshold
        self.actuals = deque(maxlen=window_size)
        self.predicteds = deque(maxlen=window_size)

    def add_record(self, actual, predicted):
        self.actuals.append(actual)
        self.predicteds.append(predicted)

    def check_drift(self):
        if len(self.actuals) < 10:
            return False, 0.0
        
        arr_actual = np.array(self.actuals)
        arr_pred = np.array(self.predicteds)
        
        mae = np.mean(np.abs(arr_actual - arr_pred))
        
        is_drift = mae > self.threshold
        return bool(is_drift), float(mae)
