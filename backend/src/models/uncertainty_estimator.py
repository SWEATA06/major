import numpy as np

class EnsemblePredictor:
    def __init__(self, models):
        """
        Initialize with a list of trained Keras models.
        """
        self.models = models
        
    def predict_with_uncertainty(self, X):
        """
        Predict using ensemble and compute variance/std.
        Returns mean predictions and std dev (uncertainty).
        """
        predictions = []
        for model in self.models:
            pred = model.predict(X, verbose=0)
            predictions.append(pred)
            
        predictions = np.array(predictions)
        
        # predictions shape: (num_models, batch_size, num_targets)
        mean_preds = np.mean(predictions, axis=0)
        std_preds = np.std(predictions, axis=0)
        
        return mean_preds, std_preds
