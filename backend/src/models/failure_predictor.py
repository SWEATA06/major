import xgboost as xgb
import joblib

def train_failure_predictor(X_train, y_train, scale_pos_weight=1.0):
    """
    Train an XGBoost model for failure/overload prediction.
    Utilizes weighted parameters to heavily penalize missing actual failures.
    """
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,       # Handling Stage 2 imbalances
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def save_xgboost_model(model, filepath='models/failure_predictor.joblib'):
    joblib.dump(model, filepath)

def load_xgboost_model(filepath='models/failure_predictor.joblib'):
    return joblib.load(filepath)
