import joblib
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def predict(model, scaler, X):
    """Выполняет предсказание на новых данных."""
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)

def evaluate_model(y_true, y_pred):
    """Рассчитывает метрики качества."""
    return {
        "R2_Score": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred))
    }