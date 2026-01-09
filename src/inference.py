import joblib
import numpy as np
import os
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def load_model_and_scaler(model_path, scaler_path):
    """Загружает модель и скейлер из файлов."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель не найдена по пути: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Скейлер не найден по пути: {scaler_path}")
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

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