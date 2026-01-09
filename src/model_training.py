import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def train_model(model_name, X_train, y_train):
    """Обучает выбранную модель."""
    if model_name == "LinearRegression":
        model = LinearRegression()
    elif model_name == "RandomForestRegressor":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_name == "XGBRegressor":
        model = XGBRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Модель {model_name} не поддерживается")
        
    model.fit(X_train, y_train)
    return model

def save_model(model, path):
    """Сохраняет модель в файл."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)