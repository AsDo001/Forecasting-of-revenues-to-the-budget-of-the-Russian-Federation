import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def create_features(df):
    """Формирует признаки X и целевую переменную y."""
    if df is None:
        return None, None
        

    if 'Year' in df.columns:
        df = df.drop("Year", axis=1)
        
    target = "tax_receipts(billion USD)"
    if target not in df.columns:
        raise ValueError(f"Целевая переменная {target} не найдена в данных")
        
    X = df.drop(target, axis=1)
    y = df[target]
    return X, y

def split_data(X, y, test_size=0.3, random_state=42):
    """Разделяет данные на train и test."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def scale_features(X_train, X_test):
    """Масштабирует признаки."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler