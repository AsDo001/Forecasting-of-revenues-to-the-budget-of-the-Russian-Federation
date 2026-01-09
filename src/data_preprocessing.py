import pandas as pd
import numpy as np

def load_data(file_path):
    """Загружает данные из CSV файла."""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return None

def preprocess_data(df):
    """Выполняет предобработку данных на основе логики проекта."""
    if df is None:
        return None
        
    
    df = df.drop_duplicates()

    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    if 'tax_rates(PIT%)' in df.columns:
        df['tax_rates(PIT%)'] = df['tax_rates(PIT%)'].astype(str).str.replace('–', '-')
        def parse_pit(value):
            try:
                if '-' in value:
                    parts = value.split('-')
                    return (float(parts[0]) + float(parts[1])) / 2
                return float(value)
            except:
                return 13.0 
        df['tax_rates(PIT%)'] = df['tax_rates(PIT%)'].apply(parse_pit)


    bounds = {
        "oil_prices(barrel/USD)": (10.0, 150.0),
        "gas_prices(MMBtu/USD)": (1.0, 15.0),
        "Key_rate(%)": (3.0, 25.0),
        "inflation_rate(%)": (0.0, 200.0),
        "exchange_rates(RUB/USD)": (1.0, 120.0),
        "unemployment_rate(%)": (3.0, 15.0),
        "tax_rates(VAT%)": (10.0, 25.0)
    }

    for col, (low, high) in bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=low, upper=high)

    round_cols = ["oil_prices(barrel/USD)", "gas_prices(MMBtu/USD)", "Key_rate(%)", "inflation_rate(%)", "exchange_rates(RUB/USD)"]
    for col in round_cols:
        if col in df.columns:
            df[col] = df[col].round(2)

    return df