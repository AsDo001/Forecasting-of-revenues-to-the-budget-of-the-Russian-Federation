import pandas as pd
import numpy as np

def load_data(file_path):
    """Загружает данные из CSV файла."""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """Выполняет предобработку данных."""
    # Удаление дубликатов
    df.drop_duplicates(inplace=True)

    # Обработка пропусков (пример: заполнение медианой)
    # В вашем ноутбуке пропуски обрабатываются позже, но для пайплайна лучше сделать это здесь
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())

    # Обработка специфических колонок (из вашего main.ipynb)
    # 'tax_rates(PIT%)' - это строка, нужно преобразовать
    if 'tax_rates(PIT%)' in df.columns:
        df['tax_rates(PIT%)'] = df['tax_rates(PIT%)'].astype(str).str.replace('–', '-')
        # Пример обработки диапазона '12-60' -> 36 (среднее)
        # Это упрощенная логика, возможно, потребуется более сложная обработка
        def parse_pit(value):
            if '-' in value:
                parts = value.split('-')
                return (float(parts[0]) + float(parts[1])) / 2
            else:
                return float(value)
        df['tax_rates(PIT%)'] = df['tax_rates(PIT%)'].apply(parse_pit)

    # Аугментация/Клиппинг (из вашего main.ipynb)
    # Предполагается, что aug_df - это df после обработки
    aug_df = df.copy()
    bounds = {
        "oil_prices(barrel/USD)": (10.0, 150.0),
        "gas_prices(MMBtu/USD)": (1.0, 15.0),
        "Oil_production_volume(million_b/y)": (2.0, 6.0),
        "Gas_production_volume(billion_c_m/y)": (500.0, 800.0),
        "Oil_export_volume(million tons)": (80.0, 250.0),
        "Gas_export_volume(billion_c_m)": (150.0, 250.0),
        "Share_of_oil_and_gas_revenues(%)": (10.0, 60.0),
        "TB(billion USD)": (-5000.0, 100000.0),
        "FDI(billion USD)": (-5000.0, 100000.0),
        "Import_volume(billion USD)": (20.0, 400.0),
        "Key_rate(%)": (3.0, 20.0),
        "level_of_public_debt(% of GDP)": (5.0, 20.0),
        "tock_Market_Index(MOEX Index)": (100.0, 4000.0),
        "inflation_rate(%)": (0.0, 200.0),
        "exchange_rates(RUB/USD)": (1.0, 100.0),
        "GNP(milliard USD)": (100.0, 2000.0),
        "ISI(0-10)": (0.0, 10.0),
        "Migration_rate(net_migration th/p)": (-500.0, 500.0),
        "Gini_coefficient(%)": (20.0, 50.0),
        "population_size(p)": (140000000.0, 150000000.0),
        "unemployment_rate(%)": (4.0, 15.0),
        "per_c_i(thousands/USD)": (1.0, 20.0),
        "Non_oil_GDP(%)": (50.0, 90.0),
        "CPI": (0.0, 500.0),
        "Military_expenditures(% of GDP)": (2.0, 10.0),
        "tax_rates(VAT%)": (10.0, 25.0),
        "tax_rates(PIT%)": (10.0, 20.0),
    }

    for c, (lo, hi) in bounds.items():
        if c in aug_df.columns:
            aug_df[c] = aug_df[c].clip(lower=lo, upper=hi)

    round_2_cols = [
        "oil_prices(barrel/USD)", "gas_prices(MMBtu/USD)",
        "Key_rate(%)", "inflation_rate(%)", "exchange_rates(RUB/USD)",
        "unemployment_rate(%)", "Gini_coefficient(%)", "Share_of_oil_and_gas_revenues(%)"
    ]

    for c in round_2_cols:
        if c in aug_df.columns:
            aug_df[c] = aug_df[c].round(2)

    return aug_df