import pandas as pd
import os
import glob
import joblib
import sys
from src.data_preprocessing import preprocess_data
from src.feature_engineering import create_features
from src.inference import predict

def run_prediction(input_file, model_type="lin_reg"):
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, "src", "models", f"{model_type}.joblib")
    scaler_path = os.path.join(base_path, "src", "models", f"{model_type}_scaler.joblib")
    
    print(f"\n--- Запуск предсказания для файла: {os.path.basename(input_file)} (Модель: {model_type}) ---")
    
    if not os.path.exists(model_path):
        print(f"Ошибка: Файл модели {model_type}.joblib не найден in src/models/")
        return

    try:
        model = joblib.load(model_path)
        
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print("Модель и скейлер успешно загружены.")
        else:
            scaler = None
            print("Модель успешно загружена (работаем без скейлера).")
            
    except Exception as e:
        print(f"Ошибка загрузки joblib файлов: {e}")
        return

    try:
        df = pd.read_csv(input_file)
        print(f"Загружен файл: {input_file}")
        
        processed_df = preprocess_data(df.copy())
        
        X = processed_df.copy()
        if 'Year' in X.columns:
            X = X.drop('Year', axis=1)
        
        target = "tax_receipts(billion USD)"
        if target in X.columns:
            X = X.drop(target, axis=1)
            
        if scaler is not None:
            X_processed = scaler.transform(X)
        else:
            X_processed = X
            
        predictions = model.predict(X_processed)
        
        result_df = df.copy()
        result_df['predicted_tax_receipts'] = predictions
        
        output_path = input_file.replace(".csv", f"_{model_type}_predicted.csv")
        result_df.to_csv(output_path, index=False)
        
        print(f"--- Предсказания успешно сохранены в: {output_path}")
        print("Первые строки результата:")
        print("Предсказанные поступления:")
        print(result_df['predicted_tax_receipts'].head())
        
    except Exception as e:
        print(f"Ошибка при выполнении предсказания: {e}")

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_path, "data")
    
    if os.path.exists(data_dir):
        search_pattern = os.path.join(data_dir, "*.csv")
        all_csv_files = glob.glob(search_pattern)
        
        input_files = [f for f in all_csv_files if not f.endswith("_predicted.csv")]
        
        if len(input_files) == 0:
            print(f"В папке {data_dir} {os.linesep}Не найдено новых исходных CSV-файлов для прогноза.")
        else:
            print(f"Найдено файлов для обработки: {len(input_files)}")
            
            if len(sys.argv) > 1:
                chosen_model = sys.argv[1]
            else:
                chosen_model = "lin_reg"
            
            allowed_models = ["lin_reg", "rf_reg", "lgbm", "xgb"]
            
            if chosen_model not in allowed_models:
                print(f"Ошибка: Модель '{chosen_model}' не поддерживается скриптом.")
                print(f"Допустимые варианты для ввода: {', '.join(allowed_models)}")
            else:
                for file_path in input_files:
                    run_prediction(file_path, model_type=chosen_model)
    else:
        print(f"Папка {data_dir} не найдена. Проверьте структуру проекта.")