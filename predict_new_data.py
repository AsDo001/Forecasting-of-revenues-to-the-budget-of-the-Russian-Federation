import pandas as pd
import os
import glob
import joblib
from src.data_preprocessing import preprocess_data
from src.feature_engineering import create_features
from src.inference import predict

def run_prediction(input_file, model_type="lin_reg"):
    """
    Загружает предобученную модель и делает предсказания для конкретного файла.
    """
    # Пути к моделям внутри папки src/models
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, "src", "models", f"{model_type}.joblib")
    scaler_path = os.path.join(base_path, "src", "models", "lin_reg_scaler.joblib")
    
    print(f"--- Запуск предсказания (Модель: {model_type}) ---")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Ошибка: Файлы модели или скейлера не найдены в src/models/")
        return

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print("Модель и скейлер успешно загружены.")
    except Exception as e:
        print(f"Ошибка загрузки joblib файлов: {e}")
        return

    try:
        # 1. Загрузка новых данных
        df = pd.read_csv(input_file)
        print(f"Загружен файл: {input_file}")
        
        # 2. Предобработка
        processed_df = preprocess_data(df.copy())
        
        # 3. Подготовка признаков (X)
        # Убираем Year и целевую переменную, если они есть
        X = processed_df.copy()
        if 'Year' in X.columns:
            X = X.drop('Year', axis=1)
        
        target = "tax_receipts(billion USD)"
        if target in X.columns:
            X = X.drop(target, axis=1)
            
        # 4. Предсказание
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        
        # 5. Вывод результата
        result_df = df.copy()
        result_df['predicted_tax_receipts'] = predictions
        
        output_path = input_file.replace(".csv", "_predicted.csv")
        result_df.to_csv(output_path, index=False)
        print(f"✅ Предсказания успешно сохранены в: {output_path}")
        print("\nПервые 5 строк результата:")
        print(result_df[['Year', 'predicted_tax_receipts']].head() if 'Year' in result_df.columns else result_df['predicted_tax_receipts'].head())
        
    except Exception as e:
        print(f"Ошибка при выполнении предсказания: {e}")

if __name__ == "__main__":
    # По умолчанию предсказываем для dataset.csv в папке data
    data_to_predict = "data/test.csv"
    if os.path.exists(data_to_predict):
        run_prediction(data_to_predict)
    else:
        print(f"Файл {data_to_predict} не найден. Проверьте путь.")