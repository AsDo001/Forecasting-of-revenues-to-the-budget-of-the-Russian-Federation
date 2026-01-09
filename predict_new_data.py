import pandas as pd
import os
import glob
from src.data_preprocessing import preprocess_data
from src.feature_engineering import create_features
from src.inference import load_model_and_scaler, predict

def run_prediction(input_folder, output_folder, model_type="randomforestregressor"):
    """
    Загружает предобученную модель и делает предсказания для всех CSV файлов в папке.
    """
    # Пути к моделям (согласно структуре вашего репозитория)
    model_path = f"src/models/{model_type}_model.joblib"
    scaler_path = "src/models/scaler.joblib"
    
    print(f"--- Запуск процесса предсказания (Модель: {model_type}) ---")
    
    try:
        model, scaler = load_model_and_scaler(model_path, scaler_path)
        print("Модель и скейлер успешно загружены.")
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        return

    # Создаем папку для результатов, если её нет
    os.makedirs(output_folder, exist_ok=True)

    # Ищем все CSV файлы в папке
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    
    if not csv_files:
        print(f"В папке {input_folder} не найдено CSV файлов.")
        return

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        print(f"Обработка файла: {file_name}...")
        
        try:
            # 1. Загрузка
            df = pd.read_csv(file_path)
            
            # 2. Предобработка (используем ту же логику, что при обучении)
            processed_df = preprocess_data(df.copy())
            
            # 3. Выделение признаков
            # Если в новых данных нет целевой переменной, create_features может выдать ошибку.
            # Поэтому адаптируем:
            if 'Year' in processed_df.columns:
                X = processed_df.drop("Year", axis=1)
            else:
                X = processed_df.copy()
            
            # Удаляем целевую переменную, если она есть (для оценки)
            target = "tax_receipts(billion USD)"
            if target in X.columns:
                X = X.drop(target, axis=1)
            
            # 4. Предсказание
            predictions = predict(model, scaler, X)
            
            # 5. Сохранение результата
            result_df = df.copy()
            result_df['predicted_tax_receipts'] = predictions
            
            output_path = os.path.join(output_folder, f"predicted_{file_name}")
            result_df.to_csv(output_path, index=False)
            print(f"Результат сохранен в: {output_path}")
            
        except Exception as e:
            print(f"Ошибка при обработке {file_name}: {e}")

    print("--- Процесс предсказания завершен ---")

if __name__ == "__main__":
    # Пример: берем данные из папки 'data/new_data' и сохраняем в 'data/predictions'
    # Вы можете изменить эти пути под свои нужды
    run_prediction(
        input_folder="data", 
        output_folder="data/predictions",
        model_type="randomforestregressor" # или "xgbregressor", "linearregression"
    )