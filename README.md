# 💰 Предсказание налоговых поступлений в бюджет РФ

> Прогнозирование налоговых поступлений в бюджет Российской Федерации с использованием машинного обучения на основе макроэкономических индикаторов (1991-2025)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF.svg)](https://www.kaggle.com/datasets/arsseniidonskov/russian-economy-90s-chaos-2020s-oil)

---

## 📋 Содержание
- [О проекте](#-о-проекте)
- [Проблема и решение](#-проблема-и-решение)
- [Датасет и признаки](#-датасет-и-признаки)
- [Структура проекта](#-структура-проекта)
- [Технологии](#-технологии)
- [Установка](#-установка)
- [Использование](#-использование)
- [Результаты](#-результаты)
- [Roadmap](#-roadmap)
- [Вклад в проект](#-вклад-в-проект)
- [Лицензия](#-лицензия)
- [Контакты](#-контакты)

---

## 🎯 О проекте

Этот проект использует **регрессионные модели машинного обучения** для прогнозирования налоговых поступлений в бюджет Российской Федерации на основе комплексного анализа экономических, социальных и геополитических факторов за период с 1991 по 2025 год.

### Ключевые особенности:
- 📊 **35 лет исторических данных** (1991-2025)
- 🔢 **27 признаков** для прогнозирования
- 🎯 **Целевая переменная**: налоговые поступления в бюджет (млрд USD)
- 🌍 **Уникальный датасет** по российской экономике
- 🤖 **Ensemble методы** для повышения точности

---

## 🤔 Проблема и решение

### Проблема
Бюджет Российской Федерации исторически сильно зависит от:
- 🛢️ **Экспорта углеводородов** (нефть и газ)
- 💱 **Волатильности мировых цен** на энергоносители
- 🌐 **Геополитической обстановки** и санкционного давления
- 📉 **Макроэкономической нестабильности**

Прогнозирование налоговых поступлений в таких условиях — сложная задача, требующая учета множества взаимосвязанных факторов.

### Решение
**Machine Learning подход** для анализа исторических паттернов и построения прогнозных моделей:
- Регрессионные алгоритмы (Linear Regression, Random Forest, Gradient Boosting, XGBoost)
- Feature Engineering для улучшения предсказательной способности
- Обработка временных рядов и учет исторических трендов
- Валидация на out-of-sample данных

**Практическое применение:**
- Планирование бюджета
- Оценка рисков и сценарный анализ
- Принятие управленческих решений
- Академические исследования

---

## 📊 Датасет и признаки

### Источник данных
- **Датасет**: [Russian Economy: 90s Chaos → 2020s Oil](https://www.kaggle.com/datasets/arsseniidonskov/russian-economy-90s-chaos-2020s-oil)
- **Период**: 1991 - 2025
- **Объем**: 35 наблюдений
- **Источники**: Росстат, ЦБ РФ, World Bank, MOEX

### 🎯 Целевая переменная

| Переменная | Описание | Единицы |
|-----------|----------|---------|
| **tax_receipts** ⭐ | Налоговые поступления в бюджет РФ | млрд USD |

### 📈 Признаки (Features)

#### 🛢️ Энергетический сектор (ключевые драйверы бюджета)
| Признак | Описание | Единицы |
|---------|----------|---------|
| `oil_prices` | Цены на нефть | USD/barrel |
| `gas_prices` | Цены на газ | USD/MMBtu |
| `Oil_production_volume` | Объём добычи нефти | млн баррелей/год |
| `Gas_production_volume` | Объём добычи газа | млрд куб.м/год |
| `Oil_export_volume` | Объём экспорта нефти | млн тонн |
| `Gas_export_volume` | Объём экспорта газа | млрд куб.м |
| `Share_of_oil_and_gas_revenues` | Доля нефтегазовых доходов в бюджете | % |

#### 💰 Макроэкономические индикаторы
| Признак | Описание | Единицы |
|---------|----------|---------|
| `GNP` | Валовой национальный продукт | млрд USD |
| `Non_oil_GDP` | Доля ненефтяного ВВП (диверсификация) | % |
| `TB` | Торговый баланс | млрд USD |
| `FDI` | Прямые иностранные инвестиции | млрд USD |
| `Import_volume` | Объём импорта | млрд USD |

#### 💱 Монетарная политика и финансовые рынки
| Признак | Описание | Единицы |
|---------|----------|---------|
| `Key_rate` | Ключевая ставка ЦБ РФ | % |
| `inflation_rate` | Уровень инфляции | % |
| `CPI` | Индекс потребительских цен | индекс |
| `exchange_rates` | Курс обмена рубля к доллару | RUB/USD |
| `Stock_Market_Index` | Индекс Московской биржи | индекс MOEX |

#### 🏛️ Фискальная политика и долг
| Признак | Описание | Единицы |
|---------|----------|---------|
| `level_of_public_debt` | Уровень государственного долга | % от ВВП |
| `Military_expenditures` | Военные расходы | % от ВВП |
| `tax_rates (VAT)` | Ставка налога на добавленную стоимость | % |
| `tax_rates (PIT)` | Ставка налога на доходы физических лиц | % |

#### 👥 Социально-демографические факторы
| Признак | Описание | Единицы |
|---------|----------|---------|
| `population_size` | Численность населения | человек |
| `Migration_rate` | Чистая миграция | тыс. человек |
| `unemployment_rate` | Уровень безработицы | % |
| `per_c_i` | Доход на душу населения | тыс. USD |
| `Gini_coefficient` | Коэффициент Джини (неравенство) | % |

#### 🌍 Геополитические факторы
| Признак | Описание | Единицы |
|---------|----------|---------|
| `ISI` | Индекс санкционного давления | шкала 0-10 |

---

## 📁 Структура проекта

```
Forecasting-of-revenues-to-the-budget-of-the-Russian-Federation/
│
├── data/
│   ├── raw/                          # Исходные данные
│   ├── processed/                    # Обработанные данные
│   └── README.md                     # Описание данных
│
├── notebooks/
│   ├── 01_EDA.ipynb                 # Exploratory Data Analysis
│   ├── 02_Feature_Engineering.ipynb  # Создание и отбор признаков
│   ├── 03_Modeling.ipynb            # Обучение моделей
│   └── 04_Evaluation.ipynb          # Оценка и сравнение моделей
│
├── src/
│   ├── data_preprocessing.py        # Препроцессинг данных
│   ├── feature_engineering.py       # Feature engineering
│   ├── models.py                    # Модели ML
│   ├── evaluation.py                # Метрики и оценка
│   └── utils.py                     # Вспомогательные функции
│
├── models/                          # Сохранённые модели
│   ├── best_model.pkl
│   └── model_performance.json
│
├── results/
│   ├── figures/                     # Графики и визуализации
│   └── reports/                     # Отчёты и метрики
│
├── requirements.txt                 # Python зависимости
├── README.md                        # Этот файл
└── LICENSE                          # Лицензия MIT
```

---

## 🛠 Технологии

### Core Stack
- **Python** 3.8+
- **Pandas** 1.5+ — обработка данных
- **NumPy** 1.23+ — вычисления
- **Scikit-learn** 1.2+ — ML алгоритмы
- **XGBoost** / **LightGBM** — gradient boosting

### Визуализация
- **Matplotlib** 3.6+
- **Seaborn** 0.12+
- **Plotly** 5.0+ (интерактивные графики)

### Дополнительно
- **Jupyter Notebook** — exploratory analysis
- **SHAP** — интерпретация моделей
- **optuna** — hyperparameter tuning

---

## 🚀 Установка

### 1️⃣ Клонирование репозитория
```bash
git clone https://github.com/AsDo001/Forecasting-of-revenues-to-the-budget-of-the-Russian-Federation.git
cd Forecasting-of-revenues-to-the-budget-of-the-Russian-Federation
```

### 2️⃣ Создание виртуального окружения (рекомендуется)
```bash
# Создание виртуального окружения
python -m venv venv

# Активация (Windows)
venv\Scripts\activate

# Активация (Linux/Mac)
source venv/bin/activate
```

### 3️⃣ Установка зависимостей
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4️⃣ Скачивание данных
Скачайте датасет с Kaggle:
- [Russian Economy Dataset](https://www.kaggle.com/datasets/arsseniidonskov/russian-economy-90s-chaos-2020s-oil)
- Поместите файл в папку `data/raw/`

**Или используйте Kaggle API:**
```bash
kaggle datasets download -d arsseniidonskov/russian-economy-90s-chaos-2020s-oil
unzip russian-economy-90s-chaos-2020s-oil.zip -d data/raw/
```

---

## 💻 Использование

### Быстрый старт
```python
# Импортируем необходимые модули
from src.data_preprocessing import load_data, preprocess_data
from src.models import train_model, predict
from src.evaluation import evaluate_model

# Загружаем и обрабатываем данные
data = load_data('data/raw/russian_economy.csv')
X_train, X_test, y_train, y_test = preprocess_data(data)

# Обучаем модель
model = train_model(X_train, y_train, model_type='xgboost')

# Делаем предсказания
predictions = predict(model, X_test)

# Оценка качества модели
metrics = evaluate_model(y_test, predictions)
print(f"R² Score: {metrics['r2']:.4f}")
print(f"RMSE: {metrics['rmse']:.2f} млрд USD")
```

### Запуск Jupyter ноутбуков
```bash
jupyter notebook notebooks/01_EDA.ipynb
```

### Обучение модели из командной строки
```bash
python src/train.py --model xgboost --cv 5 --optimize
```

---

## 📈 Результаты

### Лучшая модель: XGBoost Regressor

**Метрики на тестовой выборке:**
- **R² Score**: 0.XX
- **RMSE**: XX.XX млрд USD
- **MAE**: XX.XX млрд USD
- **MAPE**: XX.X%

### Важность признаков (Feature Importance)
ТОП-5 наиболее важных признаков:
1. `oil_prices` — XX%
2. `exchange_rates` — XX%
3. `Share_of_oil_and_gas_revenues` — XX%
4. `GNP` — XX%
5. `inflation_rate` — XX%

### Ключевые инсайты
- 📊 Цены на нефть остаются главным драйвером бюджетных поступлений
- 💱 Курс рубля сильно влияет на налоговые сборы
- 🌍 Санкционное давление (ISI) имеет значимый негативный эффект
- 📈 Диверсификация экономики (Non_oil_GDP) снижает зависимость от нефти

---

## 🗺️ Roadmap

### ✅ Реализовано
- [x] Сбор и очистка данных
- [x] Exploratory Data Analysis (EDA)
- [x] Baseline модели (Linear Regression, Decision Tree)
- [x] Advanced модели (Random Forest, XGBoost)

### 🚧 В работе
- [ ] Feature Engineering (лаги, rolling statistics)
- [ ] Hyperparameter Tuning с Optuna
- [ ] Ансамблирование моделей (Stacking)
- [ ] Интерпретация с SHAP values

### 🔮 Планы
- [ ] Временные ряды (ARIMA, SARIMA, Prophet)
- [ ] Deep Learning (LSTM, Transformer)
- [ ] Web-приложение для прогнозирования (Streamlit/FastAPI)
- [ ] Автоматизация парсинга данных (Росстат API)
- [ ] Dockerизация проекта
- [ ] CI/CD pipeline

---

## 🤝 Вклад в проект

Буду рад вашим предложениям и улучшениям! 🚀

### Как внести вклад:
1. **Fork** репозиторий
2. Создайте **feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit** изменения (`git commit -m 'Add some AmazingFeature'`)
4. **Push** в branch (`git push origin feature/AmazingFeature`)
5. Откройте **Pull Request**

### Идеи для развития:
- 💡 Новые признаки (добавление данных о санкциях, нефтяных квотах OPEC+)
- 🔧 Улучшение моделей (новые алгоритмы, tuning гиперпараметров)
- 📊 Визуализация (интерактивные дашборды)
- 🧪 Unit-тесты для кода
- 📚 Улучшение документации
- 😄 Мем-генератор для неудачных прогнозов

---

## 📄 Лицензия

Этот проект распространяется под лицензией **MIT License**.  
Смотрите файл [LICENSE](LICENSE) для подробностей.

---

## 📞 Контакты

**Автор**: Arsenii Donskov

- 🐱 **GitHub**: [@AsDo001](https://github.com/AsDo001)
- 📊 **Kaggle**: [kaggle.com/yourprofile](https://www.kaggle.com/arsseniidonskov)

---

## 🙏 Благодарности

- **Росстат** — за публичные данные по российской экономике
- **ЦБ РФ** — за статистику по монетарной политике
- **World Bank** — за международные индикаторы
- **Kaggle Community** — за площадку для шаринга датасетов

---

<div align="center">

### ⭐ Если проект был полезен — поставьте звезду! ⭐

**Создано с ❤️ и данными**

*P.S. Если бюджет РФ рухнет — не вините модель, вините волатильность нефтяных рынков и картель OPEC+* 😄

</div>
