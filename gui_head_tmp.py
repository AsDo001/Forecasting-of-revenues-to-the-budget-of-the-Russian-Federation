import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
import os
import json
from pathlib import Path
warnings.filterwarnings('ignore')

# ====================================================================================================
# КОНФИГУРАЦИЯ МОДЕЛЕЙ
# ====================================================================================================

MODEL_CONFIG = {
    'Linear Regression': {
        'file': 'lin_reg.joblib',
        'scaler_needed': True,
        'description': 'Базовая линейная модель, быстрая и интерпретируемая',
        'best_for': 'Базовые прогнозы, понимание линейных зависимостей',
        'color': '#1f77b4'
    },
    'Random Forest': {
        'file': 'rf_reg.joblib',
        'scaler_needed': False,
        'description': 'Ансамбль решающих деревьев',
        'best_for': 'Нелинейные зависимости, высокая точность',
        'color': '#9467bd'
    },
    'XGBoost': {
        'file': 'xgb.joblib',
        'scaler_needed': False,
        'description': 'Оптимизированный градиентный бустинг',
        'best_for': 'Максимальная производительность',
        'color': '#e377c2'
    },
    'LightGBM': {
        'file': 'lgbm.joblib',
        'scaler_needed': False,
        'description': 'Быстрый градиентный бустинг от Microsoft',
        'best_for': 'Большие датасеты, быстрое обучение',
        'color': '#7f7f7f'
    }
}

# ====================================================================================================
# ФУНКЦИИ ЗАГРУЗКИ МОДЕЛЕЙ
# ====================================================================================================

@st.cache_resource
def load_models_system():
    """Загружает все доступные модели, скалеры и метрики"""
    models = {}
    scalers = {}
    metrics = {}
    available_models = []

    models_path = Path('src/models')

    for model_name, config in MODEL_CONFIG.items():
        model_file = models_path / config['file']

        if model_file.exists():
            try:
                models[model_name] = joblib.load(model_file)
                available_models.append(model_name)

                # Загрузка скалера для линейных моделей
                if config['scaler_needed']:
                    scaler_file = models_path / config['file'].replace('.joblib', '_scaler.joblib')
                    if scaler_file.exists():
                        scalers[model_name] = joblib.load(scaler_file)
                    else:
                        # Пробуем общий скалер
                        general_scaler = models_path / 'scaler.joblib'
                        if general_scaler.exists():
                            scalers[model_name] = joblib.load(general_scaler)

                # Загрузка метрик
                metrics_file = models_path / config['file'].replace('.joblib', '_metrics.json')
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        metrics[model_name] = json.load(f)

            except Exception as e:
                st.sidebar.error(f"Ошибка загрузки {model_name}: {str(e)}")

    return models, scalers, metrics, available_models


def predict_with_model(model, model_name, features, scalers):
    """Делает предсказание с учетом скалера"""
    config = MODEL_CONFIG[model_name]

    if config['scaler_needed'] and model_name in scalers:
        features_scaled = scalers[model_name].transform(features)
        return model.predict(features_scaled)[0]
    else:
        return model.predict(features)[0]


def get_model_rating(metrics_dict, model_name):
    """Вычисляет рейтинг модели на основе метрик"""
    if model_name not in metrics_dict:
        return 0

    m = metrics_dict[model_name]
    r2 = m.get('r2', 0)
    mae = m.get('mae', 100)
    rmse = m.get('rmse', 100)

    rating = (r2 * 100) - (mae * 0.5) - (rmse * 0.3)
    return max(0, min(100, rating))

# ====================================================================================================
# UI КОМПОНЕНТЫ
# ====================================================================================================

def render_model_selector(models, metrics, available_models):
    """Селектор моделей в sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Выбор модели")

    if not available_models:
        st.sidebar.error("❌ Модели не найдены!")
        return None

    # Группировка
    linear_models = [m for m in available_models if 'Regression' in m]
    tree_models = [m for m in available_models if m not in linear_models]

    model_type = st.sidebar.radio(
        "Тип модели:",
        ["🎯 Все модели", "Линейные", "Деревья"],
        help="Фильтр по типу модели"
    )

    if model_type == "Линейные":
        filtered_models = linear_models
    elif model_type == "Деревья":
        filtered_models = tree_models
    else:
        filtered_models = available_models

    if not filtered_models:
        st.sidebar.warning("Нет доступных моделей этого типа")
        return None

    # Селектор с метриками
    model_options = []
    for model_name in filtered_models:
        if model_name in metrics and 'r2' in metrics[model_name]:
            r2 = metrics[model_name]['r2']
            model_options.append(f"{model_name} (R²: {r2:.3f})")
        else:
            model_options.append(model_name)

    selected_option = st.sidebar.selectbox(
        "Выберите модель:",
        model_options,
        help="Выберите модель для прогнозирования"
    )

    selected_model = selected_option.split(' (')[0]
    return selected_model


def render_model_info(model_name, metrics):
    """Информация о модели"""
    config = MODEL_CONFIG[model_name]

    with st.sidebar.expander("ℹ️ О модели", expanded=True):
        st.markdown(f"**{model_name}**")
        st.caption(config['description'])
        st.markdown("**Оптимальна для:**")
        st.caption(config['best_for'])

        if model_name in metrics and 'r2' in metrics[model_name]:
            m = metrics[model_name]
            col1, col2 = st.columns(2)
            with col1:
                st.metric("R²", f"{m.get('r2', 0):.3f}")
                st.metric("MAE", f"{m.get('mae', 0):.2f}")
            with col2:
                st.metric("RMSE", f"{m.get('rmse', 0):.2f}")
                if 'mape' in m:
                    st.metric("MAPE", f"{m.get('mape', 0):.1f}%")

            rating = get_model_rating(metrics, model_name)
            st.progress(rating / 100)
            st.caption(f"Рейтинг: {rating:.0f}/100")


def render_comparison_mode(models, scalers, metrics, features):
    """Режим сравнения всех моделей"""
    st.markdown("### 🔬 Режим сравнения моделей")

    predictions = {}
    for model_name, model in models.items():
        try:
            pred = predict_with_model(model, model_name, features, scalers)
            predictions[model_name] = pred
        except Exception as e:
            st.warning(f"⚠️ {model_name}: ошибка")
            predictions[model_name] = None

    valid_predictions = {k: v for k, v in predictions.items() if v is not None}

    if not valid_predictions:
        st.error("Ни одна модель не смогла сделать прогноз")
        return None

    # Статистика
    pred_values = list(valid_predictions.values())
    consensus = np.mean(pred_values)
    std_dev = np.std(pred_values)
    min_pred = min(pred_values)
    max_pred = max(pred_values)

    # Консенсус
    st.markdown("#### 🎯 Консенсус-прогноз")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Среднее", f"{consensus:.2f} млрд")
    with col2:
        st.metric("Разброс", f"±{std_dev:.2f} млрд")
    with col3:
        st.metric("Минимум", f"{min_pred:.2f} млрд")
    with col4:
        st.metric("Максимум", f"{max_pred:.2f} млрд")

    # Таблица
    st.markdown("#### 📊 Сравнение моделей")
    comparison_data = []
    for model_name, pred in valid_predictions.items():
        r2 = metrics[model_name].get('r2') if model_name in metrics else None
        mae = metrics[model_name].get('mae') if model_name in metrics else None

        comparison_data.append({
            'Модель': model_name,
            'Прогноз (млрд USD)': f"{pred:.2f}",
            'Откл. от средн.': f"{pred - consensus:+.2f}",
            'R²': f"{r2:.3f}" if r2 else "—",
            'MAE': f"{mae:.2f}" if mae else "—"
        })

    df_comparison = pd.DataFrame(comparison_data)
    df_comparison = df_comparison.sort_values('Прогноз (млрд USD)', ascending=False)
    st.dataframe(df_comparison, use_container_width=True, hide_index=True)

    # График
    st.markdown("#### 📈 Визуализация прогнозов")
    fig = go.Figure()

    for model_name, pred in valid_predictions.items():
        color = MODEL_CONFIG[model_name]['color']
        fig.add_trace(go.Bar(
            x=[model_name],
            y=[pred],
            name=model_name,
            marker_color=color,
            text=[f"{pred:.2f}"],
            textposition='auto',
        ))

    fig.add_hline(
        y=consensus,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Консенсус: {consensus:.2f}",
        annotation_position="right"
    )

    fig.add_hrect(
        y0=consensus - std_dev,
        y1=consensus + std_dev,
        fillcolor="gray",
        opacity=0.2,
        line_width=0
    )

    fig.update_layout(
        title="Прогнозы моделей",
        yaxis_title="Млрд USD",
        xaxis_title="Модель",
        showlegend=False,
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # Анализ
    st.markdown("#### 🔍 Анализ согласованности")
    agreement_pct = (1 - std_dev / consensus) * 100 if consensus > 0 else 0

    if std_dev < 10:
        status_text = "🟢 **Высокая согласованность** - модели единодушны"
    elif std_dev < 30:
        status_text = "🟡 **Умеренная согласованность**"
    else:
        status_text = "🔴 **Низкая согласованность** - большой разброс"

    st.info(f"{status_text}\n\nСогласованность: {agreement_pct:.1f}%")

    # Рекомендация
    best_model = max(valid_predictions.items(), 
                    key=lambda x: metrics.get(x[0], {}).get('r2', 0))

    st.success(f"""
**💡 Рекомендация:**

- **Консенсус-прогноз:** {consensus:.2f} млрд USD (рекомендуется)
- **Лучшая модель:** {best_model[0]} → {best_model[1]:.2f} млрд USD
- **Диапазон:** {min_pred:.2f} - {max_pred:.2f} млрд USD
    """)

    return consensus

# ====================================================================================================
# КОНФИГУРАЦИЯ ПРИЛОЖЕНИЯ
# ====================================================================================================

st.set_page_config(
    page_title="Прогноз налоговых поступлений РФ",
    page_icon="🏛️",
    layout="wide"
)

st.title("🏛️ Прогнозирование налоговых поступлений РФ")
st.markdown("---")

# ====================================================================================================
# ЗАГРУЗКА МОДЕЛЕЙ
# ====================================================================================================

models, scalers, metrics, available_models = load_models_system()

# ====================================================================================================
# НАВИГАЦИЯ
# ====================================================================================================

page = st.sidebar.radio(
    "Навигация",
    ["🎯 Прогнозирование", "📊 О проекте", "📈 Исторические данные"]
)

# ====================================================================================================
# СТРАНИЦА: ПРОГНОЗИРОВАНИЕ
# ====================================================================================================

if page == "🎯 Прогнозирование":
    st.header("Прогнозирование налоговых поступлений")

    # Статус загрузки
    if available_models:
        st.sidebar.success(f"✅ Загружено моделей: {len(available_models)}")
    else:
        st.sidebar.error("❌ Модели не найдены!")

    # Выбор режима
    mode = st.sidebar.radio(
        "Режим работы:",
        ["🎯 Одна модель", "🔬 Сравнение всех"],
        help="Выберите режим прогнозирования"
    )

    # Селектор модели
    selected_model = None
    if mode == "🎯 Одна модель" and available_models:
        selected_model = render_model_selector(models, metrics, available_models)
        if selected_model:
            render_model_info(selected_model, metrics)

    # ВВОД ПАРАМЕТРОВ
    col1, col2, col3 = st.columns(3)
#Проверь нижние и верхние планки слайдеров и инпутов, и значения по умолчанию
    with col1:
        st.subheader("🛢️ Энергетика")
        oil_price = st.slider("Цена нефти ($/barrel)", 10.0, 200.0, 75.0, 0.5)
        gas_price = st.slider("Цена газа ($/MMBtu)", 1.0, 100.0, 6.5, 0.1)
        oil_production = st.number_input("Добыча нефти (млн б/г)", 1.0, 10.0, 3.65, 0.01)
        gas_production = st.number_input("Добыча газа (млрд м³/г)", 500.0, 800.0, 700.0, 1.0)
        oil_export = st.number_input("Экспорт нефти (млн тонн)", 100.0, 200.0, 140.0, 1.0)
        gas_export = st.number_input("Экспорт газа (млрд м³)", 150.0, 300.0, 188.0, 1.0)
        share_oil_gas = st.slider("Доля нефтегаз. доходов (%)", 15.0, 60.0, 28.0, 1.0)

    with col2:
        st.subheader("💰 Макроэкономика")
        gnp = st.number_input("ВНП (млрд USD)", 1000, 3500, 2250, 10)
        non_oil_gdp = st.slider("Ненефтяной ВВП (%)", 40.0, 95.0, 85.0, 0.5)
        tb = st.number_input("Торговый баланс (млрд USD)", -50.0, 100.0, 12.0, 1.0)
        fdi = st.number_input("Инвестиции FDI (млрд USD)", -30000.0, 50000.0, -3000.0, 100.0)
        import_volume = st.number_input("Импорт (млрд USD)", 150.0, 400.0, 290.0, 5.0)

        st.subheader("💱 Финансы")
        key_rate = st.slider("Ключевая ставка (%)", 5.0, 25.0, 18.0, 0.25)
        inflation = st.slider("Инфляция (%)", 2.0, 20.0, 8.5, 0.1)
        exchange_rate = st.number_input("Курс RUB/USD", 25.0, 150.0, 95.0, 0.5)
        moex_index = st.number_input("Индекс MOEX", 1000.0, 4000.0, 2950.0, 10.0)
        cpi = st.number_input("CPI", 1.0, 35.0, 21.5, 0.1)

    with col3:
        st.subheader("🏛️ Государство")
        public_debt = st.slider("Госдолг (% ВВП)", 5.0, 50.0, 22.0, 0.5)
        military_exp = st.slider("Воен. расходы (% ВВП)", 2.0, 10.0, 7.5, 0.1)
        vat_rate = st.slider("НДС (%)", 15.0, 25.0, 20.6, 0.1)
        pit_min = st.slider("НДФЛ мин (%)", 12.0, 15.0, 13.0, 1.0)
        pit_max = st.slider("НДФЛ макс (%)", 13.0, 20.0, 15.0, 1.0)

        st.subheader("👥 Социум")
        population = st.number_input("Население", 140000000, 150000000, 143500000, 100000)
        unemployment = st.slider("Безработица (%)", 2.0, 10.0, 2.8, 0.1)
        per_capita_income = st.number_input("Доход на душу (тыс. USD)", 20.0, 60.0, 37.0, 0.5)
        gini = st.slider("Коэф. Джини (%)", 30.0, 50.0, 41.5, 0.1)
        migration = st.number_input("Миграция (тыс. чел)", -500.0, 1000.0, -150.0, 10.0)

        st.subheader("🌍 Геополитика")
        isi = st.slider("Индекс санкций (0-10)", 0.0, 10.0, 9.5, 0.5)

    st.markdown("---")

    # КНОПКА ПРОГНОЗА
    if st.button("🚀 СДЕЛАТЬ ПРОГНОЗ", type="primary", use_container_width=True):
        features = np.array([[
            oil_price, gas_price, oil_production, gas_production,
            oil_export, gas_export, share_oil_gas, tb, fdi, import_volume,
            key_rate, public_debt, moex_index, inflation, exchange_rate,
            gnp, isi, migration, gini, population, unemployment,
            per_capita_income, non_oil_gdp, cpi, military_exp,
            vat_rate, 0, pit_min, pit_max
        ]])

        if mode == "🔬 Сравнение всех":
            # Режим сравнения
            consensus = render_comparison_mode(models, scalers, metrics, features)

        else:
            # Режим одной модели
            if selected_model and selected_model in models:
                try:
                    prediction = predict_with_model(
                        models[selected_model], 
                        selected_model, 
                        features, 
                        scalers
                    )

                    st.success(f"### 💰 Прогноз: **{prediction:.2f} млрд USD**")
                    st.caption(f"Модель: {selected_model}")

                    # Метрики
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Прогноз 2026", f"{prediction:.2f} млрд")
                    with col2:
                        avg = 122.89
                        diff = prediction - avg
                        st.metric("vs Среднее", f"{diff:+.2f} млрд", f"{diff/avg*100:+.1f}%")
                    with col3:
                        if selected_model in metrics and 'r2' in metrics[selected_model]:
                            r2 = metrics[selected_model]['r2']
                            st.metric("R² модели", f"{r2:.3f}")
                        else:
                            st.metric("R² модели", "—")
                    with col4:
                        if prediction > 350:
                            status = "🟢 Высокий"
                        elif prediction > 250:
                            status = "🟡 Средний"
                        else:
                            status = "🔴 Низкий"
                        st.metric("Уровень", status)

                    # График
                    st.markdown("### 📊 Сравнение с историей")
                    historical = pd.DataFrame({
                        'Год': list(range(1991, 2026)),
                        'Налоги': [17, 18, 15, 14, 13, 12, 11, 8, 10, 15,
                                  18, 22, 28, 35, 45, 60, 80, 110, 70, 120,
                                  160, 180, 200, 210, 160, 170, 200, 230, 250, 210,
                                  340, 400, 300, 280, 290]
                    })

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=historical['Год'],
                        y=historical['Налоги'],
                        mode='lines+markers',
                        name='История',
                        line=dict(color='blue', width=2),
                        marker=dict(size=5)
                    ))
                    fig.add_trace(go.Scatter(
                        x=[2026],
                        y=[prediction],
                        mode='markers',
                        name='Прогноз 2026',
                        marker=dict(color='red', size=20, symbol='star')
                    ))
                    fig.add_hline(
                        y=122.89,
                        line_dash="dash",
                        line_color="green",
                        annotation_text="Среднее: 122.89"
                    )
                    fig.update_layout(
                        title="Налоговые поступления (1991-2026)",
                        xaxis_title="Год",
                        yaxis_title="Млрд USD",
                        height=500,
                        xaxis=dict(range=[1990, 2027], dtick=5)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Ошибка: {e}")
            else:
                st.warning("Выберите модель")

# ====================================================================================================
# СТРАНИЦА: О ПРОЕКТЕ
# ====================================================================================================

elif page == "📊 О проекте":
    st.header("О проекте") # Проект по прогнозированию налоговых поступлений РФ

    st.markdown("""
    ### 🎯 Цель проекта
    Разработка модели машинного обучения для прогнозирования налоговых поступлений 
    в бюджет Российской Федерации на основе макроэкономических показателей.

    ### 📈 Датасет
    - **Источник**: [Russian Economy: 90s Chaos → 2020s Oil (Kaggle)](https://www.kaggle.com/datasets/arsseniidonskov/russian-economy-90s-chaos-2020s-oil)
    - **Период**: 1991 - 2025
    - **Объем**: 35 наблюдений × 30 признаков
    - **Источники**: Росстат, ЦБ РФ, World Bank, MOEX

    ### 🤖 Модели
    В проекте используются 4 модели машинного обучения:
    - Linear Regression
    - Random Forest
    - XGBoost
    - LightGBM

    ### 📊 Категории признаков
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **🛢️ Энергетика (7)**
        - Цены на нефть и газ
        - Объемы добычи и экспорта
        - Доля нефтегаз. доходов

        **💰 Макроэкономика (5)**
        - ВНП, ВВП
        - Торговый баланс
        - Инвестиции и импорт

        **💱 Финансы (5)**
        - Ключевая ставка
        - Инфляция, CPI
        - Курс валюты, MOEX
        """)

    with col2:
        st.markdown("""
        **🏛️ Фискальная политика (5)**
        - Госдолг
        - Военные расходы
        - Налоговые ставки

        **👥 Социум (5)**
        - Население
        - Безработица
        - Доходы, неравенство

        **🌍 Геополитика (1)**
        - Индекс санкций
        """)

    st.markdown("---")
    st.info("💡 **Совет**: Используйте режим сравнения моделей для более надежного прогноза")

# ====================================================================================================
# СТРАНИЦА: ИСТОРИЧЕСКИЕ ДАННЫЕ
# ====================================================================================================

elif page == "📈 Исторические данные":
    st.header("Исторические данные (1991-2025)")

    historical_data = pd.DataFrame({
        'Год': list(range(1991, 2026)),
        'Налоги (млрд USD)': [17, 18, 15, 14, 13, 12, 11, 8, 10, 15,
                              18, 22, 28, 35, 45, 60, 80, 110, 70, 120,
                              160, 180, 200, 210, 160, 170, 200, 230, 250, 210,
                              340, 400, 300, 280, 290],
        'Нефть ($/b)': [26.05, 16.55, 16.3, 12.35, 16.4, 19.47, 20.5, 12.0, 18.5, 26.2,
                        22.8, 23.7, 27.2, 26.8, 50.6, 61.09, 69.29, 94.4, 61.06, 78.2,
                        109.35, 110.52, 107.88, 97.6, 51.23, 41.9, 53.03, 70.01, 63.59, 41.73,
                        69.0, 76.09, 62.99, None, None],
        'Газ ($/MMBtu)': [1.76, 1.4, 1.8, 1.6, 1.5, 1.9, 2.1, 1.8, 2.2, 3.6,
                         3.3, 2.9, 4.8, 5.2, 7.5, 6.42, 6.8, 8.5, 4.0, 5.0,
                         6.0, 5.2, 5.5, 9.84, 4.0, 3.0, 3.5, 4.0, 3.0, 2.5,
                         4.0, 18.0, 9.0, 7.0, None],
        'ВНП (млрд USD)': [475, 465, 435, 395, 395, 390, 405, 275, 200, 260,
                          310, 345, 430, 590, 765, 990, 1300, 1660, 1220, 1525,
                          1900, 2015, 2095, 2030, 1365, 1280, 1580, 1660, 1690, 1480,
                          1780, 2200, 2100, 2200, 2300],
        'Курс RUB/USD': [1.8, 0.31, 0.97, 2.2, 4.56, 5.13, 5.78, 9.69, 24.62, 28.12,
                         29.18, 31.38, 30.69, 28.8, 28.24, 27.18, 25.57, 24.87, 31.77, 30.37,
                         29.4, 31.05, 31.86, 38.59, 61.26, 67.05, 58.33, 62.81, 64.71, 72.32,
                         73.71, 69.92, 85.54, 92.88, 86.1]
    })

    st.subheader("📋 Полная таблица данных")
    st.dataframe(historical_data, use_container_width=True, height=400)

    st.subheader("📊 Статистика по периодам")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("1991-2000", "13.3 млрд", "Кризис 90-х")
    with col2:
        st.metric("2001-2010", "58.8 млрд", "Восстановление")
    with col3:
        st.metric("2011-2020", "197.0 млрд", "Стабилизация")
    with col4:
        st.metric("2021-2025", "322.0 млрд", "Современность")

    st.subheader("📈 Динамика налоговых поступлений")
    fig = px.line(
        historical_data,
        x='Год',
        y='Налоги (млрд USD)',
        title='Налоговые поступления (1991-2025)',
        markers=True
    )

    fig.add_vrect(x0=1991, x1=2000, fillcolor="red", opacity=0.1,
                  annotation_text="Кризис", annotation_position="top left")
    fig.add_vrect(x0=2001, x1=2010, fillcolor="orange", opacity=0.1,
                  annotation_text="Восстановление", annotation_position="top left")
    fig.add_vrect(x0=2011, x1=2020, fillcolor="yellow", opacity=0.1,
                  annotation_text="Стабилизация", annotation_position="top left")
    fig.add_vrect(x0=2021, x1=2025, fillcolor="green", opacity=0.1,
                  annotation_text="Современность", annotation_position="top left")

    fig.add_hline(y=122.89, line_dash="dash", line_color="blue",
                  annotation_text="Среднее: 122.89")

    fig.update_layout(yaxis_title="Млрд USD", height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🛢️ Зависимость от цены нефти")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=historical_data['Год'],
        y=historical_data['Налоги (млрд USD)'],
        name='Налоги',
        yaxis='y1',
        line=dict(color='blue', width=2)
    ))
    fig2.add_trace(go.Scatter(
        x=historical_data['Год'],
        y=historical_data['Нефть ($/b)'],
        name='Нефть',
        yaxis='y2',
        line=dict(color='red', width=2, dash='dot')
    ))
    fig2.update_layout(
        title='Налоги vs Нефть',
        yaxis=dict(title='Налоги (млрд USD)', side='left'),
        yaxis2=dict(title='Нефть ($/barrel)', overlaying='y', side='right'),
        height=500
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🔗 Ключевые зависимости")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **💚 Положительная корреляция:**
        - ВНП: **+0.91**
        - Курс: **+0.87**
        - Доход на душу: **+0.84**
        - Нефть: **+0.66**
        - Газ: **+0.61**
        """)
    with col2:
        st.markdown("""
        **💔 Отрицательная корреляция:**
        - Безработица: **-0.74**
        - Инфляция: **-0.24**

        📊 Высокая безработица снижает
        налоговые поступления
        """)

    st.markdown("---")
    st.info("""
    **💡 Интересные факты:**
    - **Минимум (1998):** 8 млрд USD - дефолт
    - **Максимум (2022):** 400 млрд USD - рекорд
    - **Рост:** ×17 раз за период!
    """)

# ====================================================================================================
# ПОДВАЛ
# ====================================================================================================

st.markdown("---")
st.markdown("**Разработчик**: AsDo001 | **09**: Января 2026 | **GitHub**: [https://github.com/AsDo001/Forecasting-of-revenues-to-the-budget-of-the-Russian-Federation]")