import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ======================== АДАПТИВНЫЙ CSS ДЛЯ МОБИЛЬНЫХ УСТРОЙСТВ ========================
st.markdown("""
<style>
    /* Ограничиваем максимальную ширину контента и добавляем отступы по бокам */
    .main > div {
        max-width: 100%;
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }
    /* Увеличиваем размер кликабельных элементов (чекбоксы, радио) */
    .stCheckbox, .stRadio, .stSlider {
        margin-bottom: 0.75rem;
    }
    label {
        font-size: 16px !important;
        font-weight: 500 !important;
    }
    /* Кнопки на всю ширину, крупный текст */
    .stButton button, .stForm button {
        width: 100%;
        font-size: 1.2rem !important;
        padding: 0.6rem !important;
        border-radius: 8px !important;
    }
    /* Уменьшаем отступы у контейнера, но оставляем достаточно сверху */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    /* Метрики: выравнивание по левому краю (вместо центра) */
    .stMetric {
        text-align: left !important;
    }
    /* Прогресс-бар – полная ширина */
    .stProgress > div {
        width: 100% !important;
    }
    /* Делаем expander более читаемым */
    .streamlit-expanderHeader {
        font-size: 1rem;
    }
    /* Заголовок: увеличиваем line-height и верхний отступ, чтобы эмодзи не обрезался */
    h1 {
        line-height: 1.3 !important;
        padding-top: 0.5rem;
    }
    
    /* ========== АДАПТИВНЫЕ ЗАГОЛОВКИ ДЛЯ ТЕЛЕФОНОВ ========== */
    @media (max-width: 768px) {
        h1 {
            font-size: 1.8rem !important;
            word-break: break-word;
            padding-top: 0.75rem; /* дополнительный отступ на мобильных */
        }
        h2, .stMarkdown h2 {
            font-size: 1.5rem !important;
        }
        h3, .stMarkdown h3 {
            font-size: 1.3rem !important;
        }
        /* Увеличим общий верхний отступ контейнера на телефонах */
        .block-container {
            padding-top: 1.5rem !important;
        }
        /* На очень маленьких экранах колонки с метриками можно дополнительно уплотнить */
        .stColumn {
            padding-left: 0 !important;
            padding-right: 0 !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ======================== НАСТРОЙКА СТРАНИЦЫ ========================
st.set_page_config(
    page_title="Оценка риска диабета",
    page_icon="🩺",
    layout="centered"
)

# ======================== ЗАГОЛОВОК ========================
st.title("🩺 Калькулятор риска диабета (ранняя диагностика)")
st.markdown("""
Этот калькулятор оценивает вероятность наличия диабета на основе **симптомов и факторов риска**.
Модель обучена на данных пациентов из Сильхетской больницы (Бангладеш) и помогает выявить диабет на ранней стадии.
*Результат носит справочный характер и не заменяет консультацию врача.*
""")

# ======================== ЗАГРУЗКА МОДЕЛИ ========================
@st.cache_resource
def load_model():
    try:
        model = joblib.load('diabetes_rf_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Файл модели 'diabetes_rf_model.pkl' не найден. Сначала выполните скрипт обучения.")
        st.stop()

model = load_model()

# ======================== ПРИЗНАКИ И ИХ ПЕРЕВОД ========================
expected_features = ['Age', 'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss',
                     'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring',
                     'Itching', 'Irritability', 'delayed healing', 'partial paresis',
                     'muscle stiffness', 'Alopecia', 'Obesity']

feature_names_ru = {
    'Age': 'Возраст',
    'Gender': 'Пол',
    'Polyuria': 'Учащенное мочеиспускание (полиурия)',
    'Polydipsia': 'Чрезмерная жажда (полидипсия)',
    'sudden weight loss': 'Резкая потеря веса',
    'weakness': 'Слабость',
    'Polyphagia': 'Повышенный аппетит (полифагия)',
    'Genital thrush': 'Генитальные инфекции (молочница)',
    'visual blurring': 'Затуманивание зрения',
    'Itching': 'Зуд',
    'Irritability': 'Раздражительность',
    'delayed healing': 'Медленное заживление ран',
    'partial paresis': 'Частичный парез',
    'muscle stiffness': 'Мышечная скованность',
    'Alopecia': 'Выпадение волос (алопеция)',
    'Obesity': 'Ожирение'
}

# ======================== ФОРМА ВВОДА ДАННЫХ (БЕЗ БОКОВОЙ ПАНЕЛИ) ========================
with st.form("risk_factors_form"):
    st.header("📋 Введите ваши данные")

    # Первая строка: возраст и пол (в две колонки)
    col_age, col_gender = st.columns(2)
    with col_age:
        age = st.slider(
            "Возраст (полных лет)",
            min_value=20, max_value=90, value=40,
            help="Укажите ваш возраст"
        )
    with col_gender:
        gender_input = st.radio(
            "Пол",
            options=["Мужской", "Женский"],
            help="Выберите ваш пол"
        )
    gender = 0 if gender_input == "Мужской" else 1

    st.subheader("Симптомы и факторы риска")
    st.caption("Отметьте признаки, которые у вас наблюдаются:")

    # Размещаем чекбоксы в две колонки для экономии места на мобильных
    symptom_values = {}
    symptoms_list = expected_features[2:]  # все признаки кроме Age и Gender
    col1, col2 = st.columns(2)
    for idx, feature in enumerate(symptoms_list):
        ru_name = feature_names_ru.get(feature, feature.replace('_', ' ').title())
        # Поочерёдно заполняем колонки
        if idx % 2 == 0:
            with col1:
                symptom_values[feature] = st.checkbox(ru_name, key=feature)
        else:
            with col2:
                symptom_values[feature] = st.checkbox(ru_name, key=feature)

    # Кнопка отправки формы (растянута на всю ширину)
    submitted = st.form_submit_button("Рассчитать риск", type="primary", use_container_width=True)

# ======================== ОБРАБОТКА РАСЧЁТА ПОСЛЕ ОТПРАВКИ ФОРМЫ ========================
if submitted:
    # Формируем вектор признаков в правильном порядке
    input_data = [age, gender]
    for feature in expected_features[2:]:
        input_data.append(1 if symptom_values[feature] else 0)

    # Преобразуем в DataFrame с именами признаков
    input_df = pd.DataFrame([input_data], columns=expected_features)

    # Предсказание
    prediction = model.predict(input_df)[0]          # 0 или 1
    probability = model.predict_proba(input_df)[0]   # [вероятность 0, вероятность 1]

    # Интерпретация
    risk_percent = probability[1] * 100
    if risk_percent < 30:
        level = "Низкий"
        color = "green"
        advice = "Вероятность диабета невысокая. Рекомендуется вести здоровый образ жизни и периодически проходить обследования."
    elif risk_percent < 60:
        level = "Средний"
        color = "orange"
        advice = ("Обнаружены некоторые факторы риска. Рекомендуется проконсультироваться с терапевтом или эндокринологом, "
                  "сдать анализ крови на сахар.")
    else:
        level = "Высокий"
        color = "red"
        advice = ("Вероятность диабета повышена. Настоятельно рекомендуется обратиться к врачу для углублённого обследования "
                  "(анализ на глюкозу, тест на толерантность к глюкозе).")

    # Вывод результатов
    st.subheader("Результат оценки 🔍")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Предсказанный класс", "Диабет" if prediction == 1 else "Нет диабета")
    with col2:
        st.metric("Вероятность диабета", f"{risk_percent:.1f}%")
    with col3:
        st.metric("Уровень риска", f":{color}[{level}]")

    st.progress(risk_percent / 100)

    st.info(advice)

    # Детализация введённых данных
    with st.expander("Введённые данные"):
        st.write(f"**Возраст:** {age}")
        st.write(f"**Пол:** {gender_input}")
        st.write("**Симптомы:**")
        active_symptoms = [name for name, val in symptom_values.items() if val]
        if active_symptoms:
            for eng_name in active_symptoms:
                ru_name = feature_names_ru.get(eng_name, eng_name.replace('_', ' ').title())
                st.write(f"- {ru_name}")
        else:
            st.write("Нет отмеченных симптомов.")
else:
    st.info("👆 Заполните форму выше и нажмите «Рассчитать риск».")

# ======================== ПОДВАЛ ========================
st.markdown("---")
st.caption("Данный калькулятор создан в образовательных целях. Модель обучена на датасете Early Stage Diabetes Risk Prediction (UCI).")
