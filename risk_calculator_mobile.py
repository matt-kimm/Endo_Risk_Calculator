import math
import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

# ======================== НАСТРОЙКА СТРАНИЦЫ ========================
st.set_page_config(
    page_title="Эндокринная медицинская карта",
    page_icon="🩺",
    layout="centered",
)

# ======================== АДАПТИВНЫЙ CSS ДЛЯ МОБИЛЬНЫХ УСТРОЙСТВ ========================
st.markdown(
    """
<style>
    .main > div {
        max-width: 100%;
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }
    .stCheckbox, .stRadio, .stSlider, .stNumberInput, .stSelectbox {
        margin-bottom: 0.75rem;
    }
    label {
        font-size: 16px !important;
        font-weight: 500 !important;
    }
    .stButton button, .stForm button {
        width: 100%;
        font-size: 1.1rem !important;
        padding: 0.6rem !important;
        border-radius: 10px !important;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .stMetric {
        text-align: left !important;
    }
    .streamlit-expanderHeader {
        font-size: 1rem;
    }
    h1 {
        line-height: 1.3 !important;
        padding-top: 0.5rem;
    }
    .card {
        border: 1px solid rgba(120,120,120,0.18);
        border-radius: 16px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.8rem;
        background: rgba(250,250,250,0.65);
    }
    .muted {
        color: #666;
        font-size: 0.92rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.85rem;
        margin-left: 0.35rem;
    }
    @media (max-width: 768px) {
        h1 {
            font-size: 1.8rem !important;
            word-break: break-word;
            padding-top: 0.75rem;
        }
        h2, .stMarkdown h2 {
            font-size: 1.45rem !important;
        }
        h3, .stMarkdown h3 {
            font-size: 1.22rem !important;
        }
        .block-container {
            padding-top: 1.5rem !important;
        }
        .stColumn {
            padding-left: 0 !important;
            padding-right: 0 !important;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)

# ======================== ПЕРЕВОДЫ И СПРАВОЧНИКИ ========================
feature_names_ru = {
    "Age": "Возраст",
    "Gender": "Пол",
    "Polyuria": "Учащенное мочеиспускание (полиурия)",
    "Polydipsia": "Чрезмерная жажда (полидипсия)",
    "sudden weight loss": "Резкая потеря веса",
    "weakness": "Слабость",
    "Polyphagia": "Повышенный аппетит (полифагия)",
    "Genital thrush": "Генитальные инфекции (молочница)",
    "visual blurring": "Затуманивание зрения",
    "Itching": "Зуд",
    "Irritability": "Раздражительность",
    "delayed healing": "Медленное заживление ран",
    "partial paresis": "Частичный парез",
    "muscle stiffness": "Мышечная скованность",
    "Alopecia": "Выпадение волос (алопеция)",
    "Obesity": "Ожирение",
    "cold intolerance": "Непереносимость холода",
    "heat intolerance": "Непереносимость жары",
    "constipation": "Запоры",
    "diarrhea": "Диарея",
    "palpitations": "Сердцебиение",
    "tremor": "Тремор",
    "dry skin": "Сухость кожи",
    "fatigue": "Утомляемость",
    "anxiety": "Тревожность",
    "neck swelling": "Увеличение / отек в области шеи",
    "irregular periods": "Нерегулярный менструальный цикл",
    "acne": "Акне",
    "hirsutism": "Избыточный рост волос по мужскому типу",
    "infertility": "Бесплодие / трудности с зачатием",
    "postmenopausal": "Постменопауза",
    "prior fracture": "Перенесенный перелом",
    "glucocorticoids": "Длительный прием глюкокортикоидов",
    "low activity": "Низкая физическая активность",
}

def badge(level: str) -> str:
    colors = {
        "Низкий": "#1f8b4c",
        "Умеренный": "#c77700",
        "Высокий": "#c62828",
        "Не оценен": "#666666",
    }
    color = colors.get(level, "#666666")
    return f"<span class='badge' style='background:{color}; color:white;'>{level}</span>"

def clamp(x, lo=0.0, hi=100.0):
    return float(max(lo, min(hi, x)))

def yes(val: bool) -> int:
    return 1 if val else 0

def risk_level(score: float, low: float = 30.0, high: float = 60.0):
    if score < low:
        return "Низкий"
    if score < high:
        return "Умеренный"
    return "Высокий"

def advice_by_level(level: str, low_msg: str, mid_msg: str, high_msg: str) -> str:
    if level == "Низкий":
        return low_msg
    if level == "Умеренный":
        return mid_msg
    return high_msg

def score_to_text(score: float) -> str:
    return f"{clamp(score):.1f}%"

def summarize_flags(flags):
    if not flags:
        return "Явно выраженных групп риска по анкете не выделено."
    return " / ".join(flags)

# ======================== МОДЕЛЬ ДЛЯ ДИАБЕТА ========================
@st.cache_resource
def load_model():
    try:
        return joblib.load("diabetes_rf_model.pkl")
    except Exception:
        return None

model = load_model()


@st.cache_resource
def load_optional_model(path):
    try:
        return joblib.load(path)
    except Exception:
        return None

METABOLIC_MODEL_PATH = "metabolic_ml_model.pkl"
THYROID_MODEL_PATH = "thyroid_ml_model.pkl"
PCOS_MODEL_PATH = "pcos_ml_model.pkl"
NETWORK_MODEL_PATH = "endo_network_ml_model.pkl"

metabolic_model = load_optional_model(METABOLIC_MODEL_PATH)
thyroid_model = load_optional_model(THYROID_MODEL_PATH)
pcos_model = load_optional_model(PCOS_MODEL_PATH)
network_model = load_optional_model(NETWORK_MODEL_PATH)


def safe_positive_probability(model, row_df):
    """Возвращает вероятность положительного класса в процентах или None."""
    if model is None or not hasattr(model, "predict_proba"):
        return None
    try:
        proba = model.predict_proba(row_df)[0]
        if len(proba) == 1:
            return float(proba[0]) * 100.0

        classes = list(getattr(model, "classes_", []))
        if 1 in classes:
            pos_idx = classes.index(1)
        elif "1" in classes:
            pos_idx = classes.index("1")
        else:
            pos_idx = 1 if len(proba) > 1 else 0
        return float(proba[pos_idx]) * 100.0
    except Exception:
        return None


def activity_to_code(activity_level):
    return {"Низкая": 0, "Средняя": 1, "Высокая": 2}.get(activity_level, 1)


def make_metabolic_features(age, gender, bmi, waist_cm, activity_level, sleep_hours, fasting_glucose, hba1c, diabetes_symptom_values):
    symptom_burden = int(sum(1 for v in diabetes_symptom_values.values() if v))
    return pd.DataFrame([{
        "age": age,
        "gender": gender,
        "bmi": float(round(bmi, 2)),
        "waist_cm": float(round(waist_cm, 2)),
        "sleep_hours": float(round(sleep_hours, 2)),
        "activity_code": activity_to_code(activity_level),
        "fasting_glucose": float(fasting_glucose) if fasting_glucose and fasting_glucose > 0 else np.nan,
        "hba1c": float(hba1c) if hba1c and hba1c > 0 else np.nan,
        "symptom_burden": symptom_burden,
        "obesity_flag": int(bmi >= 30),
        "central_obesity_flag": int(waist_cm >= (88 if gender == 1 else 94)),
        "sleep_short_flag": int(sleep_hours < 7),
    }])


def make_thyroid_features(age, gender, thyroid_values, tsh_value, ft4_value):
    return pd.DataFrame([{
        "age": age,
        "gender": gender,
        "tsh": float(tsh_value) if tsh_value and tsh_value > 0 else np.nan,
        "ft4": float(ft4_value) if ft4_value and ft4_value > 0 else np.nan,
        "cold_intolerance": yes(thyroid_values.get("cold intolerance")),
        "heat_intolerance": yes(thyroid_values.get("heat intolerance")),
        "constipation": yes(thyroid_values.get("constipation")),
        "diarrhea": yes(thyroid_values.get("diarrhea")),
        "palpitations": yes(thyroid_values.get("palpitations")),
        "tremor": yes(thyroid_values.get("tremor")),
        "dry_skin": yes(thyroid_values.get("dry skin")),
        "fatigue": yes(thyroid_values.get("fatigue")),
        "anxiety": yes(thyroid_values.get("anxiety")),
        "neck_swelling": yes(thyroid_values.get("neck swelling")),
        "alopecia": yes(thyroid_values.get("Alopecia")),
        "weakness": yes(thyroid_values.get("weakness")),
    }])


def make_pcos_features(age, gender, bmi, waist_cm, activity_level, sleep_hours, pcos_values, fasting_glucose, hba1c, insulin_resistance_score, tsh_value, ft4_value):
    return pd.DataFrame([{
        "age": age,
        "gender": gender,
        "bmi": float(round(bmi, 2)),
        "waist_cm": float(round(waist_cm, 2)),
        "sleep_hours": float(round(sleep_hours, 2)),
        "activity_code": activity_to_code(activity_level),
        "ir_score": float(insulin_resistance_score),
        "fasting_glucose": float(fasting_glucose) if fasting_glucose and fasting_glucose > 0 else np.nan,
        "hba1c": float(hba1c) if hba1c and hba1c > 0 else np.nan,
        "tsh": float(tsh_value) if tsh_value and tsh_value > 0 else np.nan,
        "ft4": float(ft4_value) if ft4_value and ft4_value > 0 else np.nan,
        "irregular_periods": yes(pcos_values.get("irregular periods")),
        "acne": yes(pcos_values.get("acne")),
        "hirsutism": yes(pcos_values.get("hirsutism")),
        "infertility": yes(pcos_values.get("infertility")),
        "obesity": yes(pcos_values.get("Obesity")),
        "alopecia": yes(pcos_values.get("Alopecia")),
        "polyphagia": yes(pcos_values.get("Polyphagia")),
    }])


def make_network_features(diabetes_score, ir_score, hypo_score, hyper_score, pcos_score, bone_score, obesity_score, metabolic_score, age, gender, bmi):
    return pd.DataFrame([{
        "diabetes_score": float(diabetes_score),
        "ir_score": float(ir_score),
        "hypothyroid_score": float(hypo_score),
        "hyperthyroid_score": float(hyper_score),
        "pcos_score": np.nan if pcos_score is None else float(pcos_score),
        "bone_score": float(bone_score),
        "obesity_score": float(obesity_score),
        "metabolic_score": float(metabolic_score),
        "age": age,
        "gender": gender,
        "bmi": float(round(bmi, 2)),
        "cross_axis_burden": float(np.nanmean([
            diabetes_score,
            ir_score,
            hypo_score,
            hyper_score,
            0.0 if pcos_score is None else pcos_score,
            bone_score,
        ])),
    }])


def ml_or_fallback_score(model, row_df, fallback_score):
    ml_score = safe_positive_probability(model, row_df)
    return ml_score if ml_score is not None else fallback_score

# ======================== БАЗОВЫЕ ПРИЗНАКИ ДЛЯ ИСХОДНОГО ДИАБЕТИЧЕСКОГО МОДУЛЯ ========================
expected_features = [
    "Age",
    "Gender",
    "Polyuria",
    "Polydipsia",
    "sudden weight loss",
    "weakness",
    "Polyphagia",
    "Genital thrush",
    "visual blurring",
    "Itching",
    "Irritability",
    "delayed healing",
    "partial paresis",
    "muscle stiffness",
    "Alopecia",
    "Obesity",
]

# ======================== МУЛЬТИФАКТОРНЫЕ БЛОКИ ========================
thyroid_symptoms = [
    "cold intolerance",
    "heat intolerance",
    "constipation",
    "diarrhea",
    "palpitations",
    "tremor",
    "dry skin",
    "fatigue",
    "anxiety",
    "neck swelling",
    "Alopecia",
    "weakness",
]

pcos_symptoms = [
    "irregular periods",
    "acne",
    "hirsutism",
    "infertility",
    "Obesity",
    "Alopecia",
    "Polyphagia",
]

bone_risk_features = [
    "postmenopausal",
    "prior fracture",
    "glucocorticoids",
    "low activity",
    "dry skin",
    "fatigue",
]

# ======================== ФУНКЦИИ РИСКА ========================
def diabetes_probability_from_model(age, gender, symptom_values, family_history_diabetes):
    input_data = [age, gender]
    for feature in expected_features[2:]:
        input_data.append(1 if symptom_values.get(feature, False) else 0)

    input_df = pd.DataFrame([input_data], columns=expected_features)

    if model is not None and hasattr(model, "predict_proba"):
        try:
            probability = safe_positive_probability(model, input_df)
            prediction = int(model.predict(input_df)[0])
            if probability is not None:
                # добавляем наследственность
                if family_history_diabetes:
                    probability = clamp(probability + 10.0)
                return probability, prediction, None
        except Exception as e:
            return None, None, f"Не удалось использовать модель: {e}"

    # Безопасный эвристический fallback
    score = 0.0
    score += 0.8 * (age - 30) if age >= 30 else 0.0
    score += 12 * yes(symptom_values.get("Polyuria"))
    score += 12 * yes(symptom_values.get("Polydipsia"))
    score += 10 * yes(symptom_values.get("Polyphagia"))
    score += 10 * yes(symptom_values.get("sudden weight loss"))
    score += 8 * yes(symptom_values.get("weakness"))
    score += 8 * yes(symptom_values.get("visual blurring"))
    score += 8 * yes(symptom_values.get("delayed healing"))
    score += 7 * yes(symptom_values.get("Genital thrush"))
    score += 7 * yes(symptom_values.get("Obesity"))
    score += 4 * yes(symptom_values.get("Irritability"))
    score += 4 * yes(symptom_values.get("Itching"))
    score += 5 * yes(symptom_values.get("Alopecia"))
    if family_history_diabetes:
        score += 10
    score = clamp(score, 0, 99)
    prediction = 1 if score >= 50 else 0
    return score, prediction, None

def obesity_proxy(bmi, waist_cm, activity_level, sleep_hours):
    score = 0.0
    if bmi >= 35:
        score += 35
    elif bmi >= 30:
        score += 28
    elif bmi >= 27:
        score += 20
    elif bmi >= 25:
        score += 12

    if waist_cm:
        if waist_cm >= 102:
            score += 20
        elif waist_cm >= 94:
            score += 14
        elif waist_cm >= 88:
            score += 10

    activity_map = {"Высокая": 0, "Средняя": 6, "Низкая": 12}
    score += activity_map.get(activity_level, 0)

    if sleep_hours < 6:
        score += 8
    elif sleep_hours < 7:
        score += 4

    return clamp(score)

def insulin_resistance_proxy(age, bmi, waist_cm, activity_level, sleep_hours, diabetes_symptom_values, family_history_diabetes):
    score = obesity_proxy(bmi, waist_cm, activity_level, sleep_hours)

    if age >= 45:
        score += 8
    elif age >= 35:
        score += 5

    score += 10 * yes(diabetes_symptom_values.get("Polyuria"))
    score += 10 * yes(diabetes_symptom_values.get("Polydipsia"))
    score += 8 * yes(diabetes_symptom_values.get("Polyphagia"))
    score += 6 * yes(diabetes_symptom_values.get("Obesity"))
    score += 6 * yes(diabetes_symptom_values.get("sudden weight loss"))
    score += 6 * yes(diabetes_symptom_values.get("weakness"))

    if family_history_diabetes:
        score += 8

    return clamp(score)

def hypothyroid_proxy(age, thyroid_values, tsh_value, ft4_value, family_history_thyroid):
    score = 0.0
    score += 12 * yes(thyroid_values.get("cold intolerance"))
    score += 10 * yes(thyroid_values.get("constipation"))
    score += 10 * yes(thyroid_values.get("fatigue"))
    score += 7 * yes(thyroid_values.get("dry skin"))
    score += 7 * yes(thyroid_values.get("Alopecia"))
    score += 6 * yes(thyroid_values.get("weakness"))
    score += 5 * yes(thyroid_values.get("neck swelling"))

    if age >= 50:
        score += 5

    if tsh_value and tsh_value > 0:
        if tsh_value > 4.5:
            score += min(25, (tsh_value - 4.5) * 8)
        elif tsh_value < 0.4:
            score -= 8

    if ft4_value and ft4_value > 0:
        if ft4_value < 0.8:
            score += 10

    if family_history_thyroid:
        score += 8

    return clamp(score)

def hyperthyroid_proxy(age, thyroid_values, tsh_value, ft4_value, family_history_thyroid):
    score = 0.0
    score += 12 * yes(thyroid_values.get("heat intolerance"))
    score += 10 * yes(thyroid_values.get("palpitations"))
    score += 9 * yes(thyroid_values.get("tremor"))
    score += 8 * yes(thyroid_values.get("anxiety"))
    score += 7 * yes(thyroid_values.get("diarrhea"))
    score += 7 * yes(thyroid_values.get("sudden weight loss"))
    score += 5 * yes(thyroid_values.get("neck swelling"))
    score += 4 * yes(thyroid_values.get("weakness"))

    if age < 50:
        score += 3

    if tsh_value and tsh_value > 0:
        if tsh_value < 0.4:
            score += min(25, (0.4 - tsh_value) * 20)
        elif tsh_value > 4.5:
            score -= 8

    if ft4_value and ft4_value > 0:
        if ft4_value > 1.8:
            score += 10

    if family_history_thyroid:
        score += 8

    return clamp(score)

def pcos_proxy(age, sex, pcos_values, bmi, insulin_resistance_score, fasting_glucose, hba1c):
    if sex != 1:
        return None

    score = 0.0
    score += 18 * yes(pcos_values.get("irregular periods"))
    score += 12 * yes(pcos_values.get("acne"))
    score += 14 * yes(pcos_values.get("hirsutism"))
    score += 10 * yes(pcos_values.get("infertility"))
    score += 8 * yes(pcos_values.get("Alopecia"))
    score += 8 * yes(pcos_values.get("Obesity"))
    score += min(18, insulin_resistance_score * 0.18)

    if age <= 35:
        score += 4
    if bmi >= 30:
        score += 6

    if fasting_glucose and fasting_glucose > 0 and fasting_glucose >= 100:
        score += 6
    if hba1c and hba1c > 0 and hba1c >= 5.7:
        score += 6

    return clamp(score)

def osteoporosis_proxy(age, sex, bone_values, bmi, family_history_osteoporosis):
    score = 0.0
    score += 15 * yes(bone_values.get("postmenopausal"))
    score += 14 * yes(bone_values.get("prior fracture"))
    score += 12 * yes(bone_values.get("glucocorticoids"))
    score += 10 * yes(bone_values.get("low activity"))

    if bmi and bmi > 0:
        if bmi < 18.5:
            score += 16
        elif bmi < 20:
            score += 10
        elif bmi < 22:
            score += 4

    if sex == 1:
        score += 4
    if age >= 65:
        score += 10
    elif age >= 50:
        score += 6

    if family_history_osteoporosis:
        score += 10

    return clamp(score)

def metabolic_syndrome_proxy(age, sex, bmi, waist_cm, activity_level, fasting_glucose, hba1c, insulin_resistance_score, family_history_diabetes):
    score = 0.0
    score += obesity_proxy(bmi, waist_cm, activity_level, sleep_hours=7)
    score += min(20, insulin_resistance_score * 0.18)

    if age >= 45:
        score += 8
    elif age >= 35:
        score += 4

    if fasting_glucose and fasting_glucose > 0:
        if fasting_glucose >= 100:
            score += 10
        if fasting_glucose >= 126:
            score += 18

    if hba1c and hba1c > 0:
        if hba1c >= 5.7:
            score += 8
        if hba1c >= 6.5:
            score += 16

    if bmi >= 30:
        score += 6

    if family_history_diabetes:
        score += 8

    return clamp(score)

def generate_connections(diabetes_score, ir_score, hypo_score, hyper_score, pcos_score, bone_score, sex):
    items = []
    if diabetes_score >= 60 and ir_score >= 60:
        items.append("Вероятен общий метаболический драйвер: инсулинорезистентность.")
    if pcos_score is not None and pcos_score >= 50:
        items.append("Женский гормональный блок: PCOS часто связан с инсулинорезистентностью и набором веса.")
    if hypo_score >= 50:
        items.append("Щитовидная гипофункция может усиливать утомляемость, набор веса и ухудшать липидный профиль.")
    if hyper_score >= 50:
        items.append("Щитовидная гиперфункция способна усиливать сердцебиение, потерю веса и риск потери костной массы.")
    if bone_score >= 50 and hyper_score >= 50:
        items.append("Комбинация повышенного тиреоидного риска и костного риска требует внимания к костной ткани.")
    if diabetes_score >= 60 and bone_score >= 40:
        items.append("При нарушении углеводного обмена стоит помнить о более высоком риске осложнений со стороны костей и сосудов.")
    if sex == 0 and pcos_score is not None:
        items.append("PCOS не оценивается: блок включается только для женщин.")
    return items

def generate_next_steps(diabetes_score, ir_score, hypo_score, hyper_score, pcos_score, bone_score, bmi, fasting_glucose, hba1c):
    steps = []
    if diabetes_score >= 60 or (hba1c and hba1c >= 6.5) or (fasting_glucose and fasting_glucose >= 126):
        steps.append("Эндокринолог в ближайшее время + анализ HbA1c, глюкоза натощак, при необходимости ОГТТ.")
    elif diabetes_score >= 30:
        steps.append("Контроль HbA1c и глюкозы натощак, коррекция питания, веса и активности.")
    else:
        steps.append("Профилактический контроль углеводного обмена 1 раз в 6–12 месяцев.")

    if ir_score >= 50 or bmi >= 30:
        steps.append("Оценить окружность талии, режим сна, физическую активность и пищевые привычки.")
    if hypo_score >= 50 or hyper_score >= 50:
        steps.append("Сдать ТТГ и свободный Т4; при симптомах — очная консультация эндокринолога.")
    if pcos_score is not None and pcos_score >= 50:
        steps.append("Для женщин: обсудить PCOS, регулярность цикла, андрогенные симптомы и метаболический скрининг.")
    if bone_score >= 50:
        steps.append("Оценить витамин D, кальций, DEXA/денситометрию по показаниям и факторы падения костной массы.")

    return steps

# ======================== ЗАГОЛОВОК ========================
st.title("🩺 Эндокринная медицинская карта")
st.markdown(
    """
Этот прототип объединяет несколько часто встречаемых эндокринных рисков в одном экране: диабет, инсулинорезистентность/метаболический синдром, нарушения щитовидной железы, PCOS и риск снижения костной массы.
Ниже выводится не просто процент, а связанная карта слабых мест и возможных пересечений между ними.

*Результат носит справочный характер и не заменяет очную консультацию врача.*
"""
)

ml_ready_note = []
if model is not None:
    ml_ready_note.append("диабет")
if metabolic_model is not None:
    ml_ready_note.append("метаболический риск")
if thyroid_model is not None:
    ml_ready_note.append("щитовидная железа")
if pcos_model is not None:
    ml_ready_note.append("PCOS")
if network_model is not None:
    ml_ready_note.append("эндокринная сеть")

if ml_ready_note:
    st.success("ML-модели загружены для: " + ", ".join(ml_ready_note) + ".")
else:
    st.info("Для новых блоков используется безопасная клиническая логика; ML-модели можно подключить файлами .pkl без изменения интерфейса.")

# ======================== ФОРМА ВВОДА (БЕЗ st.form) ========================
st.header("📋 Введите данные")

col_age, col_gender = st.columns(2)
with col_age:
    age = st.slider("Возраст (полных лет)", min_value=18, max_value=90, value=40, help="Укажите ваш возраст")
with col_gender:
    gender_input = st.radio("Пол", options=["Мужской", "Женский"], help="Выберите пол")
gender = 0 if gender_input == "Мужской" else 1

st.subheader("🧬 Наследственность")
family_history_diabetes = st.checkbox("Наследственность по диабету 2 типа (родители, сиблинги)")
family_history_thyroid = st.checkbox("Наследственность по заболеваниям щитовидной железы")
family_history_osteoporosis = st.checkbox("Наследственность по остеопорозу")

st.subheader("🧩 Базовые данные")
col_h, col_w, col_waist = st.columns(3)
with col_h:
    height_cm = st.number_input("Рост, см", min_value=100.0, max_value=230.0, value=170.0, step=1.0)
with col_w:
    weight_kg = st.number_input("Вес, кг", min_value=30.0, max_value=250.0, value=75.0, step=0.5)
with col_waist:
    waist_cm = st.number_input("Талия, см", min_value=40.0, max_value=200.0, value=85.0, step=1.0)

col_sleep, col_activity = st.columns(2)
with col_sleep:
    sleep_hours = st.slider("Сон, часов/сутки", min_value=3.0, max_value=12.0, value=7.0, step=0.5)
with col_activity:
    activity_level = st.selectbox("Физическая активность", ["Низкая", "Средняя", "Высокая"], index=1)

bmi = weight_kg / ((height_cm / 100.0) ** 2) if height_cm > 0 else 0.0
st.caption(f"Расчетный ИМТ: {bmi:.1f}")

st.subheader("🍬 Симптомы, связанные с диабетом")
st.caption("Отметьте признаки, которые у вас наблюдаются.")
diabetes_symptom_values = {}
diabetes_features = expected_features[2:]
col1, col2 = st.columns(2)
for idx, feature in enumerate(diabetes_features):
    ru_name = feature_names_ru.get(feature, feature.replace("_", " ").title())
    with (col1 if idx % 2 == 0 else col2):
        diabetes_symptom_values[feature] = st.checkbox(ru_name, key=f"dm_{feature}")

st.subheader("🦋 Щитовидная железа")
thyroid_values = {}
th_col1, th_col2 = st.columns(2)
for idx, feature in enumerate(thyroid_symptoms):
    ru_name = feature_names_ru.get(feature, feature.replace("_", " ").title())
    with (th_col1 if idx % 2 == 0 else th_col2):
        thyroid_values[feature] = st.checkbox(ru_name, key=f"th_{feature}")

st.subheader("♀️ Женский гормональный блок (PCOS)")
pcos_values = {}
if gender == 1:
    st.caption("Этот блок активен только для женщин.")
    pcos_col1, pcos_col2 = st.columns(2)
    for idx, feature in enumerate(pcos_symptoms):
        ru_name = feature_names_ru.get(feature, feature.replace("_", " ").title())
        with (pcos_col1 if idx % 2 == 0 else pcos_col2):
            pcos_values[feature] = st.checkbox(ru_name, key=f"pcos_{feature}")
else:
    st.caption("PCOS-блок для мужчин не оценивается.")
    for feature in pcos_symptoms:
        pcos_values[feature] = False

st.subheader("🦴 Костный риск / остеопения")
bone_values = {}
bone_col1, bone_col2 = st.columns(2)
for idx, feature in enumerate(bone_risk_features):
    ru_name = feature_names_ru.get(feature, feature.replace("_", " ").title())
    with (bone_col1 if idx % 2 == 0 else bone_col2):
        bone_values[feature] = st.checkbox(ru_name, key=f"bone_{feature}")

st.subheader("🧪 Анализы (если уже есть)")
col_fg, col_hba1c, col_tsh, col_ft4 = st.columns(4)
with col_fg:
    fasting_glucose = st.number_input("Глюкоза натощак, мг/дл", min_value=0.0, max_value=1000.0, value=0.0, step=1.0, help="0 = не указывать")
with col_hba1c:
    hba1c = st.number_input("HbA1c, %", min_value=0.0, max_value=20.0, value=0.0, step=0.1, help="0 = не указывать")
with col_tsh:
    tsh_value = st.number_input("ТТГ, мМЕ/л", min_value=0.0, max_value=100.0, value=0.0, step=0.1, help="0 = не указывать")
with col_ft4:
    ft4_value = st.number_input("Св. T4, нг/дл", min_value=0.0, max_value=10.0, value=0.0, step=0.1, help="0 = не указывать")

st.subheader("📈 Мультифрактальный анализ гликемии (экспериментально)")
st.caption("Если есть ряд глюкозы по времени, можно вставить его сюда. Это исследовательский блок, а не стандартная клиническая методика.")
glucose_series_text = st.text_area(
    "Глюкозный ряд (числа через запятую, пробел или перенос строки)",
    height=110,
    placeholder="Например: 92, 95, 90, 101, 115, 108, 98, 94 ..."
)
glucose_file = st.file_uploader(
    "Или загрузите файл с рядом глюкозы (.txt, .csv)",
    type=["txt", "csv"],
    help="Подходит файл с одним числом в строке или с числами, разделёнными запятыми / пробелами / точками с запятой.",
)
enable_mfdfa = st.checkbox("Выполнить MF-DFA-анализ, если данных достаточно", value=False)

submitted = st.button("Собрать медицинскую карту", type="primary", use_container_width=True)

# ======================== MF-DFA ========================

def extract_numeric_series(text: str):
    if not text or not text.strip():
        return None
    tokens = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text.replace(",", " "))
    if not tokens:
        return None
    arr = np.asarray([float(tok) for tok in tokens], dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr if arr.size else None

def parse_series(text: str):
    arr = extract_numeric_series(text)
    if arr is None:
        return None
    return arr if arr.size >= 12 else None

def parse_uploaded_glucose_file(uploaded_file):
    if uploaded_file is None:
        return None
    raw = uploaded_file.getvalue()
    text = None
    for encoding in ("utf-8", "utf-8-sig", "cp1251", "latin1"):
        try:
            text = raw.decode(encoding)
            break
        except Exception:
            continue
    if text is None:
        return None

    arr = extract_numeric_series(text)
    if arr is not None and arr.size >= 12:
        return arr

    try:
        df = pd.read_csv(io.StringIO(text), header=None, engine="python")
        numeric = pd.to_numeric(df.stack(), errors="coerce").dropna().to_numpy(dtype=float)
        numeric = numeric[np.isfinite(numeric)]
        if numeric.size >= 12:
            return numeric
    except Exception:
        pass

    return None

def mfdfa(series, q_vals=None, min_scale=4, max_scale=None, scale_count=8):
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 12:
        return None

    x = x - np.mean(x)
    y = np.cumsum(x)

    if max_scale is None:
        max_scale = max(min_scale + 2, n // 3)
    max_scale = min(max_scale, max(min_scale + 2, n // 2))
    if max_scale <= min_scale:
        return None

    if max_scale - min_scale <= 10:
        scales = np.arange(min_scale, max_scale + 1, dtype=int)
    else:
        scales = np.unique(
            np.floor(np.logspace(np.log10(min_scale), np.log10(max_scale), scale_count)).astype(int)
        )
    scales = scales[scales >= 4]
    scales = np.unique(scales)
    if scales.size < 3:
        return None

    if q_vals is None:
        q_vals = np.array([-2, -1, 0, 1, 2], dtype=float) if n < 24 else np.array([-4, -2, -1, 0, 1, 2, 4], dtype=float)

    Fq = np.full((len(q_vals), len(scales)), np.nan, dtype=float)

    for si, s in enumerate(scales):
        nseg = n // s
        if nseg < 2:
            continue
        rms = []
        for v in range(2 * nseg):
            if v < nseg:
                start = v * s
            else:
                start = n - (v - nseg + 1) * s
            segment = y[start:start + s]
            if segment.size < s:
                continue
            t = np.arange(s, dtype=float)
            coef = np.polyfit(t, segment, 1)
            trend = np.polyval(coef, t)
            resid = segment - trend
            rms.append(np.mean(resid ** 2))

        rms = np.asarray(rms, dtype=float)
        rms = rms[rms > 0]
        if rms.size == 0:
            continue

        for qi, q in enumerate(q_vals):
            if abs(q) < 1e-12:
                Fq[qi, si] = np.exp(0.5 * np.mean(np.log(rms)))
            else:
                Fq[qi, si] = (np.mean(rms ** (q / 2.0))) ** (1.0 / q)

    Hq = []
    for qi in range(len(q_vals)):
        valid = np.isfinite(Fq[qi]) & (Fq[qi] > 0)
        if valid.sum() < 3:
            Hq.append(np.nan)
            continue
        slope, _ = np.polyfit(np.log(scales[valid]), np.log(Fq[qi, valid]), 1)
        Hq.append(slope)

    Hq = np.asarray(Hq, dtype=float)
    if np.all(~np.isfinite(Hq)):
        return None

    width = float(np.nanmax(Hq) - np.nanmin(Hq))

    tau = q_vals * Hq - 1.0
    alpha = np.full_like(tau, np.nan, dtype=float)
    f_alpha = np.full_like(tau, np.nan, dtype=float)
    valid_tau = np.isfinite(tau) & np.isfinite(q_vals)
    if np.sum(valid_tau) >= 2:
        alpha_valid = np.gradient(tau[valid_tau], q_vals[valid_tau])
        alpha[valid_tau] = alpha_valid
        f_alpha[valid_tau] = q_vals[valid_tau] * alpha_valid - tau[valid_tau]

    return {
        "scales": scales,
        "q_vals": q_vals,
        "Hq": Hq,
        "Fq": Fq,
        "tau": tau,
        "alpha": alpha,
        "f_alpha": f_alpha,
        "width": width,
        "mean_h": float(np.nanmean(Hq)),
    }

def mfdfa_interpretation(result):
    if result is None:
        return "Недостаточно данных для MF-DFA."
    width = result["width"]
    mean_h = result["mean_h"]
    if width < 0.12:
        level = "Низкая мультифрактальность"
        note = "Ряд относительно однородный и менее вариабельный."
    elif width < 0.25:
        level = "Умеренная мультифрактальность"
        note = "Есть заметная неоднородность колебаний."
    else:
        level = "Высокая мультифрактальность"
        note = "Колебания выраженно неоднородны; это может отражать нестабильную гликемическую динамику."
    return f"{level}. Ширина спектра: {width:.3f}. Средний H(q): {mean_h:.3f}. {note}"

def plot_mfdfa_scaling(result):
    if result is None:
        return None
    scales = np.asarray(result.get("scales", []), dtype=float)
    q_vals = np.asarray(result.get("q_vals", []), dtype=float)
    Fq = np.asarray(result.get("Fq", []), dtype=float)
    if scales.size == 0 or q_vals.size == 0 or Fq.size == 0:
        return None

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for qi, q in enumerate(q_vals):
        y = Fq[qi] if Fq.ndim == 2 and qi < Fq.shape[0] else None
        if y is None:
            continue
        valid = np.isfinite(y) & (y > 0)
        if valid.sum() < 3:
            continue
        x = np.log10(scales[valid])
        yy = np.log10(y[valid])
        ax.plot(x, yy, marker='o', linewidth=1.3, markersize=3.5, label=f"q={q:g}")
        if valid.sum() >= 2:
            coef = np.polyfit(x, yy, 1)
            xfit = np.linspace(x.min(), x.max(), 50)
            ax.plot(xfit, np.polyval(coef, xfit), linestyle='--', linewidth=1, alpha=0.6)

    ax.set_xlabel("log10(scale)")
    ax.set_ylabel("log10(Fq)")
    ax.set_title("MF-DFA scaling plot")
    ax.grid(True, alpha=0.25)
    if len(q_vals) <= 7:
        ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    return fig

def plot_mfdfa_spectrum(result):
    if result is None:
        return None
    alpha = np.asarray(result.get("alpha", []), dtype=float)
    f_alpha = np.asarray(result.get("f_alpha", []), dtype=float)
    valid = np.isfinite(alpha) & np.isfinite(f_alpha)
    if valid.sum() < 2:
        return None

    order = np.argsort(alpha[valid])
    alpha_sorted = alpha[valid][order]
    f_sorted = f_alpha[valid][order]

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(alpha_sorted, f_sorted, marker='o', linewidth=1.5, markersize=4)
    ax.set_xlabel("α")
    ax.set_ylabel("f(α)")
    ax.set_title("Multifractal spectrum")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig

def interpret_complexity(width):
    if width >= 0.8:
        return "Высокая метаболическая сложность / адаптивность"
    elif width >= 0.45:
        return "Умеренная метаболическая сложность"
    else:
        return "Сниженная сложность, возможна потеря адаптивности"

def compare_to_reference(current_width):
    reference_width = 0.75
    delta = current_width - reference_width
    if delta > 0.15:
        status = "Сложность выше условной нормы"
    elif delta < -0.15:
        status = "Сложность ниже условной нормы"
    else:
        status = "Близко к условной норме"
    return {
        "reference": reference_width,
        "delta": delta,
        "status": status
    }

# ======================== РЕЗУЛЬТАТЫ ========================
if submitted:
    diabetes_score, diabetes_prediction, diabetes_fallback_error = diabetes_probability_from_model(
        age, gender, diabetes_symptom_values, family_history_diabetes
    )
    if diabetes_score is None:
        diabetes_score = 0.0
        diabetes_prediction = 0

    ir_score = insulin_resistance_proxy(age, bmi, waist_cm, activity_level, sleep_hours, diabetes_symptom_values, family_history_diabetes)
    obesity_score = obesity_proxy(bmi, waist_cm, activity_level, sleep_hours)
    hypothyroid_rule_score = hypothyroid_proxy(age, thyroid_values, tsh_value, ft4_value, family_history_thyroid)
    hyperthyroid_rule_score = hyperthyroid_proxy(age, thyroid_values, tsh_value, ft4_value, family_history_thyroid)
    pcos_rule_score = pcos_proxy(age, gender, pcos_values, bmi, ir_score, fasting_glucose, hba1c)
    bone_score = osteoporosis_proxy(age, gender, bone_values, bmi, family_history_osteoporosis)
    metabolic_rule_score = metabolic_syndrome_proxy(age, gender, bmi, waist_cm, activity_level, fasting_glucose, hba1c, ir_score, family_history_diabetes)

    metabolic_ml_df = make_metabolic_features(age, gender, bmi, waist_cm, activity_level, sleep_hours, fasting_glucose, hba1c, diabetes_symptom_values)
    thyroid_ml_df = make_thyroid_features(age, gender, thyroid_values, tsh_value, ft4_value)
    pcos_ml_df = make_pcos_features(age, gender, bmi, waist_cm, activity_level, sleep_hours, pcos_values, fasting_glucose, hba1c, ir_score, tsh_value, ft4_value)

    metabolic_score = ml_or_fallback_score(metabolic_model, metabolic_ml_df, metabolic_rule_score)
    hypothyroid_score = ml_or_fallback_score(thyroid_model, thyroid_ml_df, hypothyroid_rule_score)
    hyperthyroid_score = ml_or_fallback_score(thyroid_model, thyroid_ml_df, hyperthyroid_rule_score)
    pcos_score = None if gender == 0 else ml_or_fallback_score(pcos_model, pcos_ml_df, pcos_rule_score)

    endo_network_df = make_network_features(
        diabetes_score, ir_score, hypothyroid_score, hyperthyroid_score,
        pcos_score, bone_score, obesity_score, metabolic_score, age, gender, bmi
    )
    network_score = safe_positive_probability(network_model, endo_network_df)
    if network_score is None:
        network_score = clamp(
            0.18 * diabetes_score
            + 0.18 * ir_score
            + 0.14 * hypothyroid_score
            + 0.14 * hyperthyroid_score
            + 0.14 * (0.0 if pcos_score is None else pcos_score)
            + 0.12 * bone_score
            + 0.10 * obesity_score
        )

    diabetes_level = risk_level(diabetes_score)
    ir_level = risk_level(ir_score)
    obesity_level = risk_level(obesity_score)
    hypo_level = risk_level(hypothyroid_score)
    hyper_level = risk_level(hyperthyroid_score)
    pcos_level = risk_level(pcos_score) if pcos_score is not None else "Не оценен"
    bone_level = risk_level(bone_score)
    metabolic_level = risk_level(metabolic_score)
    network_level = risk_level(network_score)

    diabetes_advice = advice_by_level(
        diabetes_level,
        "Риск диабета по текущим данным выглядит невысоким. Поддерживайте активность и базовый скрининг 1 раз в год.",
        "Есть признаки, которые стоит перепроверить лабораторно: глюкоза натощак, HbA1c, окружность талии, вес.",
        "Риск диабета высокий. Нужна очная оценка и лабораторное подтверждение в ближайшее время.",
    )

    ir_advice = advice_by_level(
        ir_level,
        "Явных признаков выраженной инсулинорезистентности немного.",
        "Есть смысл усилить сон, активность и снизить висцеральный жир; стоит проверить HbA1c и липиды.",
        "Картина хорошо укладывается в инсулинорезистентность / метаболический синдром.",
    )

    hypo_advice = advice_by_level(
        hypo_level,
        "Убедительных признаков гипофункции щитовидной железы немного.",
        "Стоит проверить ТТГ и свободный Т4, особенно если есть утомляемость или набор веса.",
        "Нужна очная оценка щитовидной железы и лабораторное подтверждение.",
    )

    hyper_advice = advice_by_level(
        hyper_level,
        "Выраженных признаков тиреотоксикоза немного.",
        "При сердцебиении, дрожи и потере веса стоит проверить ТТГ и свободный Т4.",
        "Есть признаки, требующие проверки гиперфункции щитовидной железы.",
    )

    pcos_advice = "PCOS не оценивается у мужчин." if pcos_score is None else advice_by_level(
        pcos_level,
        "Выраженных признаков PCOS немного.",
        "Есть признаки, совместимые с PCOS; полезна оценка цикла, андрогенных симптомов и метаболического статуса.",
        "Картина может соответствовать PCOS; рекомендована очная консультация гинеколога-эндокринолога.",
    )

    network_advice = advice_by_level(
        network_level,
        "Эндокринная сеть сейчас выглядит относительно спокойной.",
        "Есть несколько взаимосвязанных зон, за которыми стоит наблюдать в динамике.",
        "Выраженная нагрузка на эндокринную сеть: стоит смотреть не только отдельные диагнозы, но и их сочетания.",
    )

    bone_advice = advice_by_level(
        bone_level,
        "Выраженного костного риска по анкете немного.",
        "Стоит обратить внимание на витамин D, физическую нагрузку и причины снижения костной массы.",
        "Есть смысл обсудить оценку костной ткани и факторов остеопороза.",
    )

    connections = generate_connections(
        diabetes_score,
        ir_score,
        hypothyroid_score,
        hyperthyroid_score,
        pcos_score if pcos_score is not None else 0.0,
        bone_score,
        gender,
    )

    next_steps = generate_next_steps(
        diabetes_score,
        ir_score,
        hypothyroid_score,
        hyperthyroid_score,
        pcos_score if pcos_score is not None else 0.0,
        bone_score,
        bmi,
        fasting_glucose,
        hba1c,
    )

    # Результаты — одна связанная карта
    st.header("🗺️ Медицинская карта рисков")
    st.caption("Ниже — не диагноз, а структурированная карта вероятных слабых мест и взаимосвязей между ними.")

    # Ключевые метрики
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Диабет", score_to_text(diabetes_score))
    with m2:
        st.metric("Инсулинорезистентность / метаболизм", score_to_text(ir_score))
    with m3:
        st.metric("Щитовидная ось", score_to_text(max(hypothyroid_score, hyperthyroid_score)))

    m4, m5, m6 = st.columns(3)
    with m4:
        st.metric("PCOS", "—" if pcos_score is None else score_to_text(pcos_score))
    with m5:
        st.metric("Костный риск", score_to_text(bone_score))
    with m6:
        st.metric("ИМТ", f"{bmi:.1f}")

    st.progress(clamp(max(diabetes_score, ir_score, hypothyroid_score, hyperthyroid_score, bone_score) / 100.0))

    if diabetes_fallback_error:
        st.warning(diabetes_fallback_error)

    # Общий вывод
    strong_points = []
    if diabetes_score >= 60:
        strong_points.append("углеводный обмен")
    if ir_score >= 60:
        strong_points.append("инсулинорезистентность")
    if max(hypothyroid_score, hyperthyroid_score) >= 60:
        strong_points.append("щитовидная железа")
    if pcos_score is not None and pcos_score >= 60:
        strong_points.append("PCOS")
    if bone_score >= 60:
        strong_points.append("костная ткань")

    if strong_points:
        st.error("Зоны наибольшего внимания: " + ", ".join(strong_points) + ".")
    else:
        st.success("Пока нет одной ярко выраженной зоны риска; полезен профилактический контроль и поддержка образа жизни.")

    # Карточки заболеваний
    disease_cards = [
        {
            "name": "Диабет",
            "score": diabetes_score,
            "level": diabetes_level,
            "advice": diabetes_advice,
            "drivers": [
                "Симптомы диабета",
                "Возраст",
                "Вес / метаболическая нагрузка",
                "Наследственность" if family_history_diabetes else None,
            ],
        },
        {
            "name": "Инсулинорезистентность / метаболический синдром",
            "score": ir_score,
            "level": ir_level,
            "advice": ir_advice,
            "drivers": [
                "ИМТ",
                "Талия",
                "Сон и активность",
                "Симптомы углеводного обмена",
                "Наследственность" if family_history_diabetes else None,
            ],
        },
        {
            "name": "Щитовидная железа: гипофункция",
            "score": hypothyroid_score,
            "level": hypo_level,
            "advice": hypo_advice,
            "drivers": [
                "Холод / запоры / сухость кожи",
                "Утомляемость",
                "ТТГ / свободный T4",
                "Наследственность" if family_history_thyroid else None,
            ],
        },
        {
            "name": "Щитовидная железа: гиперфункция",
            "score": hyperthyroid_score,
            "level": hyper_level,
            "advice": hyper_advice,
            "drivers": [
                "Жара / сердцебиение / тремор",
                "Потеря веса",
                "ТТГ / свободный T4",
                "Наследственность" if family_history_thyroid else None,
            ],
        },
        {
            "name": "PCOS",
            "score": pcos_score,
            "level": pcos_level if pcos_score is not None else "Не оценен",
            "advice": pcos_advice,
            "drivers": [
                "Нерегулярный цикл",
                "Андрогенные симптомы",
                "Инсулинорезистентность",
            ],
        },
        {
            "name": "Эндокринная сеть",
            "score": network_score,
            "level": network_level,
            "advice": network_advice,
            "drivers": [
                "Совокупность всех осей",
                "Перекрёстные влияния",
                "Суммарная метаболическая нагрузка",
            ],
        },
        {
            "name": "Костная ткань / остеопения",
            "score": bone_score,
            "level": bone_level,
            "advice": bone_advice,
            "drivers": [
                "Возраст",
                "Переломы / стероиды",
                "Низкая активность / низкий ИМТ",
                "Наследственность" if family_history_osteoporosis else None,
            ],
        },
    ]

    for card in disease_cards:
        if card["score"] is None:
            continue
        # убираем None из drivers
        drivers = [d for d in card["drivers"] if d is not None]
        st.markdown(
            f"""
<div class="card">
  <div><strong>{card['name']}</strong> {badge(card['level'])}</div>
  <div style="margin-top:0.35rem;"><strong>Риск:</strong> {score_to_text(card['score'])}</div>
  <div class="muted" style="margin-top:0.35rem;"><strong>Основные драйверы:</strong> {", ".join(drivers)}</div>
  <div style="margin-top:0.5rem;">{card["advice"]}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.subheader("🔗 Как все связано между собой")
    if connections:
        for item in connections:
            st.write(f"- {item}")
    else:
        st.write("Явных взаимосвязей по анкете не выделено.")

    st.subheader("📌 Что стоит сделать дальше")
    for item in next_steps:
        st.write(f"- {item}")

    st.subheader("🧾 Краткая сводка")
    summary_rows = [
        ("Показатель", "Значение"),
        ("ИМТ", f"{bmi:.1f}"),
        ("Диабет", f"{score_to_text(diabetes_score)} ({diabetes_level})"),
        ("Инсулинорезистентность", f"{score_to_text(ir_score)} ({ir_level})"),
        ("Щитовидная гипофункция", f"{score_to_text(hypothyroid_score)} ({hypo_level})"),
        ("Щитовидная гиперфункция", f"{score_to_text(hyperthyroid_score)} ({hyper_level})"),
        ("PCOS", "—" if pcos_score is None else f"{score_to_text(pcos_score)} ({pcos_level})"),
        ("Эндокринная сеть", f"{score_to_text(network_score)} ({network_level})"),
        ("Костный риск", f"{score_to_text(bone_score)} ({bone_level})"),
    ]
    summary_df = pd.DataFrame(summary_rows[1:], columns=summary_rows[0])
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    with st.expander("Введенные данные"):
        st.write(f"**Возраст:** {age}")
        st.write(f"**Пол:** {gender_input}")
        st.write(f"**Рост:** {height_cm:.0f} см")
        st.write(f"**Вес:** {weight_kg:.1f} кг")
        st.write(f"**Талия:** {waist_cm:.0f} см")
        st.write(f"**Сон:** {sleep_hours:.1f} ч/сутки")
        st.write(f"**Активность:** {activity_level}")
        st.write(f"**Наследственность по диабету:** {'Да' if family_history_diabetes else 'Нет'}")
        st.write(f"**Наследственность по щитовидной железе:** {'Да' if family_history_thyroid else 'Нет'}")
        st.write(f"**Наследственность по остеопорозу:** {'Да' if family_history_osteoporosis else 'Нет'}")

        st.write("**Симптомы диабета:**")
        active_diab = [feature_names_ru.get(k, k) for k, v in diabetes_symptom_values.items() if v]
        st.write(active_diab if active_diab else "Нет отмеченных симптомов.")

        st.write("**Симптомы щитовидной железы:**")
        active_th = [feature_names_ru.get(k, k) for k, v in thyroid_values.items() if v]
        st.write(active_th if active_th else "Нет отмеченных симптомов.")

        if gender == 1:
            st.write("**PCOS-признаки:**")
            active_pcos = [feature_names_ru.get(k, k) for k, v in pcos_values.items() if v]
            st.write(active_pcos if active_pcos else "Нет отмеченных симптомов.")

        st.write("**Костный риск:**")
        active_bone = [feature_names_ru.get(k, k) for k, v in bone_values.items() if v]
        st.write(active_bone if active_bone else "Нет отмеченных симптомов.")

    
if enable_mfdfa:
        uploaded_series = parse_uploaded_glucose_file(glucose_file)
        manual_series = parse_series(glucose_series_text)
        series = uploaded_series if uploaded_series is not None else manual_series

        st.subheader("🧠 Результат MF-DFA")
        if series is None:
            st.info(
                "Нужен числовой ряд хотя бы из 12 значений. Можно вставить его вручную или загрузить файл .txt/.csv."
            )
        else:
            source_label = "из загруженного файла" if uploaded_series is not None else "из ручного ввода"
            st.caption(f"Источник ряда: {source_label}. Всего значений: {len(series)}.")
            result = mfdfa(series)
            if result is None:
                st.info(
                    "Ряд получен, но для MF-DFA всё ещё мало данных или они слишком однородны. "
                    "Попробуйте длиннее ряд — хотя бы 16–20 точек."
                )
            else:
                st.write(mfdfa_interpretation(result))

                comparison = compare_to_reference(result["width"])
                st.info(
                    f"Сравнение с эталоном (ширина спектра {comparison['reference']:.2f}): "
                    f"{comparison['status']}. Отклонение {comparison['delta']:+.3f}."
                )
                st.metric("Интерпретация по ширине спектра", interpret_complexity(result["width"]))

                mfdfa_df = pd.DataFrame(
                    {
                        "q": result["q_vals"],
                        "H(q)": result["Hq"],
                        "tau(q)": result["tau"],
                        "alpha": result["alpha"],
                        "f(alpha)": result["f_alpha"],
                    }
                )
                st.dataframe(mfdfa_df, use_container_width=True, hide_index=True)

                width = result["width"]
                if width < 0.12:
                    st.success("Для ряда глюкозы характерна низкая вариабельная сложность.")
                elif width < 0.25:
                    st.warning("Для ряда глюкозы характерна умеренная сложность и неоднородность.")
                else:
                    st.error("Для ряда глюкозы характерна высокая неоднородность — это исследовательский сигнал, а не диагноз.")

                c1, c2 = st.columns(2)
                with c1:
                    fig1 = plot_mfdfa_scaling(result)
                    if fig1 is not None:
                        st.pyplot(fig1, clear_figure=True, use_container_width=True)
                        plt.close(fig1)
                    else:
                        st.info("Не удалось построить график масштабирования: мало валидных масштабов.")
                with c2:
                    fig2 = plot_mfdfa_spectrum(result)
                    if fig2 is not None:
                        st.pyplot(fig2, clear_figure=True, use_container_width=True)
                        plt.close(fig2)
                    else:
                        st.info("Не удалось построить спектр: недостаточно валидных точек.")

                with st.expander("Подробности MF-DFA"):
                    st.write(f"**Ширина спектра:** {result['width']:.3f}")
                    st.write(f"**Средний H(q):** {result['mean_h']:.3f}")
                    st.write("**Интерпретация:** MF-DFA оценивает масштабную организацию колебаний глюкозы; это экспериментальный исследовательский показатель.")

else:
    st.info("👆 Заполните форму выше и нажмите «Собрать медицинскую карту».")


# ======================== ПОДВАЛ ========================
st.markdown("---")
st.caption(
    "Прототип создан в образовательных целях. Диагностические решения и назначения должен подтверждать врач. "
    "MF-DFA блок является экспериментальным исследовательским модулем; ряд можно вводить вручную или загружать файлом."
)
