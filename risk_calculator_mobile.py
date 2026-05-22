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

    :root {
        --risk-low: #1fa971;
        --risk-mid: #f0a500;
        --risk-high: #e53935;
    }

    .risk-card {
        background: var(--secondary-background-color);
        color: var(--text-color);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 18px;
        padding: 1rem 1rem 0.9rem 1rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.06);
    }
    .risk-header {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 0.75rem;
        margin-bottom: 0.65rem;
    }
    .risk-title {
        font-size: 1.12rem;
        font-weight: 800;
        line-height: 1.2;
        word-break: break-word;
    }
    .risk-badge {
        flex: 0 0 auto;
        padding: 0.28rem 0.65rem;
        border-radius: 999px;
        color: white;
        font-size: 0.8rem;
        font-weight: 800;
        white-space: nowrap;
    }
    .risk-percent {
        font-size: 1.9rem;
        font-weight: 900;
        line-height: 1;
        margin-bottom: 0.45rem;
        letter-spacing: -0.02em;
    }
    .risk-bar {
        width: 100%;
        height: 12px;
        border-radius: 999px;
        overflow: hidden;
        background: rgba(127, 127, 127, 0.18);
        margin-bottom: 0.65rem;
    }
    .risk-fill {
        height: 100%;
        border-radius: 999px;
    }
    .risk-summary {
        font-size: 0.95rem;
        line-height: 1.5;
        opacity: 0.95;
    }
    .risk-meta {
        margin-top: 0.65rem;
        padding-top: 0.65rem;
        border-top: 1px solid rgba(127, 127, 127, 0.12);
        font-size: 0.92rem;
        line-height: 1.45;
    }
    .risk-meta strong {
        opacity: 0.95;
    }
    .muted {
        color: var(--text-color);
        opacity: 0.8;
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
        .risk-title {
            font-size: 1.02rem;
        }
        .risk-percent {
            font-size: 1.7rem;
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
    "facial fullness": "Округлое (луноподобное) лицо",
    "purple striae": "Фиолетовые растяжки",
    "easy bruising": "Лёгкое появление синяков",
    "proximal weakness": "Проксимальная мышечная слабость",
    "centripetal obesity": "Центральное ожирение",
    "hypertension": "Повышенное артериальное давление",
    "depression": "Депрессивное настроение",
    "hyperpigmentation": "Гиперпигментация кожи",
    "salt craving": "Тяга к солёному",
    "orthostatic dizziness": "Головокружение при вставании",
    "nausea": "Тошнота",
    "vomiting": "Рвота",
    "weight loss": "Похудение",
    "low blood pressure": "Низкое артериальное давление",
    "autoimmune history": "Аутоиммунные заболевания в анамнезе",
    "kidney stones": "Почечные камни",
    "bone pain": "Боли в костях",
    "abdominal pain": "Боль в животе",
    "frequent urination": "Частое мочеиспускание",
    "thirst": "Жажда",
    "muscle weakness": "Мышечная слабость",
    "mental fog": "Затуманенность мышления",
}

def badge(level: str) -> str:
    colors = {
        "Низкая": "#1f8b4c",
        "Лёгкая": "#1f8b4c",
        "Легкая": "#1f8b4c",
        "Умеренная": "#c77700",
        "Средняя": "#c77700",
        "Выраженная": "#c62828",
        "Тяжёлая": "#c62828",
        "Тяжелая": "#c62828",
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


def theme_risk_color(score: float) -> str:
    if score < 30:
        return "var(--risk-low)"
    if score < 60:
        return "var(--risk-mid)"
    return "var(--risk-high)"


def render_risk_card(title, score, stage_text, advice, confidence_text="—", posterior_text="—", drivers=None, signals=None):
    drivers = [d for d in (drivers or []) if d]
    signals = [s for s in (signals or []) if s]
    color = theme_risk_color(float(score))
    clipped_advice = advice if advice else ""
    percent_width = max(0.0, min(100.0, float(score)))
    st.markdown(
        f"""
<div class="risk-card">
  <div class="risk-header">
    <div class="risk-title">{title}</div>
    <div class="risk-badge" style="background:{color};">{stage_text}</div>
  </div>
  <div class="risk-percent">{float(score):.1f}%</div>
  <div class="risk-bar"><div class="risk-fill" style="width:{percent_width}%; background:{color};"></div></div>
  <div class="risk-summary">{clipped_advice}</div>
</div>
""",
        unsafe_allow_html=True,
    )
    with st.expander("Подробнее", expanded=False):
        st.write(f"**Уверенность:** {confidence_text}")
        st.write(f"**Апостериорная вероятность:** {posterior_text}")
        if drivers:
            st.write("**Основные драйверы:** " + ", ".join(drivers))
        if signals:
            st.write("**Отмеченные признаки:** " + ", ".join(signals))
        else:
            st.write("**Отмеченные признаки:** Нет выраженных симптомов по этому блоку.")

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

cushing_symptoms = [
    "facial fullness",
    "purple striae",
    "easy bruising",
    "proximal weakness",
    "hypertension",
    "centripetal obesity",
    "depression",
]

addison_symptoms = [
    "hyperpigmentation",
    "salt craving",
    "orthostatic dizziness",
    "nausea",
    "vomiting",
    "weight loss",
    "low blood pressure",
    "autoimmune history",
]

hyperparathyroidism_symptoms = [
    "kidney stones",
    "bone pain",
    "constipation",
    "abdominal pain",
    "depression",
    "muscle weakness",
    "frequent urination",
    "thirst",
    "fatigue",
]

cushing_labels = {
    "facial fullness": "Округлое (луноподобное) лицо",
    "purple striae": "Фиолетовые растяжки",
    "easy bruising": "Лёгкое появление синяков",
    "proximal weakness": "Проксимальная мышечная слабость",
    "hypertension": "Повышенное артериальное давление",
    "centripetal obesity": "Центральное ожирение",
    "depression": "Депрессивное настроение",
}

addison_labels = {
    "hyperpigmentation": "Гиперпигментация кожи",
    "salt craving": "Тяга к солёному",
    "orthostatic dizziness": "Головокружение при вставании",
    "nausea": "Тошнота",
    "vomiting": "Рвота",
    "weight loss": "Похудение",
    "low blood pressure": "Низкое артериальное давление",
    "autoimmune history": "Аутоиммунные заболевания в анамнезе",
}

hyperpara_labels = {
    "kidney stones": "Почечные камни",
    "bone pain": "Боли в костях",
    "constipation": "Запоры",
    "abdominal pain": "Боль в животе",
    "depression": "Депрессивное настроение",
    "muscle weakness": "Мышечная слабость",
    "frequent urination": "Частое мочеиспускание",
    "thirst": "Жажда",
    "fatigue": "Утомляемость",
}

DISEASE_PRIORS = {
    "Диабет": 0.12,
    "Инсулинорезистентность / метаболический синдром": 0.20,
    "Щитовидная железа: гипофункция": 0.08,
    "Щитовидная железа: гиперфункция": 0.03,
    "PCOS": 0.10,
    "Эндокринная сеть": 0.15,
    "Костная ткань / остеопения": 0.15,
    "Синдром Кушинга": 0.01,
    "Болезнь Аддисона": 0.005,
    "Первичный гиперпаратиреоз": 0.01,
}

def count_positive_flags(flags: dict) -> int:
    return int(sum(1 for v in flags.values() if bool(v)))

def severity_label(score: float | None) -> str:
    if score is None:
        return "Не оценен"
    if score < 20:
        return "Низкая"
    if score < 40:
        return "Лёгкая"
    if score < 60:
        return "Умеренная"
    if score < 80:
        return "Выраженная"
    return "Тяжёлая"

def evidence_confidence(score: float, symptom_count: int = 0, lab_count: int = 0, red_flag_count: int = 0, family_history: bool = False) -> float:
    conf = 30.0 + 0.35 * clamp(score) + 4.0 * min(symptom_count, 6) + 6.0 * min(lab_count, 4) + 7.0 * red_flag_count
    if family_history:
        conf += 4.0
    return clamp(conf, 0.0, 100.0)

def bayes_like_probability(score: float, prior: float = 0.05) -> float:
    prior = float(min(max(prior, 1e-4), 0.9999))
    prior_logit = math.log(prior / (1.0 - prior))
    evidence_logit = (clamp(score) - 50.0) / 12.0
    posterior = 1.0 / (1.0 + math.exp(-(prior_logit + evidence_logit)))
    return clamp(100.0 * posterior, 0.0, 100.0)

def assess_risk(score: float | None, disease_name: str, symptom_count: int = 0, lab_count: int = 0, red_flag_count: int = 0, family_history: bool = False) -> dict:
    if score is None:
        return {
            "stage": "Не оценен",
            "confidence": None,
            "posterior": None,
        }
    return {
        "stage": severity_label(score),
        "confidence": evidence_confidence(score, symptom_count, lab_count, red_flag_count, family_history),
        "posterior": bayes_like_probability(score, DISEASE_PRIORS.get(disease_name, 0.05)),
    }

def score_to_text(score: float) -> str:
    return f"{clamp(score):.1f}%"

def summarize_flags(flags):
    if not flags:
        return "Явно выраженных групп риска по анкете не выделено."
    return " / ".join(flags)

# ======================== ФУНКЦИИ РИСКА ========================
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

cushing_symptoms = [
    "facial fullness",
    "purple striae",
    "easy bruising",
    "proximal weakness",
    "hypertension",
    "centripetal obesity",
    "depression",
]

addison_symptoms = [
    "hyperpigmentation",
    "salt craving",
    "orthostatic dizziness",
    "nausea",
    "vomiting",
    "weight loss",
    "low blood pressure",
    "autoimmune history",
]

hyperparathyroidism_symptoms = [
    "kidney stones",
    "bone pain",
    "constipation",
    "abdominal pain",
    "depression",
    "muscle weakness",
    "frequent urination",
    "thirst",
    "fatigue",
]

cushing_labels = {
    "facial fullness": "Округлое (луноподобное) лицо",
    "purple striae": "Фиолетовые растяжки",
    "easy bruising": "Лёгкое появление синяков",
    "proximal weakness": "Проксимальная мышечная слабость",
    "hypertension": "Повышенное артериальное давление",
    "centripetal obesity": "Центральное ожирение",
    "depression": "Депрессивное настроение",
}

addison_labels = {
    "hyperpigmentation": "Гиперпигментация кожи",
    "salt craving": "Тяга к солёному",
    "orthostatic dizziness": "Головокружение при вставании",
    "nausea": "Тошнота",
    "vomiting": "Рвота",
    "weight loss": "Похудение",
    "low blood pressure": "Низкое артериальное давление",
    "autoimmune history": "Аутоиммунные заболевания в анамнезе",
}

hyperpara_labels = {
    "kidney stones": "Почечные камни",
    "bone pain": "Боли в костях",
    "constipation": "Запоры",
    "abdominal pain": "Боль в животе",
    "depression": "Депрессивное настроение",
    "muscle weakness": "Мышечная слабость",
    "frequent urination": "Частое мочеиспускание",
    "thirst": "Жажда",
    "fatigue": "Утомляемость",
}

DISEASE_PRIORS = {
    "Диабет": 0.12,
    "Инсулинорезистентность / метаболический синдром": 0.20,
    "Щитовидная железа: гипофункция": 0.08,
    "Щитовидная железа: гиперфункция": 0.03,
    "PCOS": 0.10,
    "Эндокринная сеть": 0.15,
    "Костная ткань / остеопения": 0.15,
    "Синдром Кушинга": 0.01,
    "Болезнь Аддисона": 0.005,
    "Первичный гиперпаратиреоз": 0.01,
}

def count_positive_flags(flags: dict) -> int:
    return int(sum(1 for v in flags.values() if bool(v)))

def severity_label(score: float | None) -> str:
    if score is None:
        return "Не оценен"
    if score < 20:
        return "Низкая"
    if score < 40:
        return "Лёгкая"
    if score < 60:
        return "Умеренная"
    if score < 80:
        return "Выраженная"
    return "Тяжёлая"

def evidence_confidence(score: float, symptom_count: int = 0, lab_count: int = 0, red_flag_count: int = 0, family_history: bool = False) -> float:
    conf = 30.0 + 0.35 * clamp(score) + 4.0 * min(symptom_count, 6) + 6.0 * min(lab_count, 4) + 7.0 * red_flag_count
    if family_history:
        conf += 4.0
    return clamp(conf, 0.0, 100.0)

def bayes_like_probability(score: float, prior: float = 0.05) -> float:
    prior = float(min(max(prior, 1e-4), 0.9999))
    prior_logit = math.log(prior / (1.0 - prior))
    evidence_logit = (clamp(score) - 50.0) / 12.0
    posterior = 1.0 / (1.0 + math.exp(-(prior_logit + evidence_logit)))
    return clamp(100.0 * posterior, 0.0, 100.0)

def assess_risk(score: float | None, disease_name: str, symptom_count: int = 0, lab_count: int = 0, red_flag_count: int = 0, family_history: bool = False) -> dict:
    if score is None:
        return {
            "stage": "Не оценен",
            "confidence": None,
            "posterior": None,
        }
    return {
        "stage": severity_label(score),
        "confidence": evidence_confidence(score, symptom_count, lab_count, red_flag_count, family_history),
        "posterior": bayes_like_probability(score, DISEASE_PRIORS.get(disease_name, 0.05)),
    }

def diabetes_age_modifier(age):
    """
    Мягкий возрастной модификатор.
    Возраст НЕ должен доминировать над симптомами и лабораториями.
    """

    if age < 35:
        return 1.00

    elif age < 45:
        return 1.05

    elif age < 55:
        return 1.10

    elif age < 65:
        return 1.15

    else:
        return 1.20

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
    score = 5.0
    score = min(score, 97)
    score *= diabetes_age_modifier(age)
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

def cushing_proxy(age, cushing_values, bmi, glucocorticoids, fasting_glucose, hba1c):
    score = 0.0
    score += 16 * yes(cushing_values.get("facial fullness"))
    score += 16 * yes(cushing_values.get("purple striae"))
    score += 12 * yes(cushing_values.get("easy bruising"))
    score += 14 * yes(cushing_values.get("proximal weakness"))
    score += 10 * yes(cushing_values.get("hypertension"))
    score += 12 * yes(cushing_values.get("centripetal obesity"))
    score += 8 * yes(cushing_values.get("depression"))

    if glucocorticoids:
        score += 18
    if bmi >= 30:
        score += 6
    if age >= 40:
        score += 4
    if fasting_glucose and fasting_glucose >= 100:
        score += 4
    if hba1c and hba1c >= 5.7:
        score += 4

    return clamp(score)

def addison_proxy(age, addison_values):
    score = 0.0
    score += 16 * yes(addison_values.get("hyperpigmentation"))
    score += 14 * yes(addison_values.get("salt craving"))
    score += 14 * yes(addison_values.get("orthostatic dizziness"))
    score += 12 * yes(addison_values.get("nausea"))
    score += 12 * yes(addison_values.get("vomiting"))
    score += 14 * yes(addison_values.get("weight loss"))
    score += 16 * yes(addison_values.get("low blood pressure"))
    score += 10 * yes(addison_values.get("autoimmune history"))

    if age < 50:
        score += 3

    return clamp(score)

def hyperparathyroid_proxy(age, hyperpara_values, serum_calcium):
    score = 0.0
    score += 16 * yes(hyperpara_values.get("kidney stones"))
    score += 12 * yes(hyperpara_values.get("bone pain"))
    score += 10 * yes(hyperpara_values.get("constipation"))
    score += 8 * yes(hyperpara_values.get("abdominal pain"))
    score += 8 * yes(hyperpara_values.get("depression"))
    score += 10 * yes(hyperpara_values.get("muscle weakness"))
    score += 8 * yes(hyperpara_values.get("frequent urination"))
    score += 8 * yes(hyperpara_values.get("thirst"))
    score += 8 * yes(hyperpara_values.get("fatigue"))

    if serum_calcium and serum_calcium > 0:
        if serum_calcium >= 10.5:
            score += min(28, (serum_calcium - 10.5) * 8)
        elif serum_calcium >= 10.0:
            score += 8

    if age >= 50:
        score += 4

    return clamp(score)

def generate_connections(diabetes_score, ir_score, hypo_score, hyper_score, pcos_score, bone_score, sex, cushing_score=None, addison_score=None, hyperpara_score=None):
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
    if cushing_score is not None and cushing_score >= 50:
        items.append("Признаки гиперкортицизма могут усиливать инсулинорезистентность, давление и риск потери костной массы.")
    if addison_score is not None and addison_score >= 50:
        items.append("Аутоиммунный профиль и симптомы Аддисона стоит сопоставить с другими эндокринными осами.")
    if hyperpara_score is not None and hyperpara_score >= 50:
        items.append("Гиперкальциемия и костные симптомы могут сочетаться с повышенным риском камней и остеопении.")
    return items

def generate_next_steps(diabetes_score, ir_score, hypo_score, hyper_score, pcos_score, bone_score, bmi, fasting_glucose, hba1c, cushing_score=None, addison_score=None, hyperpara_score=None, serum_calcium=None):
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
    if cushing_score is not None and cushing_score >= 50:
        steps.append("Обсудить признаки гиперкортицизма; при длительном приёме глюкокортикоидов нужна очная оценка.")
    if addison_score is not None and addison_score >= 50:
        steps.append("Проверить электролиты, утренний кортизол и АСТН по назначению врача при подозрении на Аддисона.")
    if hyperpara_score is not None and hyperpara_score >= 50:
        steps.append("Проверить кальций, альбумин, витамин D, ПТГ и риск почечных камней.")
    if serum_calcium is not None and serum_calcium >= 10.5:
        steps.append("Повышенный кальций требует перепроверки и очной оценки, особенно при костных симптомах.")

    return steps


def generate_red_flags(diabetes_score, ir_score, hypo_score, hyper_score, pcos_score, bone_score, cushing_score, addison_score, hyperpara_score, fasting_glucose, hba1c, tsh_value, ft4_value, serum_calcium):
    flags = []
    if fasting_glucose and fasting_glucose >= 126:
        flags.append("Глюкоза натощак в диагностическом диапазоне диабета.")
    if hba1c and hba1c >= 6.5:
        flags.append("HbA1c в диагностическом диапазоне диабета.")
    if tsh_value and tsh_value >= 10:
        flags.append("ТТГ выше 10 мМЕ/л — требуется очная оценка щитовидной железы.")
    if tsh_value and tsh_value < 0.1 and ft4_value and ft4_value > 1.8:
        flags.append("Профиль совместим с выраженным тиреотоксикозом.")
    if cushing_score is not None and cushing_score >= 70:
        flags.append("Картина слабо совместима с гиперкортицизмом — нужна очная оценка.")
    if addison_score is not None and addison_score >= 70:
        flags.append("Картина требует исключения надпочечниковой недостаточности.")
    if hyperpara_score is not None and hyperpara_score >= 70:
        flags.append("Подозрение на гиперкальциемию / гиперпаратиреоз.")
    if serum_calcium and serum_calcium >= 11.0:
        flags.append("Значимо повышенный кальций — нужна перепроверка.")
    if bone_score >= 75:
        flags.append("Высокий костный риск — имеет смысл обсудить денситометрию.")
    return flags

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
st.caption("Сверните ненужные блоки, чтобы экран был компактнее. Симптомы сгруппированы по разделам.")

def render_symptom_checkboxes(features, values_dict, key_prefix, labels_map=None, columns=2, disabled=False):
    cols = st.columns(columns)
    for idx, feature in enumerate(features):
        label = labels_map.get(feature, feature.replace("_", " ").title()) if labels_map else feature.replace("_", " ").title()
        with cols[idx % columns]:
            values_dict[feature] = st.checkbox(label, key=f"{key_prefix}_{feature}", disabled=disabled)
    return values_dict

col_age, col_gender = st.columns(2)
with col_age:
    age = st.slider("Возраст (полных лет)", min_value=18, max_value=90, value=40, help="Укажите ваш возраст")
with col_gender:
    gender_input = st.radio("Пол", options=["Мужской", "Женский"], help="Выберите пол")
gender = 0 if gender_input == "Мужской" else 1

with st.expander("🧬 Наследственность", expanded=False):
    family_history_diabetes = st.checkbox("Наследственность по диабету 2 типа (родители, сиблинги)")
    family_history_thyroid = st.checkbox("Наследственность по заболеваниям щитовидной железы")
    family_history_osteoporosis = st.checkbox("Наследственность по остеопорозу")

with st.expander("🧩 Базовые данные", expanded=True):
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

with st.expander("🍬 Диабет и симптомы обмена", expanded=True):
    st.caption("Отметьте признаки, которые у вас наблюдаются.")
    diabetes_symptom_values = {}
    diabetes_features = expected_features[2:]
    render_symptom_checkboxes(diabetes_features, diabetes_symptom_values, "dm", feature_names_ru, columns=2)

with st.expander("🦋 Щитовидная железа", expanded=False):
    thyroid_values = {}
    render_symptom_checkboxes(thyroid_symptoms, thyroid_values, "th", feature_names_ru, columns=2)

with st.expander("♀️ Женский гормональный блок (PCOS)", expanded=False):
    pcos_values = {}
    if gender == 1:
        st.caption("Этот блок активен только для женщин.")
        render_symptom_checkboxes(pcos_symptoms, pcos_values, "pcos", feature_names_ru, columns=2)
    else:
        st.info("PCOS-блок для мужчин не оценивается.")
        for feature in pcos_symptoms:
            pcos_values[feature] = False

with st.expander("🦴 Костный риск / остеопения", expanded=False):
    bone_values = {}
    render_symptom_checkboxes(bone_risk_features, bone_values, "bone", feature_names_ru, columns=2)

with st.expander("🩸 Дополнительные эндокринные блоки", expanded=False):
    st.caption("Здесь собраны редкие, но клинически значимые синдромы.")

    with st.expander("Синдром Кушинга", expanded=False):
        cushing_values = {}
        render_symptom_checkboxes(cushing_symptoms, cushing_values, "cushing", cushing_labels, columns=2)

    with st.expander("Болезнь Аддисона", expanded=False):
        addison_values = {}
        render_symptom_checkboxes(addison_symptoms, addison_values, "addison", addison_labels, columns=2)

    with st.expander("Первичный гиперпаратиреоз", expanded=False):
        hyperpara_values = {}
        render_symptom_checkboxes(hyperparathyroidism_symptoms, hyperpara_values, "hyperpara", hyperpara_labels, columns=2)

with st.expander("🧪 Анализы (если уже есть)", expanded=False):
    col_fg, col_hba1c, col_tsh, col_ft4 = st.columns(4)
    with col_fg:
        fasting_glucose = st.number_input("Глюкоза натощак, мг/дл", min_value=0.0, max_value=1000.0, value=0.0, step=1.0, help="0 = не указывать")
    with col_hba1c:
        hba1c = st.number_input("HbA1c, %", min_value=0.0, max_value=20.0, value=0.0, step=0.1, help="0 = не указывать")
    with col_tsh:
        tsh_value = st.number_input("ТТГ, мМЕ/л", min_value=0.0, max_value=100.0, value=0.0, step=0.1, help="0 = не указывать")
    with col_ft4:
        ft4_value = st.number_input("Св. T4, нг/дл", min_value=0.0, max_value=10.0, value=0.0, step=0.1, help="0 = не указывать")

    serum_calcium = st.number_input("Кальций общий, мг/дл", min_value=0.0, max_value=20.0, value=0.0, step=0.1, help="0 = не указывать")

with st.expander("📈 Мультифрактальный анализ гликемии (экспериментально)", expanded=False):
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
    cushing_rule_score = cushing_proxy(age, cushing_values, bmi, bone_values.get("glucocorticoids"), fasting_glucose, hba1c)
    addison_rule_score = addison_proxy(age, addison_values)
    hyperpara_rule_score = hyperparathyroid_proxy(age, hyperpara_values, serum_calcium)

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
            0.15 * diabetes_score
            + 0.15 * ir_score
            + 0.12 * hypothyroid_score
            + 0.12 * hyperthyroid_score
            + 0.10 * (0.0 if pcos_score is None else pcos_score)
            + 0.10 * bone_score
            + 0.10 * obesity_score
            + 0.08 * cushing_rule_score
            + 0.05 * addison_rule_score
            + 0.03 * hyperpara_rule_score
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
    cushing_level = risk_level(cushing_rule_score)
    addison_level = risk_level(addison_rule_score)
    hyperpara_level = risk_level(hyperpara_rule_score)

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

    cushing_advice = advice_by_level(
        cushing_level,
        "Выраженных признаков гиперкортицизма немного.",
        "Стоит перепроверить давление, вес, стрии и факт приёма глюкокортикоидов.",
        "Картина может соответствовать гиперкортицизму; нужна очная оценка.",
    )

    addison_advice = advice_by_level(
        addison_level,
        "Выраженных признаков надпочечниковой недостаточности немного.",
        "Стоит обратить внимание на давление, соль, тошноту и потерю веса.",
        "Нужно очно исключать болезнь Аддисона.",
    )

    hyperpara_advice = advice_by_level(
        hyperpara_level,
        "Выраженных признаков гиперпаратиреоза немного.",
        "Есть смысл перепроверить кальций и симптомы со стороны костей / почек.",
        "Нужна очная оценка на гиперпаратиреоз и гиперкальциемию.",
    )

    diabetes_assessment = assess_risk(
        diabetes_score,
        "Диабет",
        symptom_count=count_positive_flags(diabetes_symptom_values),
        lab_count=int(fasting_glucose > 0) + int(hba1c > 0),
        red_flag_count=1 if (fasting_glucose and fasting_glucose >= 126) or (hba1c and hba1c >= 6.5) else 0,
        family_history=family_history_diabetes,
    )
    ir_assessment = assess_risk(
        ir_score,
        "Инсулинорезистентность / метаболический синдром",
        symptom_count=count_positive_flags(diabetes_symptom_values),
        lab_count=int(fasting_glucose > 0) + int(hba1c > 0),
        red_flag_count=1 if bmi >= 30 else 0,
        family_history=family_history_diabetes,
    )
    hypo_assessment = assess_risk(
        hypothyroid_score,
        "Щитовидная железа: гипофункция",
        symptom_count=count_positive_flags(thyroid_values),
        lab_count=int(tsh_value > 0) + int(ft4_value > 0),
        red_flag_count=1 if tsh_value and tsh_value >= 10 else 0,
        family_history=family_history_thyroid,
    )
    hyper_assessment = assess_risk(
        hyperthyroid_score,
        "Щитовидная железа: гиперфункция",
        symptom_count=count_positive_flags(thyroid_values),
        lab_count=int(tsh_value > 0) + int(ft4_value > 0),
        red_flag_count=1 if (tsh_value and tsh_value < 0.1 and ft4_value and ft4_value > 1.8) else 0,
        family_history=family_history_thyroid,
    )
    pcos_assessment = None if pcos_score is None else assess_risk(
        pcos_score,
        "PCOS",
        symptom_count=count_positive_flags(pcos_values),
        lab_count=int(fasting_glucose > 0) + int(hba1c > 0),
        red_flag_count=0,
        family_history=False,
    )
    bone_assessment = assess_risk(
        bone_score,
        "Костная ткань / остеопения",
        symptom_count=count_positive_flags(bone_values),
        lab_count=0,
        red_flag_count=1 if family_history_osteoporosis else 0,
        family_history=family_history_osteoporosis,
    )
    cushing_assessment = assess_risk(
        cushing_rule_score,
        "Синдром Кушинга",
        symptom_count=count_positive_flags(cushing_values),
        lab_count=int(fasting_glucose > 0) + int(hba1c > 0),
        red_flag_count=1 if cushing_rule_score >= 70 else 0,
        family_history=False,
    )
    addison_assessment = assess_risk(
        addison_rule_score,
        "Болезнь Аддисона",
        symptom_count=count_positive_flags(addison_values),
        lab_count=0,
        red_flag_count=1 if addison_rule_score >= 70 else 0,
        family_history=False,
    )
    hyperpara_assessment = assess_risk(
        hyperpara_rule_score,
        "Первичный гиперпаратиреоз",
        symptom_count=count_positive_flags(hyperpara_values),
        lab_count=int(serum_calcium > 0),
        red_flag_count=1 if (serum_calcium and serum_calcium >= 11.0) else 0,
        family_history=False,
    )

    connections = generate_connections(
        diabetes_score,
        ir_score,
        hypothyroid_score,
        hyperthyroid_score,
        pcos_score if pcos_score is not None else 0.0,
        bone_score,
        gender,
        cushing_rule_score,
        addison_rule_score,
        hyperpara_rule_score,
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
        cushing_rule_score,
        addison_rule_score,
        hyperpara_rule_score,
        serum_calcium,
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

    m7, m8, m9 = st.columns(3)
    with m7:
        st.metric("Кушинг", score_to_text(cushing_rule_score))
    with m8:
        st.metric("Аддисон", score_to_text(addison_rule_score))
    with m9:
        st.metric("Гиперпаратиреоз", score_to_text(hyperpara_rule_score))

    st.progress(clamp(max(diabetes_score, ir_score, hypothyroid_score, hyperthyroid_score, bone_score, cushing_rule_score, addison_rule_score, hyperpara_rule_score) / 100.0))

    if diabetes_fallback_error:
        st.warning(diabetes_fallback_error)

    red_flags = generate_red_flags(
        diabetes_score,
        ir_score,
        hypothyroid_score,
        hyperthyroid_score,
        pcos_score,
        bone_score,
        cushing_rule_score,
        addison_rule_score,
        hyperpara_rule_score,
        fasting_glucose,
        hba1c,
        tsh_value,
        ft4_value,
        serum_calcium,
    )
    if red_flags:
        st.error("Красные флаги: " + " | ".join(red_flags))

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
    if cushing_rule_score >= 60:
        strong_points.append("гиперкортицизм")
    if addison_rule_score >= 60:
        strong_points.append("надпочечники")
    if hyperpara_rule_score >= 60:
        strong_points.append("кальциевый обмен")

    if strong_points:
        st.error("Зоны наибольшего внимания: " + ", ".join(strong_points) + ".")
    else:
        st.success("Пока нет одной ярко выраженной зоны риска; полезен профилактический контроль и поддержка образа жизни.")

    # Карточки заболеваний
    disease_cards = [
        {
            "name": "Диабет",
            "score": diabetes_score,
            "assessment": diabetes_assessment,
            "advice": diabetes_advice,
            "drivers": ["Симптомы диабета", "Возраст", "Вес / метаболическая нагрузка", "Наследственность" if family_history_diabetes else None],
            "signals": [feature_names_ru.get(k, k) for k, v in diabetes_symptom_values.items() if v],
        },
        {
            "name": "Инсулинорезистентность / метаболический синдром",
            "score": ir_score,
            "assessment": ir_assessment,
            "advice": ir_advice,
            "drivers": ["ИМТ", "Талия", "Сон и активность", "Симптомы углеводного обмена", "Наследственность" if family_history_diabetes else None],
            "signals": [
                f"ИМТ {bmi:.1f}",
                f"Талия {waist_cm:.0f} см",
                f"Сон {sleep_hours:.1f} ч/сутки",
            ] + [feature_names_ru.get(k, k) for k, v in diabetes_symptom_values.items() if v][:4],
        },
        {
            "name": "Щитовидная железа: гипофункция",
            "score": hypothyroid_score,
            "assessment": hypo_assessment,
            "advice": hypo_advice,
            "drivers": ["Холод / запоры / сухость кожи", "Утомляемость", "ТТГ / свободный T4", "Наследственность" if family_history_thyroid else None],
            "signals": [feature_names_ru.get(k, k) for k, v in thyroid_values.items() if v and k in {"cold intolerance", "constipation", "fatigue", "dry skin", "Alopecia", "weakness", "neck swelling"}],
        },
        {
            "name": "Щитовидная железа: гиперфункция",
            "score": hyperthyroid_score,
            "assessment": hyper_assessment,
            "advice": hyper_advice,
            "drivers": ["Жара / сердцебиение / тремор", "Потеря веса", "ТТГ / свободный T4", "Наследственность" if family_history_thyroid else None],
            "signals": [feature_names_ru.get(k, k) for k, v in thyroid_values.items() if v and k in {"heat intolerance", "palpitations", "tremor", "anxiety", "diarrhea", "sudden weight loss", "neck swelling", "weakness"}],
        },
        {
            "name": "PCOS",
            "score": pcos_score,
            "assessment": pcos_assessment,
            "advice": pcos_advice,
            "drivers": ["Нерегулярный цикл", "Андрогенные симптомы", "Инсулинорезистентность"],
            "signals": [feature_names_ru.get(k, k) for k, v in pcos_values.items() if v],
        },
        {
            "name": "Эндокринная сеть",
            "score": network_score,
            "assessment": assess_risk(network_score, "Эндокринная сеть", symptom_count=0, lab_count=0, red_flag_count=len(red_flags), family_history=False),
            "advice": network_advice,
            "drivers": ["Совокупность всех осей", "Перекрёстные влияния", "Суммарная метаболическая нагрузка"],
            "signals": ["Интегральная оценка взаимосвязей"],
        },
        {
            "name": "Костная ткань / остеопения",
            "score": bone_score,
            "assessment": bone_assessment,
            "advice": bone_advice,
            "drivers": ["Возраст", "Переломы / стероиды", "Низкая активность / низкий ИМТ", "Наследственность" if family_history_osteoporosis else None],
            "signals": [feature_names_ru.get(k, k) for k, v in bone_values.items() if v],
        },
        {
            "name": "Синдром Кушинга",
            "score": cushing_rule_score,
            "assessment": cushing_assessment,
            "advice": cushing_advice,
            "drivers": ["Гиперкортицизм", "Глюкокортикоиды", "Центральное ожирение", "Гипертензия"],
            "signals": [cushing_labels.get(k, k) for k, v in cushing_values.items() if v],
        },
        {
            "name": "Болезнь Аддисона",
            "score": addison_rule_score,
            "assessment": addison_assessment,
            "advice": addison_advice,
            "drivers": ["Аутоиммунность", "Гипотензия", "Тяга к солёному", "Гиперпигментация"],
            "signals": [addison_labels.get(k, k) for k, v in addison_values.items() if v],
        },
        {
            "name": "Первичный гиперпаратиреоз",
            "score": hyperpara_rule_score,
            "assessment": hyperpara_assessment,
            "advice": hyperpara_advice,
            "drivers": ["Кальций", "Почки", "Кости", "Когнитивные/общие симптомы"],
            "signals": [hyperpara_labels.get(k, k) for k, v in hyperpara_values.items() if v],
        },
    ]

    for card in disease_cards:
        if card["score"] is None:
            continue
        assessment = card.get("assessment", {})
        drivers = [d for d in card.get("drivers", []) if d is not None]
        signals = card.get("signals", [])
        confidence_text = "—" if assessment.get("confidence") is None else f"{assessment['confidence']:.0f}%"
        posterior_text = "—" if assessment.get("posterior") is None else f"{assessment['posterior']:.0f}%"
        stage_text = assessment.get("stage", "Не оценен")

        render_risk_card(
            title=card["name"],
            score=card["score"],
            stage_text=stage_text,
            advice=card["advice"],
            confidence_text=confidence_text,
            posterior_text=posterior_text,
            drivers=drivers,
            signals=signals,
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
        ("Кушинг", f"{score_to_text(cushing_rule_score)} ({cushing_level})"),
        ("Аддисон", f"{score_to_text(addison_rule_score)} ({addison_level})"),
        ("Гиперпаратиреоз", f"{score_to_text(hyperpara_rule_score)} ({hyperpara_level})"),
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
        st.write(f"**Кальций общий:** {serum_calcium:.1f} мг/дл")
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

        st.write("**Синдром Кушинга:**")
        active_cushing = [cushing_labels.get(k, k) for k, v in cushing_values.items() if v]
        st.write(active_cushing if active_cushing else "Нет отмеченных симптомов.")

        st.write("**Болезнь Аддисона:**")
        active_addison = [addison_labels.get(k, k) for k, v in addison_values.items() if v]
        st.write(active_addison if active_addison else "Нет отмеченных симптомов.")

        st.write("**Первичный гиперпаратиреоз:**")
        active_hyperpara = [hyperpara_labels.get(k, k) for k, v in hyperpara_values.items() if v]
        st.write(active_hyperpara if active_hyperpara else "Нет отмеченных симптомов.")

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
