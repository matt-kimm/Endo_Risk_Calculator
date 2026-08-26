import math
import io
import re
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, recall_score,
                             f1_score, matthews_corrcoef, brier_score_loss, accuracy_score, confusion_matrix)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import shap
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# ======================== НАСТРОЙКА СТРАНИЦЫ ========================

st.set_page_config(
    page_title="Endo Risk Calculator",
    page_icon="🩺"
)

# Инициализация состояний для expander'ов
if 'expander_states' not in st.session_state:
    st.session_state['expander_states'] = {
        'heredity': False,
        'basic_data': True,
        'diabetes': True,
        'thyroid': False,
        'pcos': False,
        'bone': False,
        'additional': False,
        'labs': False,
        'mfdfa': False,
        'training': False,
        'analysis': False,
        'mfdfa': False
    }

# Инициализация состояния для MF-DFA
if 'mfdfa_series_list' not in st.session_state:
    st.session_state['mfdfa_series_list'] = [{"name": "Глюкоза", "data": ""}]

if 'mfdfa_results' not in st.session_state:
    st.session_state['mfdfa_results'] = None

if 'analysis_df' not in st.session_state:
    st.session_state['analysis_df'] = None

# Функция для сохранения состояния expander'а
def toggle_expander(key):
    st.session_state['expander_states'][key] = not st.session_state['expander_states'][key]

# Инициализация результатов (чтобы избежать NameError)
if 'results' not in st.session_state:
    st.session_state['results'] = {
        'diabetes_score': 0,
        'ir_score': 0,
        'obesity_score': 0,
        'hypothyroid_score': 0,
        'hyperthyroid_score': 0,
        'pcos_score': None,
        'bone_score': 0,
        'metabolic_score': 0,
        'cushing_rule_score': 0,
        'addison_rule_score': 0,
        'hyperpara_rule_score': 0,
        'network_score': 0,
    }

diabetes_score = st.session_state['results']['diabetes_score']
ir_score = st.session_state['results']['ir_score']
obesity_score = st.session_state['results']['obesity_score']
hypothyroid_score = st.session_state['results']['hypothyroid_score']
hyperthyroid_score = st.session_state['results']['hyperthyroid_score']
pcos_score = st.session_state['results']['pcos_score']
bone_score = st.session_state['results']['bone_score']
metabolic_score = st.session_state['results']['metabolic_score']
cushing_rule_score = st.session_state['results']['cushing_rule_score']
addison_rule_score = st.session_state['results']['addison_rule_score']
hyperpara_rule_score = st.session_state['results']['hyperpara_rule_score']
network_score = st.session_state['results']['network_score']

if 'input_data' not in st.session_state:
    st.session_state['input_data'] = {}

# ======================== CSS (сокращён, основные стили оставлены) ========================
st.markdown(
    """
<style>
    .main > div { max-width: 100%; padding-left: 0.5rem; padding-right: 0.5rem; }
    .stCheckbox, .stRadio, .stSlider, .stNumberInput, .stSelectbox { margin-bottom: 0.75rem; }
    label { font-size: 16px !important; font-weight: 500 !important; }
    .stButton button, .stForm button { width: 100%; font-size: 1.1rem !important; padding: 0.6rem !important; border-radius: 10px !important; }
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    .stMetric { text-align: left !important; }
    .streamlit-expanderHeader { font-size: 1rem; }
    h1 { line-height: 1.3 !important; padding-top: 0.5rem; }
    :root { --risk-low: #1fa971; --risk-mid: #f0a500; --risk-high: #e53935; }
    .risk-card { background: var(--secondary-background-color); color: var(--text-color); border: 1px solid rgba(128,128,128,0.18); border-radius: 18px; padding: 1rem; margin-bottom: 0.9rem; box-shadow: 0 2px 10px rgba(0,0,0,0.06); }
    .risk-header { display: flex; align-items: flex-start; justify-content: space-between; gap: 0.75rem; margin-bottom: 0.65rem; }
    .risk-title { font-size: 1.12rem; font-weight: 800; line-height: 1.2; word-break: break-word; }
    .risk-badge { flex: 0 0 auto; padding: 0.28rem 0.65rem; border-radius: 999px; color: white; font-size: 0.8rem; font-weight: 800; white-space: nowrap; }
    .risk-percent { font-size: 1.9rem; font-weight: 900; line-height: 1; margin-bottom: 0.45rem; }
    .risk-bar { width: 100%; height: 12px; border-radius: 999px; overflow: hidden; background: rgba(127,127,127,0.18); margin-bottom: 0.65rem; }
    .risk-fill { height: 100%; border-radius: 999px; }
    .risk-summary { font-size: 0.95rem; line-height: 1.5; opacity: 0.95; }
    .risk-meta { margin-top: 0.65rem; padding-top: 0.65rem; border-top: 1px solid rgba(127,127,127,0.12); font-size: 0.92rem; line-height: 1.45; }
    .muted { color: var(--text-color); opacity: 0.8; font-size: 0.92rem; }
    .badge { display: inline-block; padding: 0.25rem 0.6rem; border-radius: 999px; font-weight: 700; font-size: 0.85rem; margin-left: 0.35rem; }
    @media (max-width: 768px) { h1 { font-size: 1.8rem !important; word-break: break-word; padding-top: 0.75rem; } h2 { font-size: 1.45rem !important; } h3 { font-size: 1.22rem !important; } .block-container { padding-top: 1.5rem !important; } .risk-title { font-size: 1.02rem; } .risk-percent { font-size: 1.7rem; } }
</style>
""",
    unsafe_allow_html=True,
)

# ======================== ПЕРЕВОДЫ И СПРАВОЧНИКИ ========================
feature_names_ru = {
    "Age": "Возраст", "Gender": "Пол", "Polyuria": "Учащенное мочеиспускание (полиурия)",
    "Polydipsia": "Чрезмерная жажда (полидипсия)", "sudden weight loss": "Резкая потеря веса",
    "weakness": "Слабость", "Polyphagia": "Повышенный аппетит (полифагия)",
    "Genital thrush": "Генитальные инфекции (молочница)", "visual blurring": "Затуманивание зрения",
    "Itching": "Зуд", "Irritability": "Раздражительность", "delayed healing": "Медленное заживление ран",
    "partial paresis": "Частичный парез", "muscle stiffness": "Мышечная скованность",
    "Alopecia": "Выпадение волос (алопеция)", "Obesity": "Ожирение",
    "cold intolerance": "Непереносимость холода", "heat intolerance": "Непереносимость жары",
    "constipation": "Запоры", "diarrhea": "Диарея", "palpitations": "Сердцебиение",
    "tremor": "Тремор", "dry skin": "Сухость кожи", "fatigue": "Утомляемость",
    "anxiety": "Тревожность", "neck swelling": "Увеличение / отек в области шеи",
    "irregular periods": "Нерегулярный менструальный цикл", "acne": "Акне",
    "hirsutism": "Избыточный рост волос по мужскому типу", "infertility": "Бесплодие / трудности с зачатием",
    "postmenopausal": "Постменопауза", "prior fracture": "Перенесенный перелом",
    "glucocorticoids": "Длительный прием глюкокортикоидов", "low activity": "Низкая физическая активность",
    "facial fullness": "Округлое (луноподобное) лицо", "purple striae": "Фиолетовые растяжки",
    "easy bruising": "Лёгкое появление синяков", "proximal weakness": "Проксимальная мышечная слабость",
    "centripetal obesity": "Центральное ожирение", "hypertension": "Повышенное артериальное давление",
    "depression": "Депрессивное настроение", "hyperpigmentation": "Гиперпигментация кожи",
    "salt craving": "Тяга к солёному", "orthostatic dizziness": "Головокружение при вставании",
    "nausea": "Тошнота", "vomiting": "Рвота", "weight loss": "Похудение",
    "low blood pressure": "Низкое артериальное давление", "autoimmune history": "Аутоиммунные заболевания в анамнезе",
    "kidney stones": "Почечные камни", "bone pain": "Боли в костях", "abdominal pain": "Боль в животе",
    "frequent urination": "Частое мочеиспускание", "thirst": "Жажда", "muscle weakness": "Мышечная слабость",
    "mental fog": "Затуманенность мышления",
}

def badge(level: str) -> str:
    colors = {"Низкая": "#1f8b4c", "Лёгкая": "#1f8b4c", "Легкая": "#1f8b4c",
              "Умеренная": "#c77700", "Средняя": "#c77700", "Выраженная": "#c62828",
              "Тяжёлая": "#c62828", "Тяжелая": "#c62828", "Низкий": "#1f8b4c",
              "Умеренный": "#c77700", "Высокий": "#c62828", "Не оценен": "#666666"}
    color = colors.get(level, "#666666")
    return f"<span class='badge' style='background:{color}; color:white;'>{level}</span>"

def clamp(x, lo=0.0, hi=100.0): return float(max(lo, min(hi, x)))
def yes(val: bool) -> int: return 1 if val else 0

def risk_level(score: float, low: float = 30.0, high: float = 60.0):
    if score < low: return "Низкий"
    if score < high: return "Умеренный"
    return "Высокий"

def advice_by_level(level: str, low_msg: str, mid_msg: str, high_msg: str) -> str:
    return low_msg if level == "Низкий" else (mid_msg if level == "Умеренный" else high_msg)

def score_to_text(score: float) -> str: return f"{clamp(score):.1f}%"
def summarize_flags(flags): return "Явно выраженных групп риска по анкете не выделено." if not flags else " / ".join(flags)

def theme_risk_color(score: float) -> str:
    if score < 30: return "var(--risk-low)"
    if score < 60: return "var(--risk-mid)"
    return "var(--risk-high)"

def render_risk_card(title, score, stage_text, advice, confidence_text="—", posterior_text="—", drivers=None, signals=None):
    drivers = [d for d in (drivers or []) if d]
    signals = [s for s in (signals or []) if s]
    color = theme_risk_color(float(score))
    clipped_advice = advice if advice else ""
    percent_width = max(0.0, min(100.0, float(score)))
    st.markdown(f"""
<div class="risk-card">
  <div class="risk-header">
    <div class="risk-title">{title}</div>
    <div class="risk-badge" style="background:{color};">{stage_text}</div>
  </div>
  <div class="risk-percent">{float(score):.1f}%</div>
  <div class="risk-bar"><div class="risk-fill" style="width:{percent_width}%; background:{color};"></div></div>
  <div class="risk-summary">{clipped_advice}</div>
</div>""", unsafe_allow_html=True)
    with st.expander("Подробнее", expanded=False):
        st.write(f"**Уверенность:** {confidence_text}")
        st.write(f"**Апостериорная вероятность:** {posterior_text}")
        if drivers:
            st.write("**Основные драйверы:** " + ", ".join(drivers))
        if signals:
            st.write("**Отмеченные признаки:** " + ", ".join(signals))
        else:
            st.write("**Отмеченные признаки:** Нет выраженных симптомов по этому блоку.")

# ======================== ЗАГРУЗКА МОДЕЛЕЙ ========================
@st.cache_resource
def load_model():
    try: return joblib.load("diabetes_rf_model.pkl")
    except: return None

@st.cache_resource
def load_optional_model(path):
    try: return joblib.load(path)
    except: return None

model = load_model()

# Раздельные модели щитовидной железы
hypothyroid_model = load_optional_model("hypothyroid_ml_model.pkl")
hyperthyroid_model = load_optional_model("hyperthyroid_ml_model.pkl")

metabolic_model = load_optional_model("metabolic_ml_model.pkl")
pcos_model = load_optional_model("pcos_ml_model.pkl")
network_model = load_optional_model("endo_network_ml_model.pkl")

# Scaler-ы (если есть)
@st.cache_resource
def load_scaler(path):
    try: return joblib.load(path)
    except: return None

scaler_metabolic = load_scaler("scaler_metabolic.pkl")
scaler_thyroid = load_scaler("scaler_thyroid.pkl")
scaler_pcos = load_scaler("scaler_pcos.pkl")

def safe_positive_probability(model, row_df, positive_class=1):
    """Возвращает вероятность положительного класса в процентах или None."""
    if model is None or not hasattr(model, "predict_proba"): return None
    try:
        proba = model.predict_proba(row_df)[0]
        classes = list(getattr(model, "classes_", []))
        if positive_class in classes:
            pos_idx = classes.index(positive_class)
        else:
            pos_idx = 1 if len(proba) > 1 else 0
        return float(proba[pos_idx]) * 100.0
    except Exception:
        return None

# ======================== ФУНКЦИИ ПРИЗНАКОВ ========================
def activity_to_code(activity_level):
    return {"Низкая": 0, "Средняя": 1, "Высокая": 2}.get(activity_level, 1)

def make_metabolic_features(age, gender, bmi, waist_cm, activity_level, sleep_hours, fasting_glucose, hba1c, diabetes_symptom_values):
    symptom_burden = int(sum(1 for v in diabetes_symptom_values.values() if v))
    return pd.DataFrame([{
        "age": age, "gender": gender, "bmi": float(round(bmi,2)),
        "waist_cm": float(round(waist_cm,2)), "sleep_hours": float(round(sleep_hours,2)),
        "activity_code": activity_to_code(activity_level),
        "fasting_glucose": float(fasting_glucose) if fasting_glucose and fasting_glucose > 0 else 0.0,
        "hba1c": float(hba1c) if hba1c and hba1c > 0 else 0.0,
        "symptom_burden": symptom_burden,
        "obesity_flag": int(bmi >= 30),
        "central_obesity_flag": int(waist_cm >= (88 if gender == 1 else 94)),
        "sleep_short_flag": int(sleep_hours < 7),
    }])

def make_thyroid_features(age, gender, thyroid_values, tsh_value, ft4_value):
    return pd.DataFrame([{
        "age": age, "gender": gender,
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
        "age": age, "gender": gender, "bmi": float(round(bmi,2)),
        "waist_cm": float(round(waist_cm,2)), "sleep_hours": float(round(sleep_hours,2)),
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
        "diabetes_score": float(diabetes_score), "ir_score": float(ir_score),
        "hypothyroid_score": float(hypo_score), "hyperthyroid_score": float(hyper_score),
        "pcos_score": 0.0 if pcos_score is None else float(pcos_score),
        "bone_score": float(bone_score), "obesity_score": float(obesity_score),
        "metabolic_score": float(metabolic_score),
        "age": age, "gender": gender, "bmi": float(round(bmi,2)),
        "cross_axis_burden": float(np.nanmean([diabetes_score, ir_score, hypo_score, hyper_score,
                                               0.0 if pcos_score is None else pcos_score, bone_score])),
    }])

def ml_or_fallback_score(model, row_df, fallback_score):
    ml_score = safe_positive_probability(model, row_df)
    return ml_score if ml_score is not None else fallback_score


def ml_or_fallback_score(model, row_df, fallback_score, scaler=None):
    """Возвращает ML-вероятность или fallback-оценку."""
    if model is None:
        return fallback_score
    
    try:
        # Заменяем NaN на 0 (простая импутация)
        row_df_clean = row_df.fillna(0)
        
        # Применяем масштабирование, если передан scaler
        if scaler is not None:
            row_scaled = scaler.transform(row_df_clean)
            row_scaled_df = pd.DataFrame(row_scaled, columns=row_df_clean.columns)
        else:
            row_scaled_df = row_df_clean
        
        # Получаем вероятности
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(row_scaled_df)[0]
            # Определяем индекс положительного класса
            classes = list(getattr(model, "classes_", []))
            positive_class = 1  # по умолчанию
            if positive_class in classes:
                pos_idx = classes.index(positive_class)
            else:
                pos_idx = 1 if len(proba) > 1 else 0
            return float(proba[pos_idx]) * 100.0
        else:
            return fallback_score
            
    except Exception as e:
        # В случае ошибки используем fallback
        return fallback_score

# ======================== БАЗОВЫЕ ПРИЗНАКИ ДИАБЕТА ========================
expected_features = ["Age", "Gender", "Polyuria", "Polydipsia", "sudden weight loss",
                     "weakness", "Polyphagia", "Genital thrush", "visual blurring",
                     "Itching", "Irritability", "delayed healing", "partial paresis",
                     "muscle stiffness", "Alopecia", "Obesity"]

# ======================== СИМПТОМЫ ДЛЯ БЛОКОВ ========================
thyroid_symptoms = ["cold intolerance", "heat intolerance", "constipation", "diarrhea",
                    "palpitations", "tremor", "dry skin", "fatigue", "anxiety",
                    "neck swelling", "Alopecia", "weakness"]
pcos_symptoms = ["irregular periods", "acne", "hirsutism", "infertility", "Obesity", "Alopecia", "Polyphagia"]
bone_risk_features = ["postmenopausal", "prior fracture", "glucocorticoids", "low activity", "dry skin", "fatigue"]
cushing_symptoms = ["facial fullness", "purple striae", "easy bruising", "proximal weakness",
                    "hypertension", "centripetal obesity", "depression"]
addison_symptoms = ["hyperpigmentation", "salt craving", "orthostatic dizziness", "nausea",
                    "vomiting", "weight loss", "low blood pressure", "autoimmune history"]
hyperparathyroidism_symptoms = ["kidney stones", "bone pain", "constipation", "abdominal pain",
                                "depression", "muscle weakness", "frequent urination", "thirst", "fatigue"]

cushing_labels = {"facial fullness": "Округлое (луноподобное) лицо", "purple striae": "Фиолетовые растяжки",
                  "easy bruising": "Лёгкое появление синяков", "proximal weakness": "Проксимальная мышечная слабость",
                  "hypertension": "Повышенное артериальное давление", "centripetal obesity": "Центральное ожирение",
                  "depression": "Депрессивное настроение"}
addison_labels = {"hyperpigmentation": "Гиперпигментация кожи", "salt craving": "Тяга к солёному",
                  "orthostatic dizziness": "Головокружение при вставании", "nausea": "Тошнота",
                  "vomiting": "Рвота", "weight loss": "Похудение", "low blood pressure": "Низкое артериальное давление",
                  "autoimmune history": "Аутоиммунные заболевания в анамнезе"}
hyperpara_labels = {"kidney stones": "Почечные камни", "bone pain": "Боли в костях", "constipation": "Запоры",
                    "abdominal pain": "Боль в животе", "depression": "Депрессивное настроение",
                    "muscle weakness": "Мышечная слабость", "frequent urination": "Частое мочеиспускание",
                    "thirst": "Жажда", "fatigue": "Утомляемость"}

DISEASE_PRIORS = {
    "Диабет": 0.12, "Инсулинорезистентность / метаболический синдром": 0.20,
    "Щитовидная железа: гипофункция": 0.08, "Щитовидная железа: гиперфункция": 0.03,
    "PCOS": 0.10, "Эндокринная сеть": 0.15, "Костная ткань / остеопения": 0.15,
    "Синдром Кушинга": 0.01, "Болезнь Аддисона": 0.005, "Первичный гиперпаратиреоз": 0.01,
}

def count_positive_flags(flags: dict) -> int: return int(sum(1 for v in flags.values() if bool(v)))
def severity_label(score: float | None) -> str:
    if score is None: return "Не оценен"
    if score < 20: return "Низкая"
    if score < 40: return "Лёгкая"
    if score < 60: return "Умеренная"
    if score < 80: return "Выраженная"
    return "Тяжёлая"

def evidence_confidence(score, symptom_count=0, lab_count=0, red_flag_count=0, family_history=False):
    conf = 30.0 + 0.35 * clamp(score) + 4.0 * min(symptom_count, 6) + 6.0 * min(lab_count, 4) + 7.0 * red_flag_count
    if family_history: conf += 4.0
    return clamp(conf, 0.0, 100.0)

def bayes_like_probability(score, prior=0.05):
    prior = float(min(max(prior, 1e-4), 0.9999))
    prior_logit = math.log(prior / (1.0 - prior))
    evidence_logit = (clamp(score) - 50.0) / 12.0
    posterior = 1.0 / (1.0 + math.exp(-(prior_logit + evidence_logit)))
    return clamp(100.0 * posterior, 0.0, 100.0)

def assess_risk(score, disease_name, symptom_count=0, lab_count=0, red_flag_count=0, family_history=False):
    if score is None: return {"stage": "Не оценен", "confidence": None, "posterior": None}
    return {
        "stage": severity_label(score),
        "confidence": evidence_confidence(score, symptom_count, lab_count, red_flag_count, family_history),
        "posterior": bayes_like_probability(score, DISEASE_PRIORS.get(disease_name, 0.05)),
    }

# ======================== ФУНКЦИИ ПРОКСИ (исправлены) ========================
def diabetes_age_modifier(age):
    if age < 35: return 1.00
    elif age < 45: return 1.05
    elif age < 55: return 1.10
    elif age < 65: return 1.15
    else: return 1.20

def diabetes_probability_from_model(age, gender, symptom_values, family_history_diabetes):
    input_data = [age, gender]
    for feature in expected_features[2:]:
        input_data.append(1 if symptom_values.get(feature, False) else 0)
    input_df = pd.DataFrame([input_data], columns=expected_features)
    if model is not None and hasattr(model, "predict_proba"):
        try:
            probability = safe_positive_probability(model, input_df)
            if probability is not None:
                if family_history_diabetes: probability = clamp(probability + 10.0)
                return probability, int(model.predict(input_df)[0]), None
        except Exception as e:
            return None, None, f"Не удалось использовать модель: {e}"
    # Fallback
    score = 5.0
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
    if family_history_diabetes: score += 10
    score = clamp(score, 0, 99)
    return score, (1 if score >= 50 else 0), None

def obesity_proxy(bmi, waist_cm, activity_level, sleep_hours):
    score = 0.0
    if bmi >= 35: score += 35
    elif bmi >= 30: score += 28
    elif bmi >= 27: score += 20
    elif bmi >= 25: score += 12
    if waist_cm:
        if waist_cm >= 102: score += 20
        elif waist_cm >= 94: score += 14
        elif waist_cm >= 88: score += 10
    activity_map = {"Высокая": 0, "Средняя": 6, "Низкая": 12}
    score += activity_map.get(activity_level, 0)
    if sleep_hours < 6: score += 8
    elif sleep_hours < 7: score += 4
    return clamp(score)

def insulin_resistance_proxy(age, bmi, waist_cm, activity_level, sleep_hours, diabetes_symptom_values, family_history_diabetes):
    score = obesity_proxy(bmi, waist_cm, activity_level, sleep_hours)
    if age >= 45: score += 8
    elif age >= 35: score += 5
    score += 10 * yes(diabetes_symptom_values.get("Polyuria"))
    score += 10 * yes(diabetes_symptom_values.get("Polydipsia"))
    score += 8 * yes(diabetes_symptom_values.get("Polyphagia"))
    score += 6 * yes(diabetes_symptom_values.get("Obesity"))
    score += 6 * yes(diabetes_symptom_values.get("sudden weight loss"))
    score += 6 * yes(diabetes_symptom_values.get("weakness"))
    if family_history_diabetes: score += 8
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
    if age >= 50: score += 5
    if tsh_value and tsh_value > 0:
        if tsh_value > 4.5: score += min(25, (tsh_value - 4.5) * 8)
        elif tsh_value < 0.4: score -= 8
    if ft4_value and ft4_value > 0:
        if ft4_value < 0.8: score += 10
    if family_history_thyroid: score += 8
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
    if age < 50: score += 3
    if tsh_value and tsh_value > 0:
        if tsh_value < 0.4: score += min(25, (0.4 - tsh_value) * 20)
        elif tsh_value > 4.5: score -= 8
    if ft4_value and ft4_value > 0:
        if ft4_value > 1.8: score += 10
    if family_history_thyroid: score += 8
    return clamp(score)

def pcos_proxy(age, sex, pcos_values, bmi, insulin_resistance_score, fasting_glucose, hba1c):
    if sex != 1: return None
    score = 0.0
    score += 18 * yes(pcos_values.get("irregular periods"))
    score += 12 * yes(pcos_values.get("acne"))
    score += 14 * yes(pcos_values.get("hirsutism"))
    score += 10 * yes(pcos_values.get("infertility"))
    score += 8 * yes(pcos_values.get("Alopecia"))
    score += 8 * yes(pcos_values.get("Obesity"))
    score += min(18, insulin_resistance_score * 0.18)
    if age <= 35: score += 4
    if bmi >= 30: score += 6
    if fasting_glucose and fasting_glucose > 0 and fasting_glucose >= 100: score += 6
    if hba1c and hba1c > 0 and hba1c >= 5.7: score += 6
    return clamp(score)

def osteoporosis_proxy(age, sex, bone_values, bmi, family_history_osteoporosis):
    score = 0.0
    score += 15 * yes(bone_values.get("postmenopausal"))
    score += 14 * yes(bone_values.get("prior fracture"))
    score += 12 * yes(bone_values.get("glucocorticoids"))
    score += 10 * yes(bone_values.get("low activity"))
    if bmi and bmi > 0:
        if bmi < 18.5: score += 16
        elif bmi < 20: score += 10
        elif bmi < 22: score += 4
    if sex == 1: score += 4
    if age >= 65: score += 10
    elif age >= 50: score += 6
    if family_history_osteoporosis: score += 10
    return clamp(score)

def metabolic_syndrome_proxy(age, sex, bmi, waist_cm, activity_level, sleep_hours, fasting_glucose, hba1c, insulin_resistance_score, family_history_diabetes):
    score = 0.0
    score += obesity_proxy(bmi, waist_cm, activity_level, sleep_hours)  # исправлено: передаём sleep_hours
    score += min(20, insulin_resistance_score * 0.18)
    if age >= 45: score += 8
    elif age >= 35: score += 4
    if fasting_glucose and fasting_glucose > 0:
        if fasting_glucose >= 100: score += 10
        if fasting_glucose >= 126: score += 18
    if hba1c and hba1c > 0:
        if hba1c >= 5.7: score += 8
        if hba1c >= 6.5: score += 16
    if bmi >= 30: score += 6
    if family_history_diabetes: score += 8
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
    if glucocorticoids: score += 18
    if bmi >= 30: score += 6
    if age >= 40: score += 4
    if fasting_glucose and fasting_glucose >= 100: score += 4
    if hba1c and hba1c >= 5.7: score += 4
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
    if age < 50: score += 3
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
        if serum_calcium >= 10.5: score += min(28, (serum_calcium - 10.5) * 8)
        elif serum_calcium >= 10.0: score += 8
    if age >= 50: score += 4
    return clamp(score)

# ======================== ФУНКЦИИ СВЯЗЕЙ И РЕКОМЕНДАЦИЙ ========================
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
    if fasting_glucose and fasting_glucose >= 126: flags.append("Глюкоза натощак в диагностическом диапазоне диабета.")
    if hba1c and hba1c >= 6.5: flags.append("HbA1c в диагностическом диапазоне диабета.")
    if tsh_value and tsh_value >= 10: flags.append("ТТГ выше 10 мМЕ/л — требуется очная оценка щитовидной железы.")
    if tsh_value and tsh_value < 0.1 and ft4_value and ft4_value > 1.8: flags.append("Профиль совместим с выраженным тиреотоксикозом.")
    if cushing_score is not None and cushing_score >= 70: flags.append("Картина слабо совместима с гиперкортицизмом — нужна очная оценка.")
    if addison_score is not None and addison_score >= 70: flags.append("Картина требует исключения надпочечниковой недостаточности.")
    if hyperpara_score is not None and hyperpara_score >= 70: flags.append("Подозрение на гиперкальциемию / гиперпаратиреоз.")
    if serum_calcium and serum_calcium >= 11.0: flags.append("Значимо повышенный кальций — нужна перепроверка.")
    if bone_score >= 75: flags.append("Высокий костный риск — имеет смысл обсудить денситометрию.")
    return flags

# ======================== MF-DFA ========================
def extract_numeric_series(text: str):
    if not text or not text.strip(): return None
    tokens = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text.replace(",", " "))
    if not tokens: return None
    arr = np.asarray([float(tok) for tok in tokens], dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr if arr.size else None

def parse_series(text: str):
    arr = extract_numeric_series(text)
    if arr is None: return None
    return arr if arr.size >= 12 else None

def parse_uploaded_glucose_file(uploaded_file):
    if uploaded_file is None: return None
    raw = uploaded_file.getvalue()
    text = None
    for encoding in ("utf-8", "utf-8-sig", "cp1251", "latin1"):
        try:
            text = raw.decode(encoding)
            break
        except: continue
    if text is None: return None
    arr = extract_numeric_series(text)
    if arr is not None and arr.size >= 12: return arr
    try:
        df = pd.read_csv(io.StringIO(text), header=None, engine="python")
        numeric = pd.to_numeric(df.stack(), errors="coerce").dropna().to_numpy(dtype=float)
        numeric = numeric[np.isfinite(numeric)]
        if numeric.size >= 12: return numeric
    except: pass
    return None

def mfdfa(series, q_vals=None, min_scale=4, max_scale=None, scale_count=8):
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 12: return None
    x = x - np.mean(x)
    y = np.cumsum(x)
    if max_scale is None:
        max_scale = max(min_scale + 2, n // 3)
    max_scale = min(max_scale, max(min_scale + 2, n // 2))
    if max_scale <= min_scale: return None
    if max_scale - min_scale <= 10:
        scales = np.arange(min_scale, max_scale + 1, dtype=int)
    else:
        scales = np.unique(np.floor(np.logspace(np.log10(min_scale), np.log10(max_scale), scale_count)).astype(int))
    scales = scales[scales >= 4]
    scales = np.unique(scales)
    if scales.size < 3: return None
    if q_vals is None:
        q_vals = np.array([-2, -1, 0, 1, 2], dtype=float) if n < 24 else np.array([-4, -2, -1, 0, 1, 2, 4], dtype=float)
    Fq = np.full((len(q_vals), len(scales)), np.nan, dtype=float)
    for si, s in enumerate(scales):
        nseg = n // s
        if nseg < 2: continue
        rms = []
        for v in range(2 * nseg):
            if v < nseg: start = v * s
            else: start = n - (v - nseg + 1) * s
            segment = y[start:start + s]
            if segment.size < s: continue
            t = np.arange(s, dtype=float)
            coef = np.polyfit(t, segment, 1)
            trend = np.polyval(coef, t)
            resid = segment - trend
            rms.append(np.mean(resid ** 2))
        rms = np.asarray(rms, dtype=float)
        rms = rms[rms > 0]
        if rms.size == 0: continue
        for qi, q in enumerate(q_vals):
            if abs(q) < 1e-12: Fq[qi, si] = np.exp(0.5 * np.mean(np.log(rms)))
            else: Fq[qi, si] = (np.mean(rms ** (q / 2.0))) ** (1.0 / q)
    Hq = []
    for qi in range(len(q_vals)):
        valid = np.isfinite(Fq[qi]) & (Fq[qi] > 0)
        if valid.sum() < 3:
            Hq.append(np.nan)
            continue
        slope, _ = np.polyfit(np.log(scales[valid]), np.log(Fq[qi, valid]), 1)
        Hq.append(slope)
    Hq = np.asarray(Hq, dtype=float)
    if np.all(~np.isfinite(Hq)): return None
    width = float(np.nanmax(Hq) - np.nanmin(Hq))
    tau = q_vals * Hq - 1.0
    alpha = np.full_like(tau, np.nan, dtype=float)
    f_alpha = np.full_like(tau, np.nan, dtype=float)
    valid_tau = np.isfinite(tau) & np.isfinite(q_vals)
    if np.sum(valid_tau) >= 2:
        alpha_valid = np.gradient(tau[valid_tau], q_vals[valid_tau])
        alpha[valid_tau] = alpha_valid
        f_alpha[valid_tau] = q_vals[valid_tau] * alpha_valid - tau[valid_tau]
    return {"scales": scales, "q_vals": q_vals, "Hq": Hq, "Fq": Fq,
            "tau": tau, "alpha": alpha, "f_alpha": f_alpha,
            "width": width, "mean_h": float(np.nanmean(Hq))}

def mfdfa_interpretation(result):
    if result is None: return "Недостаточно данных для MF-DFA."
    width = result["width"]; mean_h = result["mean_h"]
    if width < 0.12: level = "Низкая мультифрактальность"; note = "Ряд относительно однородный и менее вариабельный."
    elif width < 0.25: level = "Умеренная мультифрактальность"; note = "Есть заметная неоднородность колебаний."
    else: level = "Высокая мультифрактальность"; note = "Колебания выраженно неоднородны; это может отражать нестабильную динамику."
    return f"{level}. Ширина спектра: {width:.3f}. Средний H(q): {mean_h:.3f}. {note}"

def plot_mfdfa_scaling(result):
    if result is None: return None
    scales = np.asarray(result.get("scales", []), dtype=float)
    q_vals = np.asarray(result.get("q_vals", []), dtype=float)
    Fq = np.asarray(result.get("Fq", []), dtype=float)
    if scales.size == 0 or q_vals.size == 0 or Fq.size == 0: return None
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for qi, q in enumerate(q_vals):
        y = Fq[qi] if Fq.ndim == 2 and qi < Fq.shape[0] else None
        if y is None: continue
        valid = np.isfinite(y) & (y > 0)
        if valid.sum() < 3: continue
        x = np.log10(scales[valid]); yy = np.log10(y[valid])
        ax.plot(x, yy, marker='o', linewidth=1.3, markersize=3.5, label=f"q={q:g}")
        if valid.sum() >= 2:
            coef = np.polyfit(x, yy, 1)
            xfit = np.linspace(x.min(), x.max(), 50)
            ax.plot(xfit, np.polyval(coef, xfit), linestyle='--', linewidth=1, alpha=0.6)
    ax.set_xlabel("log10(scale)"); ax.set_ylabel("log10(Fq)")
    ax.set_title("MF-DFA scaling plot"); ax.grid(True, alpha=0.25)
    if len(q_vals) <= 7: ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout(); return fig

def plot_mfdfa_spectrum(result):
    if result is None: return None
    alpha = np.asarray(result.get("alpha", []), dtype=float)
    f_alpha = np.asarray(result.get("f_alpha", []), dtype=float)
    valid = np.isfinite(alpha) & np.isfinite(f_alpha)
    if valid.sum() < 2: return None
    order = np.argsort(alpha[valid])
    alpha_sorted = alpha[valid][order]; f_sorted = f_alpha[valid][order]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(alpha_sorted, f_sorted, marker='o', linewidth=1.5, markersize=4)
    ax.set_xlabel("α"); ax.set_ylabel("f(α)"); ax.set_title("Multifractal spectrum")
    ax.grid(True, alpha=0.25); fig.tight_layout(); return fig

def interpret_complexity(width):
    if width >= 0.8: return "Высокая сложность / адаптивность"
    elif width >= 0.45: return "Умеренная сложность"
    else: return "Сниженная сложность, возможна потеря адаптивности"

def compare_to_reference(current_width):
    reference_width = 0.75
    delta = current_width - reference_width
    if delta > 0.15: status = "Сложность выше условной нормы"
    elif delta < -0.15: status = "Сложность ниже условной нормы"
    else: status = "Близко к условной норме"
    return {"reference": reference_width, "delta": delta, "status": status}

# ======================== ИНТЕРАКТИВНОЕ ОБУЧЕНИЕ ========================
def evaluate_model_metrics(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    return {
        'ROC-AUC': roc_auc_score(y_true, y_proba),
        'PR-AUC': average_precision_score(y_true, y_proba),
        'Sensitivity': recall_score(y_true, y_pred),
        'Specificity': specificity,
        'PPV': precision,
        'NPV': npv,
        'F1': f1_score(y_true, y_pred),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'Brier': brier_score_loss(y_true, y_proba),
        'Accuracy': accuracy_score(y_true, y_pred)
    }

def get_best_model_metric(metrics_df, y_data):
    """Автоматически выбирает метрику для определения лучшей модели."""
    # Проверяем баланс классов
    class_counts = pd.Series(y_data).value_counts()
    minority_ratio = class_counts.min() / class_counts.max()
    
    # Если дисбаланс сильный (меньше 30%), используем PR-AUC
    if minority_ratio < 0.3:
        return 'PR-AUC'
    # Если умеренный дисбаланс (30-50%), используем F1
    elif minority_ratio < 0.5:
        return 'F1'
    # Если данные сбалансированы, используем ROC-AUC
    else:
        return 'ROC-AUC'

def train_models_on_data(X, y, test_size=0.2):
    """Обучает несколько моделей, возвращает DataFrame с метриками и лучшую модель."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, eval_metric='logloss', verbosity=0),
        'LightGBM': LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1)
    }
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    metrics_list = []
    best_model = None
    best_auc = 0
    for name, model in models.items():
        try:
            model.fit(X_train_scaled, y_train)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)
            metrics = evaluate_model_metrics(y_test, y_pred, y_proba)
            metrics['Model'] = name
            metrics_list.append(metrics)
            if metrics['ROC-AUC'] > best_auc:
                best_auc = metrics['ROC-AUC']
                best_model = model
                best_scaler = scaler
        except Exception as e:
            st.warning(f"Модель {name} не обучилась: {e}")
    if metrics_list:
        df_metrics = pd.DataFrame(metrics_list).set_index('Model')
        return df_metrics, best_model, best_scaler
    return None, None, None

def calibrate_model(model, X_train, y_train, X_test, y_test):
    """Калибрует вероятности модели с помощью isotonic regression."""
    try:
        calibrated = CalibratedClassifierCV(model, method='isotonic', cv=3)
        calibrated.fit(X_train, y_train)
        return calibrated
    except:
        return model

def plot_calibration_curve(y_true, y_proba, n_bins=10):
    """Строит калибровочную кривую."""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Калибровочная кривая
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_proba, n_bins=n_bins
    )
    
    ax.plot(mean_predicted_value, fraction_of_positives, "s-", label="Модель")
    ax.plot([0, 1], [0, 1], "k--", label="Идеальная калибровка")
    
    ax.set_xlabel("Средняя предсказанная вероятность")
    ax.set_ylabel("Доля положительных исходов")
    ax.set_title("Калибровочная кривая")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

# ======================== ОСНОВНОЙ ИНТЕРФЕЙС ========================
st.title("🩺 Эндокринная медицинская карта")
st.markdown("""
Этот прототип объединяет несколько часто встречаемых эндокринных рисков в одном экране: диабет, инсулинорезистентность/метаболический синдром, нарушения щитовидной железы, PCOS и риск снижения костной массы.
Ниже выводится не просто процент, а связанная карта слабых мест и возможных пересечений между ними.

*Результат носит справочный характер и не заменяет очную консультацию врача.*
""")

# Сообщение о загруженных ML-моделях
ml_ready_note = []
if model is not None: ml_ready_note.append("диабет")
if metabolic_model is not None: ml_ready_note.append("метаболический риск")
if hypothyroid_model is not None: ml_ready_note.append("гипотиреоз")
if hyperthyroid_model is not None: ml_ready_note.append("гипертиреоз")
if pcos_model is not None: ml_ready_note.append("PCOS")
if network_model is not None: ml_ready_note.append("эндокринная сеть")
#if ml_ready_note:
#    st.success("ML-модели загружены для: " + ", ".join(ml_ready_note) + ".")
#else:
#    st.info("Для новых блоков используется безопасная клиническая логика; ML-модели можно подключить файлами .pkl без изменения интерфейса.")

# --- Вкладки ---
tab1, tab2 = st.tabs(["📋 Оценка рисков", "🔬 Лаборатория данных"])

with tab1:
    # ======================== ФОРМА ВВОДА ========================
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
        age = st.slider("Возраст (полных лет)", min_value=18, max_value=90, value=40)
    with col_gender:
        gender_input = st.radio("Пол", options=["Мужской", "Женский"])
    gender = 0 if gender_input == "Мужской" else 1

    with st.expander("🧬 Наследственность", expanded=st.session_state['expander_states']['heredity']):
        family_history_diabetes = st.checkbox("Наследственность по диабету 2 типа (родители, сиблинги)")
        family_history_thyroid = st.checkbox("Наследственность по заболеваниям щитовидной железы")
        family_history_osteoporosis = st.checkbox("Наследственность по остеопорозу")

    with st.expander("🧩 Базовые данные", expanded=st.session_state['expander_states']['basic_data']):
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
        with st.expander("Дополнительно (инсулин)"):
            insulin = st.number_input("Инсулин натощак, мкЕд/мл", min_value=0.0, value=0.0, step=0.1, help="0 = не указывать")

    with st.expander("🍬 Диабет и симптомы обмена", expanded=st.session_state['expander_states']['diabetes']):
        st.caption("Отметьте признаки, которые у вас наблюдаются.")
        diabetes_symptom_values = {}
        diabetes_features = expected_features[2:]
        render_symptom_checkboxes(diabetes_features, diabetes_symptom_values, "dm", feature_names_ru, columns=2)

    with st.expander("🦋 Щитовидная железа", expanded=st.session_state['expander_states']['thyroid']):
        thyroid_values = {}
        render_symptom_checkboxes(thyroid_symptoms, thyroid_values, "th", feature_names_ru, columns=2)

    with st.expander("♀️ Женский гормональный блок (PCOS)", expanded=st.session_state['expander_states']['pcos']):
        pcos_values = {}
        if gender == 1:
            st.caption("Этот блок активен только для женщин.")
            render_symptom_checkboxes(pcos_symptoms, pcos_values, "pcos", feature_names_ru, columns=2)
        else:
            st.info("PCOS-блок для мужчин не оценивается.")
            for feature in pcos_symptoms: pcos_values[feature] = False

    with st.expander("🦴 Костный риск / остеопения", expanded=st.session_state['expander_states']['bone']):
        bone_values = {}
        render_symptom_checkboxes(bone_risk_features, bone_values, "bone", feature_names_ru, columns=2)

    with st.expander("🩸 Дополнительные эндокринные блоки", expanded=st.session_state['expander_states']['additional']):
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

    with st.expander("🧪 Анализы (если уже есть)", expanded=st.session_state['expander_states']['labs']):
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

    with st.expander("🧮 Подключение ML модели (.pkl)", expanded=st.session_state['expander_states'].get('ml_model', False)):
        st.caption("Загрузите обученную ML модель в формате .pkl для расчёта предсказания.")
        
        # Инициализация состояния для ML модели
        if 'custom_ml_model' not in st.session_state:
            st.session_state['custom_ml_model'] = None
        if 'custom_ml_features' not in st.session_state:
            st.session_state['custom_ml_features'] = None
        if 'custom_ml_prediction' not in st.session_state:
            st.session_state['custom_ml_prediction'] = None
        if 'custom_ml_inputs' not in st.session_state:
            st.session_state['custom_ml_inputs'] = {}
        
        # Загрузка модели
        ml_model_file = st.file_uploader(
            "Выберите файл модели (.pkl)",
            type=["pkl", "joblib"],
            key="ml_model_uploader",
            help="Загрузите обученную модель sklearn в формате .pkl"
        )
        
        if ml_model_file is not None:
            try:
                # Загружаем модель
                import tempfile
                import os
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
                    tmp_file.write(ml_model_file.getvalue())
                    tmp_path = tmp_file.name
                
                loaded_model = joblib.load(tmp_path)
                os.unlink(tmp_path)
                
                # Проверяем, что модель имеет метод predict
                if not hasattr(loaded_model, 'predict'):
                    st.error("❌ Загруженный объект не является ML моделью (отсутствует метод predict).")
                else:
                    st.session_state['custom_ml_model'] = loaded_model
                    st.success(f"✅ Модель успешно загружена: {type(loaded_model).__name__}")
                    
                    # Пытаемся получить список признаков
                    if hasattr(loaded_model, 'feature_names_in_'):
                        st.session_state['custom_ml_features'] = list(loaded_model.feature_names_in_)
                        st.info(f"Обнаружено признаков: {len(loaded_model.feature_names_in_)}")
                    elif hasattr(loaded_model, 'n_features_in_'):
                        # Если признаки не сохранились, генерируем имена
                        st.session_state['custom_ml_features'] = [f"feature_{i}" for i in range(loaded_model.n_features_in_)]
                        st.warning(f"Имена признаков не сохранены в модели. Используются общие имена: feature_0, feature_1, ...")
                    else:
                        st.session_state['custom_ml_features'] = None
                        st.warning("Не удалось определить список признаков модели.")
                    
                    # Сбрасываем предыдущие предсказания
                    st.session_state['custom_ml_prediction'] = None
                    st.session_state['custom_ml_inputs'] = {}
                    
            except Exception as e:
                st.error(f"❌ Ошибка при загрузке модели: {e}")
        
        # Если модель загружена, показываем поля для ввода
        if st.session_state['custom_ml_model'] is not None:
            st.markdown("---")
            st.markdown("### 📝 Ввод значений для модели")
            
            # Определяем признаки
            features = st.session_state.get('custom_ml_features')
            
            if features is not None:
                # Создаем колонки для ввода
                cols_per_row = 2
                n_features = len(features)
                
                # Создаем поля ввода
                for i in range(0, n_features, cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        if i + j < n_features:
                            feature_name = features[i + j]
                            with cols[j]:
                                # Пытаемся определить тип признака
                                # По умолчанию используем number_input
                                default_value = 0.0
                                
                                # Проверяем, есть ли уже сохраненное значение
                                if feature_name in st.session_state['custom_ml_inputs']:
                                    default_value = st.session_state['custom_ml_inputs'][feature_name]
                                
                                # Создаем поле ввода
                                value = st.number_input(
                                    f"{feature_name}",
                                    value=float(default_value),
                                    step=0.1,
                                    format="%.3f",
                                    key=f"custom_ml_input_{feature_name}"
                                )
                                st.session_state['custom_ml_inputs'][feature_name] = value
                
                # Кнопка для предсказания
                st.markdown("---")
                col_predict, col_reset = st.columns([2, 1])
                
                with col_predict:
                    if st.button("🔮 Рассчитать предсказание", key="custom_ml_predict"):
                        try:
                            # Подготавливаем данные
                            input_data = []
                            for feature in features:
                                input_data.append(st.session_state['custom_ml_inputs'].get(feature, 0.0))
                            
                            # Создаем DataFrame
                            input_df = pd.DataFrame([input_data], columns=features)
                            
                            # Предсказание
                            model = st.session_state['custom_ml_model']
                            
                            # Получаем вероятности, если возможно
                            if hasattr(model, 'predict_proba'):
                                proba = model.predict_proba(input_df)[0]
                                
                                # Определяем классы
                                if hasattr(model, 'classes_'):
                                    classes = list(model.classes_)
                                else:
                                    classes = list(range(len(proba)))
                                
                                prediction = model.predict(input_df)[0]
                                
                                # Конвертируем numpy типы в Python типы
                                prediction = int(prediction) if isinstance(prediction, (np.integer, np.int64, np.int32)) else prediction
                                proba = [float(p) for p in proba]
                                classes = [str(c) if not isinstance(c, (int, float, str)) else c for c in classes]
                                
                                # Сохраняем результат
                                st.session_state['custom_ml_prediction'] = {
                                    'prediction': prediction,
                                    'probabilities': proba,
                                    'classes': classes
                                }
                            else:
                                # Если нет predict_proba, используем predict
                                prediction = model.predict(input_df)[0]
                                prediction = int(prediction) if isinstance(prediction, (np.integer, np.int64, np.int32)) else prediction
                                
                                st.session_state['custom_ml_prediction'] = {
                                    'prediction': prediction,
                                    'probabilities': None,
                                    'classes': None
                                }
                            
                            st.success("✅ Предсказание выполнено успешно!")
                            
                        except Exception as e:
                            st.error(f"❌ Ошибка при предсказании: {e}")
                            st.info("Проверьте, что все значения введены корректно.")
                
                with col_reset:
                    if st.button("🔄 Сбросить", key="custom_ml_reset"):
                        st.session_state['custom_ml_inputs'] = {}
                        st.session_state['custom_ml_prediction'] = None
                        st.rerun()
                
                # Показываем результат
                if st.session_state['custom_ml_prediction'] is not None:
                    st.markdown("---")
                    st.markdown("### 📊 Результат предсказания")
                    
                    result = st.session_state['custom_ml_prediction']
                    prediction = result['prediction']
                    probabilities = result.get('probabilities')
                    classes = result.get('classes')
                    
                    # Отображаем предсказание
                    st.metric("Предсказание", f"Класс: {prediction}")
                    
                    # Если есть вероятности, показываем их
                    if probabilities is not None:
                        st.markdown("#### Вероятности классов:")
                        
                        # Создаем DataFrame с вероятностями
                        prob_data = []
                        for i, prob in enumerate(probabilities):
                            class_name = classes[i] if classes is not None else i
                            prob_data.append({
                                "Класс": class_name,
                                "Вероятность": f"{prob*100:.2f}%"
                            })
                        
                        prob_df = pd.DataFrame(prob_data)
                        st.dataframe(prob_df, hide_index=True, use_container_width=True)
                        
                        # Визуализация вероятностей
                        fig_probs, ax_probs = plt.subplots(figsize=(8, 4))
                        colors = ['#4CAF50' if p < 0.5 else '#F44336' for p in probabilities]
                        ax_probs.bar([str(c) for c in classes], probabilities, color=colors, alpha=0.7)
                        ax_probs.set_ylabel('Вероятность')
                        ax_probs.set_xlabel('Класс')
                        ax_probs.set_title('Вероятности классов')
                        ax_probs.set_ylim(0, 1)
                        ax_probs.grid(True, alpha=0.3, axis='y')
                        
                        # Добавляем подписи значений
                        for i, p in enumerate(probabilities):
                            ax_probs.text(i, p + 0.02, f"{p:.2f}", ha='center', fontsize=10)
                        
                        st.pyplot(fig_probs, clear_figure=True, use_container_width=True)
                        plt.close(fig_probs)
                    
                    # Сохраняем результат в общий отчет
                    if 'custom_ml_features' in st.session_state and st.session_state['custom_ml_features'] is not None:
                        # Конвертируем все numpy типы в Python типы для JSON
                        inputs_serializable = {}
                        for key, value in st.session_state['custom_ml_inputs'].items():
                            if isinstance(value, (np.integer, np.int64, np.int32)):
                                inputs_serializable[key] = int(value)
                            elif isinstance(value, (np.floating, np.float64, np.float32)):
                                inputs_serializable[key] = float(value)
                            else:
                                inputs_serializable[key] = value
                        
                        ml_results = {
                            'model_type': type(st.session_state['custom_ml_model']).__name__,
                            'features': [str(f) for f in st.session_state['custom_ml_features']],
                            'inputs': inputs_serializable,
                            'prediction': prediction if isinstance(prediction, (int, float, str, bool)) else str(prediction),
                            'probabilities': [float(p) for p in probabilities] if probabilities is not None else None,
                            'classes': [str(c) if not isinstance(c, (int, float, str, bool)) else c for c in classes] if classes is not None else None
                        }
                        
                        # Добавляем в session_state для экспорта в отчет
                        st.session_state['custom_ml_results'] = ml_results
                        
                        # Кнопка для скачивания результатов
                        ml_results_json = json.dumps(ml_results, ensure_ascii=False, indent=2)
                        st.download_button(
                            "💾 Скачать результаты ML предсказания (JSON)",
                            data=ml_results_json,
                            file_name="ml_prediction_results.json",
                            mime="application/json",
                            key="download_ml_results"
                        )
            else:
                st.warning("⚠️ Не удалось определить признаки модели. Проверьте, что модель содержит информацию о признаках.")

    submitted = st.button("Собрать медицинскую карту", type="primary", use_container_width=True)

    # ======================== РЕЗУЛЬТАТЫ ========================
    # Если нажата кнопка или уже есть сохраненные результаты - выполняем расчеты
    if submitted or st.session_state.get('report_generated', False):
        # Сохраняем все входные данные в session_state для последующего использования
        if submitted:
            # Сохраняем входные данные
            st.session_state['input_data'] = {
                'age': age,
                'gender': gender,
                'gender_input': gender_input,
                'height_cm': height_cm,
                'weight_kg': weight_kg,
                'waist_cm': waist_cm,
                'sleep_hours': sleep_hours,
                'activity_level': activity_level,
                'bmi': bmi,
                'insulin': insulin,
                'family_history_diabetes': family_history_diabetes,
                'family_history_thyroid': family_history_thyroid,
                'family_history_osteoporosis': family_history_osteoporosis,
                'fasting_glucose': fasting_glucose,
                'hba1c': hba1c,
                'tsh_value': tsh_value,
                'ft4_value': ft4_value,
                'serum_calcium': serum_calcium,
                'diabetes_symptom_values': diabetes_symptom_values.copy(),
                'thyroid_values': thyroid_values.copy(),
                'pcos_values': pcos_values.copy(),
                'bone_values': bone_values.copy(),
                'cushing_values': cushing_values.copy(),
                'addison_values': addison_values.copy(),
                'hyperpara_values': hyperpara_values.copy(),
            }
        
        # Используем сохраненные данные, если они есть
        if st.session_state.get('input_data'):
            input_data = st.session_state['input_data']
            age = input_data['age']
            gender = input_data['gender']
            gender_input = input_data['gender_input']
            height_cm = input_data['height_cm']
            weight_kg = input_data['weight_kg']
            waist_cm = input_data['waist_cm']
            sleep_hours = input_data['sleep_hours']
            activity_level = input_data['activity_level']
            bmi = input_data['bmi']
            insulin = input_data['insulin']
            family_history_diabetes = input_data['family_history_diabetes']
            family_history_thyroid = input_data['family_history_thyroid']
            family_history_osteoporosis = input_data['family_history_osteoporosis']
            fasting_glucose = input_data['fasting_glucose']
            hba1c = input_data['hba1c']
            tsh_value = input_data['tsh_value']
            ft4_value = input_data['ft4_value']
            serum_calcium = input_data['serum_calcium']
            diabetes_symptom_values = input_data['diabetes_symptom_values']
            thyroid_values = input_data['thyroid_values']
            pcos_values = input_data['pcos_values']
            bone_values = input_data['bone_values']
            cushing_values = input_data['cushing_values']
            addison_values = input_data['addison_values']
            hyperpara_values = input_data['hyperpara_values']
        
        # ---- Вычисления ----
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
        metabolic_rule_score = metabolic_syndrome_proxy(age, gender, bmi, waist_cm, activity_level, sleep_hours, fasting_glucose, hba1c, ir_score, family_history_diabetes)
        cushing_rule_score = cushing_proxy(age, cushing_values, bmi, bone_values.get("glucocorticoids"), fasting_glucose, hba1c)
        addison_rule_score = addison_proxy(age, addison_values)
        hyperpara_rule_score = hyperparathyroid_proxy(age, hyperpara_values, serum_calcium)

        metabolic_ml_df = make_metabolic_features(age, gender, bmi, waist_cm, activity_level, sleep_hours, fasting_glucose, hba1c, diabetes_symptom_values)
        thyroid_ml_df = make_thyroid_features(age, gender, thyroid_values, tsh_value, ft4_value)
        pcos_ml_df = make_pcos_features(age, gender, bmi, waist_cm, activity_level, sleep_hours, pcos_values, fasting_glucose, hba1c, ir_score, tsh_value, ft4_value)

        metabolic_score = ml_or_fallback_score(metabolic_model, metabolic_ml_df, metabolic_rule_score, scaler_metabolic)
        hypothyroid_score = ml_or_fallback_score(hypothyroid_model, thyroid_ml_df, hypothyroid_rule_score, scaler_thyroid)
        hyperthyroid_score = ml_or_fallback_score(hyperthyroid_model, thyroid_ml_df, hyperthyroid_rule_score, scaler_thyroid)
        pcos_score = None if gender == 0 else ml_or_fallback_score(pcos_model, pcos_ml_df, pcos_rule_score, scaler_pcos)

        endo_network_df = make_network_features(
            diabetes_score, ir_score, hypothyroid_score, hyperthyroid_score,
            pcos_score, bone_score, obesity_score, metabolic_score, age, gender, bmi
        )
        network_score = safe_positive_probability(network_model, endo_network_df)
        if network_score is None:
            network_score = clamp(
                0.15 * diabetes_score + 0.15 * ir_score + 0.12 * hypothyroid_score +
                0.12 * hyperthyroid_score + 0.10 * (0.0 if pcos_score is None else pcos_score) +
                0.10 * bone_score + 0.10 * obesity_score + 0.08 * cushing_rule_score +
                0.05 * addison_rule_score + 0.03 * hyperpara_rule_score
            )
        
        # Сохраняем все результаты в session_state
        st.session_state['report_generated'] = True
        st.session_state['results'] = {
            'diabetes_score': diabetes_score,
            'ir_score': ir_score,
            'obesity_score': obesity_score,
            'hypothyroid_score': hypothyroid_score,
            'hyperthyroid_score': hyperthyroid_score,
            'pcos_score': pcos_score,
            'bone_score': bone_score,
            'metabolic_score': metabolic_score,
            'cushing_rule_score': cushing_rule_score,
            'addison_rule_score': addison_rule_score,
            'hyperpara_rule_score': hyperpara_rule_score,
            'network_score': network_score,
        }

        # ---- Уровни и советы ----
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

        diabetes_advice = advice_by_level(diabetes_level,
            "Риск диабета по текущим данным выглядит невысоким. Поддерживайте активность и базовый скрининг 1 раз в год.",
            "Есть признаки, которые стоит перепроверить лабораторно: глюкоза натощак, HbA1c, окружность талии, вес.",
            "Риск диабета высокий. Нужна очная оценка и лабораторное подтверждение в ближайшее время.")
        ir_advice = advice_by_level(ir_level,
            "Явных признаков выраженной инсулинорезистентности немного.",
            "Есть смысл усилить сон, активность и снизить висцеральный жир; стоит проверить HbA1c и липиды.",
            "Картина хорошо укладывается в инсулинорезистентность / метаболический синдром.")
        hypo_advice = advice_by_level(hypo_level,
            "Убедительных признаков гипофункции щитовидной железы немного.",
            "Стоит проверить ТТГ и свободный Т4, особенно если есть утомляемость или набор веса.",
            "Нужна очная оценка щитовидной железы и лабораторное подтверждение.")
        hyper_advice = advice_by_level(hyper_level,
            "Выраженных признаков тиреотоксикоза немного.",
            "При сердцебиении, дрожи и потере веса стоит проверить ТТГ и свободный Т4.",
            "Есть признаки, требующие проверки гиперфункции щитовидной железы.")
        pcos_advice = "PCOS не оценивается у мужчин." if pcos_score is None else advice_by_level(pcos_level,
            "Выраженных признаков PCOS немного.",
            "Есть признаки, совместимые с PCOS; полезна оценка цикла, андрогенных симптомов и метаболического статуса.",
            "Картина может соответствовать PCOS; рекомендована очная консультация гинеколога-эндокринолога.")
        network_advice = advice_by_level(network_level,
            "Эндокринная сеть сейчас выглядит относительно спокойной.",
            "Есть несколько взаимосвязанных зон, за которыми стоит наблюдать в динамике.",
            "Выраженная нагрузка на эндокринную сеть: стоит смотреть не только отдельные диагнозы, но и их сочетания.")
        bone_advice = advice_by_level(bone_level,
            "Выраженного костного риска по анкете немного.",
            "Стоит обратить внимание на витамин D, физическую нагрузку и причины снижения костной массы.",
            "Есть смысл обсудить оценку костной ткани и факторов остеопороза.")
        cushing_advice = advice_by_level(cushing_level,
            "Выраженных признаков гиперкортицизма немного.",
            "Стоит перепроверить давление, вес, стрии и факт приёма глюкокортикоидов.",
            "Картина может соответствовать гиперкортицизму; нужна очная оценка.")
        addison_advice = advice_by_level(addison_level,
            "Выраженных признаков надпочечниковой недостаточности немного.",
            "Стоит обратить внимание на давление, соль, тошноту и потерю веса.",
            "Нужно очно исключать болезнь Аддисона.")
        hyperpara_advice = advice_by_level(hyperpara_level,
            "Выраженных признаков гиперпаратиреоза немного.",
            "Есть смысл перепроверить кальций и симптомы со стороны костей / почек.",
            "Нужна очная оценка на гиперпаратиреоз и гиперкальциемию.")

        # ---- Оценки с байесовскими корректировками ----
        diabetes_assessment = assess_risk(diabetes_score, "Диабет", symptom_count=count_positive_flags(diabetes_symptom_values),
                                        lab_count=int(fasting_glucose > 0) + int(hba1c > 0),
                                        red_flag_count=1 if (fasting_glucose and fasting_glucose >= 126) or (hba1c and hba1c >= 6.5) else 0,
                                        family_history=family_history_diabetes)
        ir_assessment = assess_risk(ir_score, "Инсулинорезистентность / метаболический синдром",
                                    symptom_count=count_positive_flags(diabetes_symptom_values),
                                    lab_count=int(fasting_glucose > 0) + int(hba1c > 0),
                                    red_flag_count=1 if bmi >= 30 else 0,
                                    family_history=family_history_diabetes)
        hypo_assessment = assess_risk(hypothyroid_score, "Щитовидная железа: гипофункция",
                                    symptom_count=count_positive_flags(thyroid_values),
                                    lab_count=int(tsh_value > 0) + int(ft4_value > 0),
                                    red_flag_count=1 if tsh_value and tsh_value >= 10 else 0,
                                    family_history=family_history_thyroid)
        hyper_assessment = assess_risk(hyperthyroid_score, "Щитовидная железа: гиперфункция",
                                    symptom_count=count_positive_flags(thyroid_values),
                                    lab_count=int(tsh_value > 0) + int(ft4_value > 0),
                                    red_flag_count=1 if (tsh_value and tsh_value < 0.1 and ft4_value and ft4_value > 1.8) else 0,
                                    family_history=family_history_thyroid)
        pcos_assessment = None if pcos_score is None else assess_risk(pcos_score, "PCOS",
                                                                    symptom_count=count_positive_flags(pcos_values),
                                                                    lab_count=int(fasting_glucose > 0) + int(hba1c > 0),
                                                                    red_flag_count=0,
                                                                    family_history=False)
        bone_assessment = assess_risk(bone_score, "Костная ткань / остеопения",
                                    symptom_count=count_positive_flags(bone_values),
                                    lab_count=0,
                                    red_flag_count=1 if family_history_osteoporosis else 0,
                                    family_history=family_history_osteoporosis)
        cushing_assessment = assess_risk(cushing_rule_score, "Синдром Кушинга",
                                        symptom_count=count_positive_flags(cushing_values),
                                        lab_count=int(fasting_glucose > 0) + int(hba1c > 0),
                                        red_flag_count=1 if cushing_rule_score >= 70 else 0,
                                        family_history=False)
        addison_assessment = assess_risk(addison_rule_score, "Болезнь Аддисона",
                                        symptom_count=count_positive_flags(addison_values),
                                        lab_count=0,
                                        red_flag_count=1 if addison_rule_score >= 70 else 0,
                                        family_history=False)
        hyperpara_assessment = assess_risk(hyperpara_rule_score, "Первичный гиперпаратиреоз",
                                        symptom_count=count_positive_flags(hyperpara_values),
                                        lab_count=int(serum_calcium > 0),
                                        red_flag_count=1 if (serum_calcium and serum_calcium >= 11.0) else 0,
                                        family_history=False)

        connections = generate_connections(diabetes_score, ir_score, hypothyroid_score, hyperthyroid_score,
                                        pcos_score if pcos_score is not None else 0.0, bone_score, gender,
                                        cushing_rule_score, addison_rule_score, hyperpara_rule_score)
        next_steps = generate_next_steps(diabetes_score, ir_score, hypothyroid_score, hyperthyroid_score,
                                        pcos_score if pcos_score is not None else 0.0, bone_score, bmi,
                                        fasting_glucose, hba1c, cushing_rule_score, addison_rule_score,
                                        hyperpara_rule_score, serum_calcium)
        red_flags = generate_red_flags(diabetes_score, ir_score, hypothyroid_score, hyperthyroid_score,
                                    pcos_score, bone_score, cushing_rule_score, addison_rule_score,
                                    hyperpara_rule_score, fasting_glucose, hba1c, tsh_value, ft4_value, serum_calcium)

        # ---- ВЫВОД РЕЗУЛЬТАТОВ ----
        st.header("🗺️ Медицинская карта рисков")
        st.caption("Ниже — не диагноз, а структурированная карта вероятных слабых мест и взаимосвязей между ними.")

        m1, m2, m3 = st.columns(3)
        with m1: st.metric("Диабет", score_to_text(diabetes_score))
        with m2: st.metric("Инсулинорезистентность / метаболизм", score_to_text(ir_score))
        with m3: st.metric("Щитовидная ось", score_to_text(max(hypothyroid_score, hyperthyroid_score)))
        m4, m5, m6 = st.columns(3)
        with m4: st.metric("PCOS", "—" if pcos_score is None else score_to_text(pcos_score))
        with m5: st.metric("Костный риск", score_to_text(bone_score))
        with m6: st.metric("ИМТ", f"{bmi:.1f}")
        m7, m8, m9 = st.columns(3)
        with m7: st.metric("Кушинг", score_to_text(cushing_rule_score))
        with m8: st.metric("Аддисон", score_to_text(addison_rule_score))
        with m9: st.metric("Гиперпаратиреоз", score_to_text(hyperpara_rule_score))

        # Дополнительные метрики
        if insulin and fasting_glucose:
            homa_ir = (fasting_glucose * insulin) / 405
            st.metric("HOMA-IR", f"{homa_ir:.2f}",
                    help="Индекс инсулинорезистентности. Норма < 2.7")
        if hba1c:
            eag = 28.7 * hba1c - 46.7
            st.metric("eAG (расчётная средняя глюкоза)", f"{eag:.1f} мг/дл")

        st.progress(clamp(max(diabetes_score, ir_score, hypothyroid_score, hyperthyroid_score,
                            bone_score, cushing_rule_score, addison_rule_score, hyperpara_rule_score) / 100.0))

        if diabetes_fallback_error:
            st.warning(diabetes_fallback_error)

        if red_flags:
            st.error("Красные флаги: " + " | ".join(red_flags))

        # Зоны наибольшего внимания
        strong_points = []
        if diabetes_score >= 60: strong_points.append("углеводный обмен")
        if ir_score >= 60: strong_points.append("инсулинорезистентность")
        if max(hypothyroid_score, hyperthyroid_score) >= 60: strong_points.append("щитовидная железа")
        if pcos_score is not None and pcos_score >= 60: strong_points.append("PCOS")
        if bone_score >= 60: strong_points.append("костная ткань")
        if cushing_rule_score >= 60: strong_points.append("гиперкортицизм")
        if addison_rule_score >= 60: strong_points.append("надпочечники")
        if hyperpara_rule_score >= 60: strong_points.append("кальциевый обмен")
        if strong_points:
            st.error("Зоны наибольшего внимания: " + ", ".join(strong_points) + ".")
        else:
            st.success("Пока нет одной ярко выраженной зоны риска; полезен профилактический контроль и поддержка образа жизни.")

        # Карточки
        disease_cards = [
            {"name": "Диабет", "score": diabetes_score, "assessment": diabetes_assessment, "advice": diabetes_advice,
            "drivers": ["Симптомы диабета", "Возраст", "Вес / метаболическая нагрузка", "Наследственность" if family_history_diabetes else None],
            "signals": [feature_names_ru.get(k, k) for k, v in diabetes_symptom_values.items() if v]},
            {"name": "Инсулинорезистентность / метаболический синдром", "score": ir_score, "assessment": ir_assessment, "advice": ir_advice,
            "drivers": ["ИМТ", "Талия", "Сон и активность", "Симптомы углеводного обмена", "Наследственность" if family_history_diabetes else None],
            "signals": [f"ИМТ {bmi:.1f}", f"Талия {waist_cm:.0f} см", f"Сон {sleep_hours:.1f} ч/сутки"] + [feature_names_ru.get(k, k) for k, v in diabetes_symptom_values.items() if v][:4]},
            {"name": "Щитовидная железа: гипофункция", "score": hypothyroid_score, "assessment": hypo_assessment, "advice": hypo_advice,
            "drivers": ["Холод / запоры / сухость кожи", "Утомляемость", "ТТГ / свободный T4", "Наследственность" if family_history_thyroid else None],
            "signals": [feature_names_ru.get(k, k) for k, v in thyroid_values.items() if v and k in {"cold intolerance", "constipation", "fatigue", "dry skin", "Alopecia", "weakness", "neck swelling"}]},
            {"name": "Щитовидная железа: гиперфункция", "score": hyperthyroid_score, "assessment": hyper_assessment, "advice": hyper_advice,
            "drivers": ["Жара / сердцебиение / тремор", "Потеря веса", "ТТГ / свободный T4", "Наследственность" if family_history_thyroid else None],
            "signals": [feature_names_ru.get(k, k) for k, v in thyroid_values.items() if v and k in {"heat intolerance", "palpitations", "tremor", "anxiety", "diarrhea", "sudden weight loss", "neck swelling", "weakness"}]},
            {"name": "PCOS", "score": pcos_score, "assessment": pcos_assessment, "advice": pcos_advice,
            "drivers": ["Нерегулярный цикл", "Андрогенные симптомы", "Инсулинорезистентность"],
            "signals": [feature_names_ru.get(k, k) for k, v in pcos_values.items() if v]},
            {"name": "Эндокринная сеть", "score": network_score, "assessment": assess_risk(network_score, "Эндокринная сеть", symptom_count=0, lab_count=0, red_flag_count=len(red_flags), family_history=False),
            "advice": network_advice, "drivers": ["Совокупность всех осей", "Перекрёстные влияния", "Суммарная метаболическая нагрузка"],
            "signals": ["Интегральная оценка взаимосвязей"]},
            {"name": "Костная ткань / остеопения", "score": bone_score, "assessment": bone_assessment, "advice": bone_advice,
            "drivers": ["Возраст", "Переломы / стероиды", "Низкая активность / низкий ИМТ", "Наследственность" if family_history_osteoporosis else None],
            "signals": [feature_names_ru.get(k, k) for k, v in bone_values.items() if v]},
            {"name": "Синдром Кушинга", "score": cushing_rule_score, "assessment": cushing_assessment, "advice": cushing_advice,
            "drivers": ["Гиперкортицизм", "Глюкокортикоиды", "Центральное ожирение", "Гипертензия"],
            "signals": [cushing_labels.get(k, k) for k, v in cushing_values.items() if v]},
            {"name": "Болезнь Аддисона", "score": addison_rule_score, "assessment": addison_assessment, "advice": addison_advice,
            "drivers": ["Аутоиммунность", "Гипотензия", "Тяга к солёному", "Гиперпигментация"],
            "signals": [addison_labels.get(k, k) for k, v in addison_values.items() if v]},
            {"name": "Первичный гиперпаратиреоз", "score": hyperpara_rule_score, "assessment": hyperpara_assessment, "advice": hyperpara_advice,
            "drivers": ["Кальций", "Почки", "Кости", "Когнитивные/общие симптомы"],
            "signals": [hyperpara_labels.get(k, k) for k, v in hyperpara_values.items() if v]},
        ]

        for card in disease_cards:
            if card["score"] is None: continue
            assessment = card.get("assessment", {})
            drivers = [d for d in card.get("drivers", []) if d is not None]
            signals = card.get("signals", [])
            confidence_text = "—" if assessment.get("confidence") is None else f"{assessment['confidence']:.0f}%"
            posterior_text = "—" if assessment.get("posterior") is None else f"{assessment['posterior']:.0f}%"
            stage_text = assessment.get("stage", "Не оценен")
            render_risk_card(
                title=card["name"], score=card["score"], stage_text=stage_text,
                advice=card["advice"], confidence_text=confidence_text,
                posterior_text=posterior_text, drivers=drivers, signals=signals
            )

        # Радарная диаграмма
        categories = ['Диабет', 'ИР/МС', 'Гипотиреоз', 'Гипертиреоз', 'PCOS', 'Кости', 'Кушинг', 'Аддисон', 'Гиперпаратиреоз']
        values = [diabetes_score, ir_score, hypothyroid_score, hyperthyroid_score,
                pcos_score if pcos_score else 0, bone_score, cushing_rule_score,
                addison_rule_score, hyperpara_rule_score]
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=values, theta=categories, fill='toself', name='Риски'
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])), showlegend=False,
                                title="Радар рисков")
        st.plotly_chart(fig_radar, use_container_width=True)

        st.subheader("🔗 Как все связано между собой")
        if connections:
            for item in connections: st.write(f"- {item}")
        else:
            st.write("Явных взаимосвязей по анкете не выделено.")

        st.subheader("📌 Что стоит сделать дальше")
        for item in next_steps: st.write(f"- {item}")

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

        # Экспорт отчёта JSON
        report = {
            "date": str(pd.Timestamp.now()),
            "patient": {"age": age, "sex": gender_input, "bmi": round(bmi,1)},
            "scores": {
                "diabetes": diabetes_score, "insulin_resistance": ir_score,
                "hypothyroid": hypothyroid_score, "hyperthyroid": hyperthyroid_score,
                "pcos": pcos_score, "bone": bone_score, "cushing": cushing_rule_score,
                "addison": addison_rule_score, "hyperparathyroid": hyperpara_rule_score,
                "endocrine_network": network_score
            },
            "recommendations": next_steps,
            "red_flags": red_flags,
            "connections": connections
        }
        
        # Сохраняем отчет в session_state для предотвращения перезагрузки
        if 'report_json' not in st.session_state:
            st.session_state['report_json'] = json.dumps(report, ensure_ascii=False, indent=2)
        
        # Кнопка скачивания
        col_download, col_space = st.columns([1, 2])
        with col_download:
            st.download_button(
                "Скачать отчёт (JSON)",
                data=st.session_state['report_json'],
                file_name="endocrine_report.json",
                mime="application/json",
                key="download_report_button"
            )

        # Кнопка создания PDF отчета
        st.markdown("---")
        st.subheader("📄 Создание PDF отчета")
        st.caption("Создайте подробный PDF отчет со всеми данными пациента.")
        
        if st.button("📄 Создать PDF отчет", key="create_pdf_button"):
            try:
                # Импортируем необходимые библиотеки для создания PDF
                from reportlab.lib.pagesizes import A4
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib.units import inch, cm
                from reportlab.lib import colors
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
                from reportlab.lib.enums import TA_CENTER, TA_LEFT
                from reportlab.pdfbase import pdfmetrics
                from reportlab.pdfbase.ttfonts import TTFont
                import io
                import os
                
                # Регистрируем шрифты с поддержкой кириллицы
                # Сначала пробуем DejaVu шрифты (лучшая поддержка кириллицы)
                font_registered = False
                
                # Список возможных путей к шрифтам
                font_candidates = [
                    # DejaVu Sans (Windows)
                    ("DejaVuSans", "C:/Windows/Fonts/DejaVuSans.ttf"),
                    ("DejaVuSans-Bold", "C:/Windows/Fonts/DejaVuSans-Bold.ttf"),
                    # DejaVu Sans (Linux)
                    ("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                    ("DejaVuSans-Bold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
                    # DejaVu Sans (macOS)
                    ("DejaVuSans", "/Library/Fonts/DejaVuSans.ttf"),
                    ("DejaVuSans-Bold", "/Library/Fonts/DejaVuSans-Bold.ttf"),
                    # Arial (Windows)
                    ("Arial", "C:/Windows/Fonts/arial.ttf"),
                    ("Arial-Bold", "C:/Windows/Fonts/arialbd.ttf"),
                    # Times New Roman (Windows)
                    ("TimesNewRoman", "C:/Windows/Fonts/times.ttf"),
                    ("TimesNewRoman-Bold", "C:/Windows/Fonts/timesbd.ttf"),
                ]
                
                # Ищем и регистрируем обычный шрифт
                main_font = None
                bold_font = None
                
                # Сначала ищем пару DejaVu
                regular_candidates = []
                bold_candidates = []
                
                for font_name, font_path in font_candidates:
                    if os.path.exists(font_path):
                        if "Bold" in font_name or "bold" in font_name:
                            bold_candidates.append((font_name, font_path))
                        else:
                            regular_candidates.append((font_name, font_path))
                
                # Выбираем первую доступную пару
                if regular_candidates:
                    # Регистрируем обычный шрифт
                    regular_font_name, regular_font_path = regular_candidates[0]
                    try:
                        pdfmetrics.registerFont(TTFont(regular_font_name, regular_font_path))
                        main_font = regular_font_name
                        font_registered = True
                    except:
                        pass
                
                if bold_candidates:
                    # Регистрируем жирный шрифт
                    bold_font_name, bold_font_path = bold_candidates[0]
                    try:
                        pdfmetrics.registerFont(TTFont(bold_font_name, bold_font_path))
                        bold_font = bold_font_name
                    except:
                        # Если не удалось зарегистрировать жирный, используем обычный
                        bold_font = main_font
                
                # Если шрифты не найдены, пробуем скачать или использовать системные
                if not font_registered or main_font is None:
                    # Пробуем использовать matplotlib шрифты
                    try:
                        import matplotlib.font_manager as fm
                        
                        # Ищем шрифты с поддержкой кириллицы
                        font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
                        
                        for font_path in font_list:
                            try:
                                # Пытаемся определить, поддерживает ли шрифт кириллицу
                                font_name = os.path.basename(font_path).replace('.ttf', '')
                                
                                # Пробуем зарегистрировать
                                if any(cyrillic_font in font_name.lower() for cyrillic_font in ['dejavu', 'arial', 'times', 'liberation', 'ubuntu', 'noto']):
                                    try:
                                        pdfmetrics.registerFont(TTFont(font_name, font_path))
                                        if main_font is None:
                                            main_font = font_name
                                            bold_font = font_name  # Используем тот же шрифт для жирного
                                            font_registered = True
                                            break
                                    except:
                                        continue
                            except:
                                continue
                    except:
                        pass
                
                # Если все еще не нашли, используем стандартные шрифты
                if not font_registered or main_font is None:
                    main_font = "Helvetica"
                    bold_font = "Helvetica-Bold"
                    st.warning("⚠️ Не найдены шрифты с поддержкой кириллицы. Русский текст может отображаться некорректно.")
                else:
                    if bold_font is None:
                        bold_font = main_font
                
                # Создаем буфер для PDF
                pdf_buffer = io.BytesIO()
                
                # Создаем документ
                doc = SimpleDocTemplate(
                    pdf_buffer,
                    pagesize=A4,
                    rightMargin=72,
                    leftMargin=72,
                    topMargin=72,
                    bottomMargin=72
                )
                
                # Стили
                styles = getSampleStyleSheet()
                
                # Кастомные стили с поддержкой кириллицы
                title_style = ParagraphStyle(
                    'CustomTitle',
                    parent=styles['Heading1'],
                    fontName=bold_font,
                    fontSize=20,
                    textColor=colors.HexColor('#1a237e'),
                    spaceAfter=10,
                    alignment=TA_CENTER
                )
                
                section_style = ParagraphStyle(
                    'CustomSection',
                    parent=styles['Heading2'],
                    fontName=bold_font,
                    fontSize=16,
                    textColor=colors.HexColor('#1a237e'),
                    spaceAfter=8,
                    spaceBefore=16
                )
                
                body_style = ParagraphStyle(
                    'CustomBody',
                    parent=styles['Normal'],
                    fontName=main_font,
                    fontSize=10,
                    leading=14,
                    spaceAfter=4
                )
                
                # Собираем элементы для PDF
                elements = []
                
                # Заголовок
                elements.append(Paragraph("Эндокринная медицинская карта", title_style))
                elements.append(Paragraph(f"Дата создания: {pd.Timestamp.now().strftime('%d.%m.%Y %H:%M')}", body_style))
                elements.append(Spacer(1, 20))
                
                # Основные данные пациента
                elements.append(Paragraph("Основные данные пациента", section_style))
                
                patient_data = [
                    ["Показатель", "Значение"],
                    ["Возраст", f"{age} лет"],
                    ["Пол", gender_input],
                    ["Рост", f"{height_cm:.0f} см"],
                    ["Вес", f"{weight_kg:.1f} кг"],
                    ["ИМТ", f"{bmi:.1f}"],
                    ["Окружность талии", f"{waist_cm:.1f} см"],
                    ["Сон", f"{sleep_hours:.1f} часов/сутки"],
                    ["Физическая активность", activity_level],
                ]
                
                # Добавляем инсулин, если указан
                if insulin > 0:
                    patient_data.append(["Инсулин натощак", f"{insulin:.1f} мкЕд/мл"])
                
                # Создаем таблицу с данными пациента
                patient_table = Table(patient_data, colWidths=[6*cm, 8*cm])
                patient_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a237e')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), bold_font),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('FONTNAME', (0, 1), (-1, -1), main_font),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f5f5f5')),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                elements.append(patient_table)
                
                # Наследственность
                elements.append(Paragraph("Наследственность", section_style))
                heredity_data = []
                if family_history_diabetes:
                    heredity_data.append("• Диабет 2 типа (родители, сиблинги)")
                if family_history_thyroid:
                    heredity_data.append("• Заболевания щитовидной железы")
                if family_history_osteoporosis:
                    heredity_data.append("• Остеопороз")
                
                if heredity_data:
                    for item in heredity_data:
                        elements.append(Paragraph(item, body_style))
                else:
                    elements.append(Paragraph("Не отягощена", body_style))
                
                # Лабораторные показатели
                elements.append(Paragraph("Лабораторные показатели", section_style))
                
                lab_data = []
                if fasting_glucose > 0:
                    lab_data.append(["Глюкоза натощак", f"{fasting_glucose:.1f} мг/дл"])
                if hba1c > 0:
                    lab_data.append(["HbA1c", f"{hba1c:.1f} %"])
                if tsh_value > 0:
                    lab_data.append(["ТТГ", f"{tsh_value:.1f} мМЕ/л"])
                if ft4_value > 0:
                    lab_data.append(["Св. T4", f"{ft4_value:.1f} нг/дл"])
                if serum_calcium > 0:
                    lab_data.append(["Кальций общий", f"{serum_calcium:.1f} мг/дл"])
                
                if lab_data:
                    lab_table = Table([["Показатель", "Значение"]] + lab_data, colWidths=[6*cm, 8*cm])
                    lab_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a237e')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), bold_font),
                        ('FONTSIZE', (0, 0), (-1, 0), 10),
                        ('FONTNAME', (0, 1), (-1, -1), main_font),
                        ('FONTSIZE', (0, 1), (-1, -1), 9),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f5f5f5')),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ]))
                    elements.append(lab_table)
                else:
                    elements.append(Paragraph("Лабораторные данные не указаны", body_style))
                
                # Результаты оценки рисков
                elements.append(PageBreak())
                elements.append(Paragraph("Результаты оценки рисков", section_style))
                
                # Таблица с рисками
                risk_data = [
                    ["Показатель", "Значение", "Уровень риска"],
                    ["Диабет", score_to_text(diabetes_score), diabetes_level],
                    ["Инсулинорезистентность", score_to_text(ir_score), ir_level],
                    ["Щитовидная железа (гипо)", score_to_text(hypothyroid_score), hypo_level],
                    ["Щитовидная железа (гипер)", score_to_text(hyperthyroid_score), hyper_level],
                    ["PCOS", "—" if pcos_score is None else score_to_text(pcos_score), pcos_level],
                    ["Эндокринная сеть", score_to_text(network_score), network_level],
                    ["Костный риск", score_to_text(bone_score), bone_level],
                    ["Кушинг", score_to_text(cushing_rule_score), cushing_level],
                    ["Аддисон", score_to_text(addison_rule_score), addison_level],
                    ["Гиперпаратиреоз", score_to_text(hyperpara_rule_score), hyperpara_level],
                ]
                
                risk_table = Table(risk_data, colWidths=[5*cm, 3*cm, 6*cm])
                risk_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a237e')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), bold_font),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('FONTNAME', (0, 1), (-1, -1), main_font),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f5f5f5')),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                elements.append(risk_table)
                
                # Красные флаги
                elements.append(Paragraph("Красные флаги", section_style))
                if red_flags:
                    for flag in red_flags:
                        elements.append(Paragraph(f"• {flag}", body_style))
                else:
                    elements.append(Paragraph("Красные флаги не обнаружены", body_style))
                
                # Зоны наибольшего внимания
                elements.append(Paragraph("Зоны наибольшего внимания", section_style))
                if strong_points:
                    for point in strong_points:
                        elements.append(Paragraph(f"• {point}", body_style))
                else:
                    elements.append(Paragraph("Нет ярко выраженных зон риска", body_style))
                
                # Взаимосвязи
                elements.append(Paragraph("Взаимосвязи", section_style))
                if connections:
                    for connection in connections:
                        elements.append(Paragraph(f"• {connection}", body_style))
                else:
                    elements.append(Paragraph("Явных взаимосвязей не выделено", body_style))
                
                # Рекомендации
                elements.append(Paragraph("Рекомендации", section_style))
                for i, step in enumerate(next_steps, 1):
                    elements.append(Paragraph(f"{i}. {step}", body_style))
                
                # Отмеченные симптомы
                elements.append(Paragraph("Отмеченные симптомы", section_style))
                
                symptoms_data = []
                diabetes_symptoms = [feature_names_ru.get(k, k) for k, v in diabetes_symptom_values.items() if v]
                thyroid_symptoms_list = [feature_names_ru.get(k, k) for k, v in thyroid_values.items() if v]
                pcos_symptoms_list = [feature_names_ru.get(k, k) for k, v in pcos_values.items() if v] if gender == 1 else []
                bone_symptoms_list = [feature_names_ru.get(k, k) for k, v in bone_values.items() if v]
                
                if diabetes_symptoms:
                    elements.append(Paragraph("<b>Диабет и обмен:</b>", body_style))
                    for symptom in diabetes_symptoms:
                        elements.append(Paragraph(f"• {symptom}", body_style))
                
                if thyroid_symptoms_list:
                    elements.append(Paragraph("<b>Щитовидная железа:</b>", body_style))
                    for symptom in thyroid_symptoms_list:
                        elements.append(Paragraph(f"• {symptom}", body_style))
                
                if pcos_symptoms_list:
                    elements.append(Paragraph("<b>PCOS:</b>", body_style))
                    for symptom in pcos_symptoms_list:
                        elements.append(Paragraph(f"• {symptom}", body_style))
                
                if bone_symptoms_list:
                    elements.append(Paragraph("<b>Костный риск:</b>", body_style))
                    for symptom in bone_symptoms_list:
                        elements.append(Paragraph(f"• {symptom}", body_style))
                
                if not any([diabetes_symptoms, thyroid_symptoms_list, pcos_symptoms_list, bone_symptoms_list]):
                    elements.append(Paragraph("Симптомы не отмечены", body_style))
                
                # Дополнительные эндокринные блоки
                cushing_symptoms_list = [cushing_labels.get(k, k) for k, v in cushing_values.items() if v]
                addison_symptoms_list = [addison_labels.get(k, k) for k, v in addison_values.items() if v]
                hyperpara_symptoms_list = [hyperpara_labels.get(k, k) for k, v in hyperpara_values.items() if v]
                
                if cushing_symptoms_list or addison_symptoms_list or hyperpara_symptoms_list:
                    elements.append(Paragraph("Дополнительные эндокринные блоки:", body_style))
                    
                    if cushing_symptoms_list:
                        elements.append(Paragraph("<b>Синдром Кушинга:</b>", body_style))
                        for symptom in cushing_symptoms_list:
                            elements.append(Paragraph(f"• {symptom}", body_style))
                    
                    if addison_symptoms_list:
                        elements.append(Paragraph("<b>Болезнь Аддисона:</b>", body_style))
                        for symptom in addison_symptoms_list:
                            elements.append(Paragraph(f"• {symptom}", body_style))
                    
                    if hyperpara_symptoms_list:
                        elements.append(Paragraph("<b>Гиперпаратиреоз:</b>", body_style))
                        for symptom in hyperpara_symptoms_list:
                            elements.append(Paragraph(f"• {symptom}", body_style))
                
                # Добавляем ML модель, если была загружена
                if 'custom_ml_results' in st.session_state and st.session_state['custom_ml_results'] is not None:
                    elements.append(PageBreak())
                    elements.append(Paragraph("Результаты ML модели", section_style))
                    
                    ml_results = st.session_state['custom_ml_results']
                    elements.append(Paragraph(f"Тип модели: {ml_results['model_type']}", body_style))
                    
                    if ml_results['probabilities'] is not None:
                        ml_pred_data = [["Класс", "Вероятность"]]
                        for i, prob in enumerate(ml_results['probabilities']):
                            class_name = ml_results['classes'][i]
                            ml_pred_data.append([str(class_name), f"{prob*100:.2f}%"])
                        
                        ml_table = Table(ml_pred_data, colWidths=[6*cm, 8*cm])
                        ml_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a237e')),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                            ('FONTNAME', (0, 0), (-1, 0), bold_font),
                            ('FONTSIZE', (0, 0), (-1, 0), 10),
                            ('FONTNAME', (0, 1), (-1, -1), main_font),
                            ('FONTSIZE', (0, 1), (-1, -1), 9),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f5f5f5')),
                            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ]))
                        elements.append(ml_table)
                
                # Добавляем предупреждение
                elements.append(Spacer(1, 20))
                elements.append(Paragraph(
                    "<i>Данный отчет носит справочный характер и не заменяет очную консультацию врача. "
                    "Диагностические решения и назначения должен подтверждать квалифицированный специалист.</i>",
                    body_style
                ))
                
                # Создаем PDF
                doc.build(elements)
                
                # Получаем байты PDF
                pdf_bytes = pdf_buffer.getvalue()
                pdf_buffer.close()
                
                # Сохраняем в session_state
                st.session_state['pdf_report'] = pdf_bytes
                
                st.success("✅ PDF отчет успешно создан!")
                
                # Кнопка для скачивания PDF
                st.download_button(
                    "📥 Скачать PDF отчет",
                    data=pdf_bytes,
                    file_name=f"endocrine_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    key="download_pdf_button"
                )
                
            except ImportError:
                st.error("❌ Библиотека reportlab не установлена. Установите её с помощью: pip install reportlab")
            except Exception as e:
                st.error(f"❌ Ошибка при создании PDF: {e}")
                st.info("Попробуйте перезапустить приложение и повторить попытку.")




with tab2:
    # ======================== КОМПЛЕКСНЫЙ АНАЛИЗ ДАННЫХ ========================
    st.header("🔄 Комплексный анализ данных")
    st.caption("Дополнительные аналитические инструменты для изучения введённых данных.")

    # ======================== ЭКСПАНДЕР 1: ОБУЧЕНИЕ МОДЕЛЕЙ ========================
    with st.expander("🧮 Обучение моделей на ваших данных", expanded=st.session_state['expander_states']['training']):
        st.markdown("### Загрузите CSV-файл с данными для обучения")
        st.caption("Файл должен содержать целевую переменную (бинарную). Все признаки должны быть числовыми или категориальными (будут автоматически преобразованы).")
        train_file = st.file_uploader("Выберите CSV", type=["csv"], key="train_file")
        
        if train_file is not None:
            try:
                # Читаем с определением разделителя
                train_df = pd.read_csv(train_file, sep=None, engine='python')
                st.write("Первые строки данных:")
                st.dataframe(train_df.head())
                
                # Показываем информацию о типах данных
                st.write("Типы данных:")
                st.write(train_df.dtypes)
                
                target_col = st.selectbox("Выберите целевую переменную", train_df.columns)
                
                # Настройки обучения
                st.markdown("### ⚙️ Настройки обучения")
                
                col1, col2 = st.columns(2)
                with col1:
                    test_size = st.slider("Размер тестовой выборки (%)", 10, 40, 20, 5) / 100
                    random_state = st.number_input("Random State", 0, 1000, 42, 1)
                
                with col2:
                    balance_classes = st.checkbox("Балансировать классы", value=True, 
                                                help="Автоматически балансировать веса классов")
                    use_cv = st.checkbox("Использовать кросс-валидацию", value=False,
                                        help="Использовать k-fold кросс-валидацию для оценки")
                
                if use_cv:
                    cv_folds = st.slider("Количество фолдов", 3, 10, 5, 1)
                
                # Выбор моделей для обучения
                st.markdown("### Выберите модели для обучения")
                
                model_selection = {}
                col1, col2 = st.columns(2)
                
                with col1:
                    model_selection['logistic'] = st.checkbox("Logistic Regression", value=True)
                    model_selection['random_forest'] = st.checkbox("Random Forest", value=True)
                    model_selection['xgboost'] = st.checkbox("XGBoost", value=True)
                
                with col2:
                    model_selection['lightgbm'] = st.checkbox("LightGBM", value=True)
                    model_selection['catboost'] = st.checkbox("CatBoost", value=False)
                    model_selection['mlp'] = st.checkbox("Neural Network (MLP)", value=False)
                
                # Параметры моделей
                model_params = {}
                
                if model_selection['logistic']:
                    st.markdown("#### Параметры Logistic Regression")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        model_params['logistic_C'] = st.number_input("C (регуляризация)", 0.01, 10.0, 1.0, 0.1, key="lr_c")
                    with col2:
                        model_params['logistic_max_iter'] = st.number_input("Max Iterations", 100, 5000, 1000, 100, key="lr_iter")
                    with col3:
                        model_params['logistic_penalty'] = st.selectbox("Penalty", ['l2', 'l1', 'none'], key="lr_penalty")
                
                if model_selection['random_forest']:
                    st.markdown("#### Параметры Random Forest")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        model_params['rf_n_estimators'] = st.number_input("Number of Trees", 10, 500, 100, 10, key="rf_trees")
                    with col2:
                        model_params['rf_max_depth'] = st.number_input("Max Depth (0=None)", 0, 50, 0, 1, key="rf_depth")
                    with col3:
                        model_params['rf_min_samples_split'] = st.number_input("Min Samples Split", 2, 20, 2, 1, key="rf_split")
                
                if model_selection['xgboost']:
                    st.markdown("#### Параметры XGBoost")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        model_params['xgb_n_estimators'] = st.number_input("Number of Trees", 10, 500, 100, 10, key="xgb_trees")
                    with col2:
                        model_params['xgb_learning_rate'] = st.number_input("Learning Rate", 0.01, 0.5, 0.1, 0.01, key="xgb_lr")
                    with col3:
                        model_params['xgb_max_depth'] = st.number_input("Max Depth", 3, 15, 6, 1, key="xgb_depth")
                
                if model_selection['lightgbm']:
                    st.markdown("#### Параметры LightGBM")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        model_params['lgbm_n_estimators'] = st.number_input("Number of Trees", 10, 500, 100, 10, key="lgbm_trees")
                    with col2:
                        model_params['lgbm_learning_rate'] = st.number_input("Learning Rate", 0.01, 0.5, 0.1, 0.01, key="lgbm_lr")
                    with col3:
                        model_params['lgbm_num_leaves'] = st.number_input("Number of Leaves", 2, 100, 31, 1, key="lgbm_leaves")
                
                if model_selection['mlp']:
                    st.markdown("#### Параметры Neural Network (MLP)")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        model_params['mlp_hidden'] = st.text_input("Hidden Layers (comma-separated)", "100,50", key="mlp_hidden")
                    with col2:
                        model_params['mlp_max_iter'] = st.number_input("Max Iterations", 100, 5000, 1000, 100, key="mlp_iter")
                    with col3:
                        model_params['mlp_alpha'] = st.number_input("Alpha (L2)", 0.0001, 1.0, 0.0001, 0.0001, key="mlp_alpha")
                
                # Кнопка обучения
                if st.button("Обучить модели", key="train_button"):
                    with st.spinner("Подготовка и обучение моделей..."):
                        try:
                            # Копируем для обработки
                            df_processed = train_df.copy()
                            
                            # Обработка целевой переменной
                            y = df_processed[target_col].copy()
                            
                            # Преобразование целевой переменной
                            if y.dtype == 'object':
                                unique_values = y.unique()
                                st.info(f"Найдены текстовые значения в целевой переменной: {unique_values}")
                                
                                # Автоматическое определение положительного класса
                                positive_keywords = ['yes', 'y', 'true', '1', 'positive', 'diabetic', 'diabetes', 'sick', 'ill']
                                negative_keywords = ['no', 'n', 'false', '0', 'negative', 'non-diabetic', 'healthy', 'normal']
                                
                                mapping = {}
                                for val in unique_values:
                                    val_lower = str(val).lower().strip()
                                    if val_lower in positive_keywords or 'yes' in val_lower or 'diabet' in val_lower:
                                        mapping[val] = 1
                                    elif val_lower in negative_keywords or 'no' in val_lower or 'normal' in val_lower:
                                        mapping[val] = 0
                                    else:
                                        mapping[val] = None
                                
                                if any(v is None for v in mapping.values()):
                                    le = LabelEncoder()
                                    y = le.fit_transform(y)
                                    st.warning(f"Использован LabelEncoder. Классы: {dict(zip(le.classes_, le.transform(le.classes_)))}")
                                else:
                                    y = y.map(mapping)
                                    st.success(f"Преобразование целевой переменной: {mapping}")
                            else:
                                y = y.astype(int)
                            
                            # Обработка признаков
                            X = df_processed.drop(columns=[target_col])
                            
                            # Обработка числовых колонок с запятыми
                            for col in X.columns:
                                if X[col].dtype == 'object':
                                    try:
                                        X[col] = X[col].str.replace(',', '.').astype(float)
                                    except:
                                        pass
                            
                            # Определяем категориальные колонки
                            categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
                            
                            if categorical_cols:
                                st.info(f"Категориальные колонки: {categorical_cols}")
                                X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
                            
                            # Заполнение пропусков
                            imp = SimpleImputer(strategy='median')
                            X = pd.DataFrame(imp.fit_transform(X), columns=X.columns)
                            
                            # Проверяем размер данных
                            if len(X) < 20:
                                st.error("❌ Слишком маленький датасет для обучения. Минимум 20 строк.")
                            else:
                                # Показываем распределение классов
                                class_counts = pd.Series(y).value_counts()
                                st.info(f"Распределение классов: {dict(class_counts)}")
                                
                                # Обучаем модели с выбранными параметрами
                                from sklearn.utils.class_weight import compute_class_weight
                                
                                # Подготавливаем модели
                                models_to_train = {}
                                
                                if model_selection['logistic']:
                                    penalty = model_params.get('logistic_penalty', 'l2')
                                    if penalty == 'none':
                                        models_to_train['Logistic Regression'] = LogisticRegression(
                                            C=model_params.get('logistic_C', 1.0),
                                            max_iter=int(model_params.get('logistic_max_iter', 1000)),
                                            penalty=None,
                                            random_state=random_state
                                        )
                                    else:
                                        models_to_train['Logistic Regression'] = LogisticRegression(
                                            C=model_params.get('logistic_C', 1.0),
                                            max_iter=int(model_params.get('logistic_max_iter', 1000)),
                                            penalty=penalty,
                                            random_state=random_state,
                                            class_weight='balanced' if balance_classes else None
                                        )
                                
                                if model_selection['random_forest']:
                                    max_depth = int(model_params.get('rf_max_depth', 0))
                                    models_to_train['Random Forest'] = RandomForestClassifier(
                                        n_estimators=int(model_params.get('rf_n_estimators', 100)),
                                        max_depth=max_depth if max_depth > 0 else None,
                                        min_samples_split=int(model_params.get('rf_min_samples_split', 2)),
                                        random_state=random_state,
                                        class_weight='balanced' if balance_classes else None
                                    )
                                
                                if model_selection['xgboost']:
                                    from xgboost import XGBClassifier
                                    models_to_train['XGBoost'] = XGBClassifier(
                                        n_estimators=int(model_params.get('xgb_n_estimators', 100)),
                                        learning_rate=model_params.get('xgb_learning_rate', 0.1),
                                        max_depth=int(model_params.get('xgb_max_depth', 6)),
                                        random_state=random_state,
                                        eval_metric='logloss',
                                        verbosity=0,
                                        scale_pos_weight=class_counts[0]/class_counts[1] if balance_classes else 1
                                    )
                                
                                if model_selection['lightgbm']:
                                    from lightgbm import LGBMClassifier
                                    models_to_train['LightGBM'] = LGBMClassifier(
                                        n_estimators=int(model_params.get('lgbm_n_estimators', 100)),
                                        learning_rate=model_params.get('lgbm_learning_rate', 0.1),
                                        num_leaves=int(model_params.get('lgbm_num_leaves', 31)),
                                        random_state=random_state,
                                        verbose=-1,
                                        class_weight='balanced' if balance_classes else None
                                    )
                                
                                if model_selection['mlp']:
                                    from sklearn.neural_network import MLPClassifier
                                    hidden_layers = [int(x.strip()) for x in model_params.get('mlp_hidden', '100,50').split(',')]
                                    models_to_train['Neural Network'] = MLPClassifier(
                                        hidden_layer_sizes=tuple(hidden_layers),
                                        max_iter=int(model_params.get('mlp_max_iter', 1000)),
                                        alpha=model_params.get('mlp_alpha', 0.0001),
                                        random_state=random_state
                                    )
                                
                                if not models_to_train:
                                    st.error("Выберите хотя бы одну модель для обучения.")
                                else:
                                    # Разделяем данные
                                    X_train, X_test, y_train, y_test = train_test_split(
                                        X, y, test_size=test_size, random_state=random_state, stratify=y
                                    )
                                    
                                    # Сохраняем тестовые данные для калибровочной кривой
                                    st.session_state['X_test_scaled'] = None
                                    st.session_state['y_test'] = y_test
                                    
                                    # Стандартизация
                                    scaler = StandardScaler()
                                    X_train_scaled = scaler.fit_transform(X_train)
                                    X_test_scaled = scaler.transform(X_test)
                                    
                                    # Сохраняем тестовые данные
                                    st.session_state['X_test_scaled'] = X_test_scaled
                                    
                                    # Обучаем все выбранные модели
                                    trained_models = {}
                                    metrics_list = []

                                    for name, model in models_to_train.items():
                                        try:
                                            model.fit(X_train_scaled, y_train)
                                            
                                            # Калибровка вероятностей
                                            if balance_classes:
                                                model = calibrate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
                                            
                                            y_proba = model.predict_proba(X_test_scaled)[:, 1]
                                            y_pred = (y_proba >= 0.5).astype(int)
                                            metrics = evaluate_model_metrics(y_test, y_pred, y_proba)
                                            metrics['Model'] = name
                                            metrics_list.append(metrics)
                                            trained_models[name] = model
                                            st.success(f"✅ {name} обучена")
                                        except Exception as e:
                                            st.error(f"❌ {name}: {str(e)[:100]}")                        
            
                                    if metrics_list:
                                        metrics_df = pd.DataFrame(metrics_list).set_index('Model')
                                        st.markdown("### 📊 Результаты обучения")
                                        st.dataframe(metrics_df.style.highlight_max(axis=0))
                                        
                                        # Определяем метрику для выбора лучшей модели
                                        best_metric = get_best_model_metric(metrics_df, y)
                                        st.info(f"Метрика для выбора модели: {best_metric} (автоматически определена на основе баланса классов)")
                                        
                                        # Показываем калибровочную кривую для лучшей модели
                                        best_model_name = metrics_df[best_metric].idxmax()
                                        best_model = trained_models[best_model_name]
                                        y_proba_best = best_model.predict_proba(X_test_scaled)[:, 1]
                                        
                                        with st.expander("📈 Калибровочная кривая лучшей модели"):
                                            fig_cal = plot_calibration_curve(y_test, y_proba_best)
                                            st.pyplot(fig_cal, clear_figure=True, use_container_width=True)
                                            plt.close(fig_cal)
                                        
                                        # Сохраняем все обученные модели
                                        st.session_state['trained_models'] = trained_models
                                        st.session_state['scaler'] = scaler
                                        st.session_state['X_columns'] = X.columns.tolist()
                                        st.session_state['X_data'] = X
                                        st.session_state['y_data'] = y
                                        st.session_state['model_trained'] = True
                                        st.session_state['metrics_df'] = metrics_df
                                        st.session_state['best_metric'] = best_metric

                                        # Показываем выбор модели
                                        st.markdown("### 🎯 Выберите модель для использования")
                                        
                                        # Находим лучшую модель по выбранной метрике
                                        best_model_name = metrics_df[best_metric].idxmax()
                                        
                                        # Создаем выбор модели
                                        selected_model = st.selectbox(
                                            "Выберите модель для предсказаний",
                                            options=list(trained_models.keys()),
                                            index=list(trained_models.keys()).index(best_model_name),
                                            key="selected_model"
                                        )
                                        
                                        # Сохраняем выбранную модель
                                        st.session_state['custom_model'] = trained_models[selected_model]
                                        st.session_state['custom_scaler'] = scaler
                                        st.session_state['custom_columns'] = X.columns.tolist()
                                        st.session_state['model_pr_auc'] = metrics_df.loc[selected_model, 'PR-AUC']
                                        st.session_state['model_roc_auc'] = metrics_df.loc[selected_model, 'ROC-AUC']
                                        st.session_state['model_f1'] = metrics_df.loc[selected_model, 'F1']
                                        st.session_state['best_metric'] = best_metric
                                        
                                        st.success(f"✅ Выбрана модель: {selected_model}")
                                        
                                        # Кнопка скачивания выбранной модели
                                        import tempfile
                                        import os
                                        
                                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
                                            joblib.dump(trained_models[selected_model], tmp.name)
                                            tmp_path = tmp.name
                                        
                                        with open(tmp_path, 'rb') as f:
                                            model_bytes = f.read()
                                        
                                        st.download_button(
                                            f"💾 Скачать модель ({selected_model})",
                                            data=model_bytes,
                                            file_name=f"{selected_model.lower().replace(' ', '_')}.pkl",
                                            mime="application/octet-stream"
                                        )
                                        
                                        os.unlink(tmp_path)
                                        
                                    else:
                                        st.error("Не удалось обучить ни одну модель.")
                        
                        except Exception as e:
                            st.error(f"Ошибка при обучении: {e}")
                            st.info("Проверьте формат данных и попробуйте снова.")
            
            except Exception as e:
                st.error(f"Ошибка при чтении файла: {e}")
                st.info("Проверьте формат файла. Убедитесь, что:")
                st.info("1. Файл в формате CSV")
                st.info("2. Целевая переменная содержит значения 0/1 или Yes/No")
                st.info("3. Числовые значения используют точку или запятую как десятичный разделитель")
        
        # Тестирование модели (отдельно от обучения)
        if st.session_state.get('model_trained', False):
            st.markdown("---")
            st.markdown("### 🧪 Тестирование обученной модели")
            
            # Позволяем переключить модель
            if 'trained_models' in st.session_state:
                trained_models = st.session_state['trained_models']
                if len(trained_models) > 1:
                    selected_model = st.selectbox(
                        "Выберите модель для тестирования",
                        options=list(trained_models.keys()),
                        key="test_model_select"
                    )
                    st.session_state['custom_model'] = trained_models[selected_model]
                    
                    # Обновляем метрики для выбранной модели
                    if 'metrics_df' in st.session_state:
                        metrics_df = st.session_state['metrics_df']
                        if selected_model in metrics_df.index:
                            st.session_state['model_pr_auc'] = metrics_df.loc[selected_model, 'PR-AUC']
                            st.session_state['model_roc_auc'] = metrics_df.loc[selected_model, 'ROC-AUC']
                            st.session_state['model_f1'] = metrics_df.loc[selected_model, 'F1']

            # Показываем информацию о качестве модели
            model_pr_auc = st.session_state.get('model_pr_auc', 0)
            model_roc_auc = st.session_state.get('model_roc_auc', 0)
            best_metric = st.session_state.get('best_metric', 'PR-AUC')
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("PR-AUC", f"{model_pr_auc:.4f}")
            with col2:
                st.metric("ROC-AUC", f"{model_roc_auc:.4f}")
            
            st.info(f"Модель выбрана по метрике: {best_metric}")
            
            # Предупреждение о качестве в зависимости от метрики
            if best_metric == 'PR-AUC' and model_pr_auc < 0.3:
                st.warning(f"⚠️ Внимание: модель имеет низкое качество (PR-AUC: {model_pr_auc:.3f}).")
            elif best_metric == 'ROC-AUC' and model_roc_auc < 0.7:
                st.warning(f"⚠️ Внимание: модель имеет низкое качество (ROC-AUC: {model_roc_auc:.3f}).")
            
            st.caption("Введите значения признаков для проверки модели:")
            
            # Получаем сохраненные данные
            X_test_data = st.session_state.get('X_data')
            y_data = st.session_state.get('y_data')
            
            if X_test_data is not None:
                # Показываем информацию о распределении классов
                if y_data is not None:
                    class_counts = pd.Series(y_data).value_counts()
                    st.info(f"Распределение классов: {dict(class_counts)}")
                
                # Создаем поля для ввода значений
                test_input = {}
                test_features = X_test_data.columns.tolist()
                
                # Определяем бинарные и целочисленные признаки
                binary_features = []
                integer_features = []
                
                for col in test_features:
                    unique_vals = X_test_data[col].unique()
                    if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1}):
                        binary_features.append(col)
                    elif all(float(val).is_integer() for val in X_test_data[col].dropna()):
                        integer_features.append(col)
                
                # Создаем колонки для ввода
                cols = st.columns(2)
                for idx, col_name in enumerate(test_features):
                    with cols[idx % 2]:
                        if col_name in binary_features:
                            test_input[col_name] = st.selectbox(
                                f"{col_name}",
                                options=[0, 1],
                                format_func=lambda x: "Нет (0)" if x == 0 else "Да (1)",
                                key=f"test_input_{col_name}"
                            )
                        elif col_name in integer_features:
                            mean_val = int(round(X_test_data[col_name].mean()))
                            min_val = int(X_test_data[col_name].min())
                            max_val = int(X_test_data[col_name].max())
                            test_input[col_name] = st.number_input(
                                f"{col_name}",
                                min_value=min_val,
                                max_value=max_val,
                                value=mean_val,
                                step=1,
                                key=f"test_input_{col_name}"
                            )
                        else:
                            mean_val = float(X_test_data[col_name].mean())
                            min_val = float(X_test_data[col_name].min())
                            max_val = float(X_test_data[col_name].max())
                            test_input[col_name] = st.number_input(
                                f"{col_name}",
                                min_value=min_val,
                                max_value=max_val,
                                value=round(mean_val, 2),
                                step=0.01,
                                format="%.2f",
                                key=f"test_input_{col_name}"
                            )
                
                # Кнопка предсказания
                if st.button("🔮 Предсказать", key="predict_button"):
                    try:
                        # Подготовка данных для предсказания
                        full_input = pd.DataFrame(0, index=[0], columns=st.session_state['X_columns'])
                        
                        for col_name, value in test_input.items():
                            full_input[col_name] = value
                        
                        # Применяем scaler
                        test_scaled = st.session_state['custom_scaler'].transform(full_input)
                        
                        # Предсказание
                        probability = st.session_state['custom_model'].predict_proba(test_scaled)[0]
                        
                        # Адаптивный порог
                        if y_data is not None:
                            positive_ratio = np.mean(y_data == 1)
                            threshold = min(max(positive_ratio * 0.5, 0.15), 0.3)
                        else:
                            threshold = 0.3
                        
                        prediction = 1 if probability[1] >= threshold else 0
                        
                        # Вывод результата
                        st.markdown("---")
                        st.markdown("### Результат предсказания")
                        
                        st.info(f"Использован адаптивный порог: {threshold:.2f}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if prediction == 1:
                                st.error(f"🔴 Положительный результат")
                            else:
                                st.success(f"🟢 Отрицательный результат")
                        with col2:
                            st.metric("Вероятность положительного результата", f"{probability[1]*100:.1f}%")
                        
                        st.progress(float(probability[1]))
                        
                        # Таблица вероятностей
                        prob_df = pd.DataFrame({
                            'Класс': ['Отрицательный (0)', 'Положительный (1)'],
                            'Вероятность': [f"{probability[0]*100:.1f}%", f"{probability[1]*100:.1f}%"]
                        })
                        st.dataframe(prob_df, hide_index=True, use_container_width=True)
                        
                        # Важность признаков
                        with st.expander("📊 Важность признаков"):
                            try:
                                if hasattr(st.session_state['custom_model'], 'feature_importances_'):
                                    importances = st.session_state['custom_model'].feature_importances_
                                    feature_importance_df = pd.DataFrame({
                                        'Признак': st.session_state['X_columns'],
                                        'Важность': importances
                                    }).sort_values('Важность', ascending=False).head(10)
                                    st.dataframe(feature_importance_df, hide_index=True, use_container_width=True)
                                    
                                    # Визуализация
                                    fig, ax = plt.subplots(figsize=(8, 4))
                                    ax.barh(feature_importance_df['Признак'][:5], feature_importance_df['Важность'][:5])
                                    ax.set_xlabel('Важность')
                                    ax.set_title('Топ-5 признаков')
                                    st.pyplot(fig, clear_figure=True, use_container_width=True)
                                    plt.close(fig)
                                elif hasattr(st.session_state['custom_model'], 'coef_'):
                                    coefs = st.session_state['custom_model'].coef_[0]
                                    feature_importance_df = pd.DataFrame({
                                        'Признак': st.session_state['X_columns'],
                                        'Коэффициент': coefs
                                    }).sort_values('Коэффициент', key=abs, ascending=False).head(10)
                                    st.dataframe(feature_importance_df, hide_index=True, use_container_width=True)
                            except:
                                st.info("Важность признаков недоступна для этой модели.")

                    except Exception as e:
                        st.error(f"Ошибка при предсказании: {e}")
                        st.info("Проверьте, что все значения введены корректно.")

    # ======================== ЭКСПАНДЕР 2: КОМПЛЕКСНЫЙ АНАЛИЗ ДАННЫХ ========================
    with st.expander("📊 Комплексный анализ данных", expanded=st.session_state['expander_states']['analysis']):
        # Загрузка датасета для анализа
        st.subheader("📁 Загрузка датасета для анализа")
        analysis_file = st.file_uploader(
            "Загрузите CSV-файл с данными пациентов для кластеризации и корреляционного анализа",
            type=["csv"],
            key="analysis_file",
            help="Файл должен содержать числовые и/или бинарные признаки пациентов"
        )

        analysis_df = None
        if analysis_file is not None:
            try:
                # Читаем CSV с автоопределением разделителя
                analysis_df = pd.read_csv(analysis_file, sep=None, engine='python')
                
                # Если файл не разделился правильно, пробуем другие разделители
                if analysis_df.shape[1] == 1:
                    for sep in [';', '\t', '|']:
                        try:
                            analysis_df = pd.read_csv(analysis_file, sep=sep, engine='python')
                            if analysis_df.shape[1] > 1:
                                st.info(f"Использован разделитель: '{sep}'")
                                break
                        except:
                            continue
                
                st.success(f"✅ Загружен датасет: {analysis_df.shape[0]} строк, {analysis_df.shape[1]} колонок")
                
                # Показываем первые строки
                with st.expander("Просмотр данных"):
                    st.dataframe(analysis_df.head(10))
                
                # Обработка данных
                # Заменяем запятые на точки в числовых колонках
                for col in analysis_df.columns:
                    if analysis_df[col].dtype == 'object':
                        try:
                            analysis_df[col] = analysis_df[col].str.replace(',', '.').astype(float)
                        except:
                            pass
                
                # Сохраняем в session_state
                st.session_state['analysis_df'] = analysis_df
                
            except Exception as e:
                st.error(f"Ошибка при чтении файла: {e}")
                analysis_df = None
        elif st.session_state.get('analysis_df') is not None:
            analysis_df = st.session_state['analysis_df']
            st.info("Используется ранее загруженный датасет")

        # 1. Корреляционный анализ числовых признаков
        st.subheader("📈 Корреляционный анализ")

        # Корреляционная матрица на основе загруженных данных
        if analysis_df is not None and analysis_df.shape[0] >= 5:
            st.write("### Корреляционная матрица (загруженные данные)")
            
            # Выбираем только числовые колонки
            numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 2:
                # Ограничиваем количество колонок
                if len(numeric_cols) > 20:
                    st.caption(f"Показаны первые 20 из {len(numeric_cols)} числовых колонок")
                    numeric_cols = numeric_cols[:20]
                
                corr_matrix = analysis_df[numeric_cols].corr()
                
                # Тепловая карта
                fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
                im = ax_corr.imshow(corr_matrix.values, cmap='coolwarm', vmin=-1, vmax=1)
                ax_corr.set_xticks(range(len(corr_matrix.columns)))
                ax_corr.set_yticks(range(len(corr_matrix.columns)))
                ax_corr.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=8)
                ax_corr.set_yticklabels(corr_matrix.columns, fontsize=8)
                ax_corr.set_title("Корреляция между числовыми показателями")
                
                # Добавляем значения
                for i in range(len(corr_matrix.columns)):
                    for j in range(len(corr_matrix.columns)):
                        ax_corr.text(j, i, f"{corr_matrix.values[i, j]:.2f}",
                                    ha="center", va="center", fontsize=6,
                                    color="white" if abs(corr_matrix.values[i, j]) > 0.5 else "black")
                
                fig_corr.colorbar(im, ax=ax_corr)
                fig_corr.tight_layout()
                st.pyplot(fig_corr, clear_figure=True, use_container_width=True)
                plt.close(fig_corr)
                
                # Кнопка для скачивания корреляционной матрицы
                csv_corr = corr_matrix.to_csv()
                st.download_button(
                    "Скачать корреляционную матрицу (CSV)",
                    data=csv_corr,
                    file_name="correlation_matrix.csv",
                    mime="text/csv",
                    key="download_corr"
                )
            else:
                st.info("Недостаточно числовых колонок в загруженных данных для корреляции.")
        else:
            if analysis_df is None:
                st.info("💡 Загрузите датасет выше для построения корреляционной матрицы (нужно минимум 5 наблюдений).")
            else:
                st.info(f"💡 В загруженном датасете только {analysis_df.shape[0]} строк. Нужно минимум 5 для корреляции.")

        # 2. Кластеризация
        st.subheader("🧩 Кластеризация данных")

        if analysis_df is not None and analysis_df.shape[0] >= 5:
            st.write("### Иерархическая кластеризация")
            
            # Выбор метода кластеризации
            cluster_method = st.selectbox(
                "Метод кластеризации",
                ["Иерархическая (по строкам)", "K-Means (по строкам)", "Иерархическая (по колонкам)"],
                key="cluster_method"
            )
            
            # Подготовка данных
            # Выбираем только числовые колонки
            numeric_df_cluster = analysis_df.select_dtypes(include=[np.number])
            
            # Заполняем пропуски
            if numeric_df_cluster.isnull().any().any():
                imputer = SimpleImputer(strategy='median')
                numeric_df_cluster = pd.DataFrame(
                    imputer.fit_transform(numeric_df_cluster),
                    columns=numeric_df_cluster.columns
                )
                st.caption("Пропуски заполнены медианой")
            
            # Стандартизация
            scaler_cluster = StandardScaler()
            data_scaled = scaler_cluster.fit_transform(numeric_df_cluster)
            
            if cluster_method == "Иерархическая (по строкам)":
                # Кластеризация наблюдений
                from scipy.cluster.hierarchy import dendrogram, linkage
                from scipy.spatial.distance import pdist
                
                # Ограничиваем количество строк для читаемости
                if len(data_scaled) > 100:
                    st.caption(f"Показаны первые 100 из {len(data_scaled)} наблюдений")
                    data_for_cluster = data_scaled[:100]
                    labels = [f"Пациент {i+1}" for i in range(100)]
                else:
                    data_for_cluster = data_scaled
                    labels = [f"Пациент {i+1}" for i in range(len(data_scaled))]
                
                # Вычисляем матрицу связей
                linkage_matrix = linkage(data_for_cluster, method='ward')
                
                # Дендрограмма
                fig_dendro, ax_dendro = plt.subplots(figsize=(12, 6))
                dendrogram(linkage_matrix, labels=labels, ax=ax_dendro, leaf_rotation=90, leaf_font_size=8)
                ax_dendro.set_title("Дендрограмма наблюдений (метод Ward)")
                ax_dendro.set_xlabel("Наблюдения")
                ax_dendro.set_ylabel("Расстояние")
                fig_dendro.tight_layout()
                st.pyplot(fig_dendro, clear_figure=True, use_container_width=True)
                plt.close(fig_dendro)
                
            elif cluster_method == "K-Means (по строкам)":
                # K-Means кластеризация
                from sklearn.cluster import KMeans
                
                n_clusters = st.slider("Количество кластеров", 2, 10, 3, key="n_clusters_kmeans")
                
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(data_scaled)
                
                # Добавляем кластеры к данным
                df_with_clusters = numeric_df_cluster.copy()
                df_with_clusters['Кластер'] = clusters
                
                # Визуализация с PCA
                from sklearn.decomposition import PCA
                
                pca = PCA(n_components=2)
                data_pca = pca.fit_transform(data_scaled)
                
                fig_kmeans, ax_kmeans = plt.subplots(figsize=(10, 6))
                scatter = ax_kmeans.scatter(data_pca[:, 0], data_pca[:, 1], 
                                            c=clusters, cmap='viridis', alpha=0.7, s=50)
                ax_kmeans.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
                ax_kmeans.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
                ax_kmeans.set_title(f'K-Means кластеризация ({n_clusters} кластеров)')
                ax_kmeans.grid(True, alpha=0.3)
                fig_kmeans.colorbar(scatter, ax=ax_kmeans, label='Кластер')
                fig_kmeans.tight_layout()
                st.pyplot(fig_kmeans, clear_figure=True, use_container_width=True)
                plt.close(fig_kmeans)
                
                # Статистика по кластерам
                st.write("### Статистика по кластерам")
                cluster_stats = df_with_clusters.groupby('Кластер').agg(['mean', 'std', 'count'])
                st.dataframe(cluster_stats, use_container_width=True)
                
            elif cluster_method == "Иерархическая (по колонкам)":
                # Кластеризация признаков
                from scipy.cluster.hierarchy import dendrogram, linkage
                
                # Транспонируем данные
                data_transposed = data_scaled.T
                
                if data_transposed.shape[0] < 2:
                    st.warning("Недостаточно признаков для кластеризации по колонкам")
                else:
                    linkage_matrix_cols = linkage(data_transposed, method='ward')
                    
                    fig_dendro_cols, ax_dendro_cols = plt.subplots(figsize=(12, 6))
                    dendrogram(
                        linkage_matrix_cols, 
                        labels=numeric_df_cluster.columns.tolist(), 
                        ax=ax_dendro_cols, 
                        leaf_rotation=90, 
                        leaf_font_size=8
                    )
                    ax_dendro_cols.set_title("Дендрограмма признаков (метод Ward)")
                    ax_dendro_cols.set_xlabel("Признаки")
                    ax_dendro_cols.set_ylabel("Расстояние")
                    fig_dendro_cols.tight_layout()
                    st.pyplot(fig_dendro_cols, clear_figure=True, use_container_width=True)
                    plt.close(fig_dendro_cols)
        else:
            if analysis_df is None:
                st.info("💡 Загрузите датасет выше для выполнения кластеризации (нужно минимум 5 наблюдений).")
            else:
                st.info(f"💡 В загруженном датасете только {analysis_df.shape[0]} строк. Нужно минимум 5 для кластеризации.")

        # 3. Расширенная аналитика для врача
        st.markdown("---")
        st.subheader("📊 Расширенная аналитика для врача")
        
        # Инициализация всех показателей значениями по умолчанию
        diabetes_score = 0
        ir_score = 0
        obesity_score = 0
        hypothyroid_score = 0
        hyperthyroid_score = 0
        pcos_score = None
        bone_score = 0
        metabolic_score = 0
        cushing_rule_score = 0
        addison_rule_score = 0
        hyperpara_rule_score = 0
        network_score = 0

        if not st.session_state.get('report_generated', False):
            st.warning("⚠️ **Для отображения расширенной аналитики необходимо сначала собрать медицинскую карту.**")
            st.info("Пожалуйста, заполните форму выше и нажмите кнопку 'Собрать медицинскую карту'.")
        else:
            results = st.session_state.get('results', {})
            if not results:
                st.warning("Результаты не найдены. Пожалуйста, соберите медицинскую карту заново.")
            else:
                # Переопределяем переменные реальными значениями
                diabetes_score = results.get('diabetes_score', 0)
                ir_score = results.get('ir_score', 0)
                obesity_score = results.get('obesity_score', 0)
                hypothyroid_score = results.get('hypothyroid_score', 0)
                hyperthyroid_score = results.get('hyperthyroid_score', 0)
                pcos_score = results.get('pcos_score', None)
                bone_score = results.get('bone_score', 0)
                metabolic_score = results.get('metabolic_score', 0)
                cushing_rule_score = results.get('cushing_rule_score', 0)
                addison_rule_score = results.get('addison_rule_score', 0)
                hyperpara_rule_score = results.get('hyperpara_rule_score', 0)
                network_score = results.get('network_score', 0)

                # Инициализация analysis_df из session_state, если она там есть
                if 'analysis_df' not in st.session_state:
                    st.session_state['analysis_df'] = None

                analysis_df = st.session_state['analysis_df']

                # 1. Описательная статистика и распределения
                st.subheader("1. Описательная статистика и распределения")
                if analysis_df is not None and analysis_df.shape[0] > 0:
                    numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
                    if numeric_cols:
                        st.write("#### Описательные метрики числовых колонок")
                        st.dataframe(analysis_df[numeric_cols].describe().T, use_container_width=True)

                        # Выбор колонки для детального просмотра
                        selected_col_for_dist = st.selectbox(
                            "Выберите колонку для просмотра распределения",
                            options=numeric_cols,
                            key="dist_col_select"
                        )

                        if selected_col_for_dist:
                            col1, col2 = st.columns(2)
                            with col1:
                                # Гистограмма
                                fig_hist, ax_hist = plt.subplots(figsize=(5, 4))
                                ax_hist.hist(analysis_df[selected_col_for_dist].dropna(), bins=20, edgecolor='black', alpha=0.7)
                                ax_hist.set_title(f"Гистограмма: {selected_col_for_dist}")
                                ax_hist.set_xlabel(selected_col_for_dist)
                                ax_hist.set_ylabel("Частота")
                                ax_hist.grid(True, alpha=0.3)
                                st.pyplot(fig_hist, clear_figure=True, use_container_width=True)
                                plt.close(fig_hist)
                            with col2:
                                # Boxplot
                                fig_box, ax_box = plt.subplots(figsize=(5, 4))
                                ax_box.boxplot(analysis_df[selected_col_for_dist].dropna(), vert=False)
                                ax_box.set_title(f"Boxplot: {selected_col_for_dist}")
                                ax_box.set_xlabel(selected_col_for_dist)
                                ax_box.grid(True, alpha=0.3)
                                st.pyplot(fig_box, clear_figure=True, use_container_width=True)
                                plt.close(fig_box)
                    else:
                        st.info("В загруженном датасете нет числовых колонок.")
                else:
                    st.info("💡 Загрузите датасет выше, чтобы увидеть описательную статистику.")

                # 2. Проверка нормальности
                st.subheader("2. Проверка нормальности распределения")
                if analysis_df is not None and analysis_df.shape[0] >= 3:
                    numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
                    if numeric_cols:
                        from scipy.stats import shapiro
                        normality_results = []
                        for col in numeric_cols:
                            data = analysis_df[col].dropna()
                            if len(data) >= 3:
                                stat, p = shapiro(data)
                                normality_results.append({
                                    "Признак": col,
                                    "p-value": f"{p:.4f}",
                                    "Распределение": "Нормальное" if p > 0.05 else "Отличается от нормального"
                                })
                        if normality_results:
                            st.write("#### Результаты теста Шапиро-Уилка")
                            normality_df = pd.DataFrame(normality_results)
                            st.dataframe(normality_df, use_container_width=True, hide_index=True)
                            st.caption("Порог значимости: 0.05. Если p-value > 0.05, распределение можно считать нормальным.")
                        else:
                            st.info("Недостаточно данных для проверки нормальности (нужно минимум 3 наблюдения на признак).")
                    else:
                        st.info("Нет числовых колонок для проверки.")
                else:
                    st.info("💡 Загрузите датасет (минимум 3 строки) для проверки нормальности.")

                # 3. Выбросы
                st.subheader("3. Выбросы (Outliers)")
                if analysis_df is not None and analysis_df.shape[0] > 0:
                    numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
                    if numeric_cols:
                        outlier_method = st.radio(
                            "Метод обнаружения выбросов",
                            options=["IQR (межквартильный размах)", "Z-score"],
                            key="outlier_method"
                        )
                        if outlier_method == "IQR (межквартильный размах)":
                            outlier_rows = set()
                            outlier_details = []
                            for col in numeric_cols:
                                Q1 = analysis_df[col].quantile(0.25)
                                Q3 = analysis_df[col].quantile(0.75)
                                IQR = Q3 - Q1
                                lower_bound = Q1 - 1.5 * IQR
                                upper_bound = Q3 + 1.5 * IQR
                                col_outliers = analysis_df[(analysis_df[col] < lower_bound) | (analysis_df[col] > upper_bound)]
                                if not col_outliers.empty:
                                    for idx in col_outliers.index:
                                        outlier_rows.add(idx)
                                        outlier_details.append({
                                            "Строка": idx,
                                            "Признак": col,
                                            "Значение": analysis_df.loc[idx, col],
                                            "Границы": f"[{lower_bound:.2f}, {upper_bound:.2f}]"
                                        })
                            if outlier_details:
                                st.warning(f"Обнаружено выбросов: {len(outlier_rows)} строк(и).")
                                outlier_df = pd.DataFrame(outlier_details)
                                st.dataframe(outlier_df, use_container_width=True)
                                # Скачивание
                                csv_outliers = outlier_df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    "Скачать список выбросов (CSV)",
                                    data=csv_outliers,
                                    file_name="outliers.csv",
                                    mime="text/csv",
                                    key="download_outliers"
                                )
                                # Boxplot с выбросами для каждой колонки? Слишком много графиков, можно выбрать одну.
                                selected_out_col = st.selectbox("Выберите колонку для визуализации выбросов", numeric_cols, key="out_col_select")
                                fig_out, ax_out = plt.subplots(figsize=(8, 4))
                                ax_out.boxplot(analysis_df[selected_out_col].dropna(), vert=False)
                                ax_out.set_title(f"Boxplot с выбросами: {selected_out_col}")
                                ax_out.grid(True, alpha=0.3)
                                st.pyplot(fig_out, clear_figure=True, use_container_width=True)
                                plt.close(fig_out)
                            else:
                                st.success("Выбросы по методу IQR не обнаружены.")
                        else:  # Z-score
                            from scipy import stats
                            z_threshold = st.slider("Порог Z-score", 2.0, 4.0, 3.0, 0.1, key="z_threshold")
                            outlier_rows = set()
                            outlier_details = []
                            for col in numeric_cols:
                                z_scores = np.abs(stats.zscore(analysis_df[col].dropna()))
                                col_outliers = analysis_df.loc[z_scores[z_scores > z_threshold].index]
                                if not col_outliers.empty:
                                    for idx in col_outliers.index:
                                        outlier_rows.add(idx)
                                        outlier_details.append({
                                            "Строка": idx,
                                            "Признак": col,
                                            "Значение": analysis_df.loc[idx, col],
                                            "Z-score": z_scores.loc[idx]
                                        })
                            if outlier_details:
                                st.warning(f"Обнаружено выбросов: {len(outlier_rows)} строк(и).")
                                outlier_df = pd.DataFrame(outlier_details)
                                st.dataframe(outlier_df, use_container_width=True)
                                csv_outliers = outlier_df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    "Скачать список выбросов (CSV)",
                                    data=csv_outliers,
                                    file_name="outliers_zscore.csv",
                                    mime="text/csv",
                                    key="download_outliers_z"
                                )
                            else:
                                st.success(f"Выбросы по Z-score (порог {z_threshold}) не обнаружены.")
                    else:
                        st.info("Нет числовых колонок для анализа выбросов.")
                else:
                    st.info("💡 Загрузите датасет для обнаружения выбросов.")

                # 4. Сравнение с референсными диапазонами
                st.subheader("4. Сравнение с референсными диапазонами")
                st.caption("Оценка введённых лабораторных и антропометрических показателей относительно общепринятых норм.")

                # Получаем данные из сохраненных входных данных
                input_data = st.session_state.get('input_data', {})
                fasting_glucose = input_data.get('fasting_glucose', 0)
                hba1c = input_data.get('hba1c', 0)
                tsh_value = input_data.get('tsh_value', 0)
                ft4_value = input_data.get('ft4_value', 0)
                serum_calcium = input_data.get('serum_calcium', 0)
                bmi = input_data.get('bmi', 0)
                waist_cm = input_data.get('waist_cm', 0)
                sleep_hours = input_data.get('sleep_hours', 0)
                age = input_data.get('age', 0)
                gender = input_data.get('gender', 0)
                height_cm = input_data.get('height_cm', 0)
                weight_kg = input_data.get('weight_kg', 0)

                # Словарь референсных диапазонов
                reference_ranges = {
                    "Глюкоза натощак (мг/дл)": (70, 99),
                    "HbA1c (%)": (4.0, 5.6),
                    "ТТГ (мМЕ/л)": (0.4, 4.0),
                    "Св. T4 (нг/дл)": (0.8, 1.8),
                    "Кальций общий (мг/дл)": (8.5, 10.2),
                    "ИМТ": (18.5, 24.9),
                    "Талия (см)": (None, 94 if gender == 0 else 80),
                    "Сон (часов)": (7, 9),
                    "Возраст (лет)": (18, 65),
                }

                # Собираем текущие значения
                current_values = {
                    "Глюкоза натощак (мг/дл)": fasting_glucose if fasting_glucose > 0 else None,
                    "HbA1c (%)": hba1c if hba1c > 0 else None,
                    "ТТГ (мМЕ/л)": tsh_value if tsh_value > 0 else None,
                    "Св. T4 (нг/дл)": ft4_value if ft4_value > 0 else None,
                    "Кальций общий (мг/дл)": serum_calcium if serum_calcium > 0 else None,
                    "ИМТ": bmi,
                    "Талия (см)": waist_cm,
                    "Сон (часов)": sleep_hours,
                    "Возраст (лет)": age,
                }

                ref_table = []
                for indicator, (low, high) in reference_ranges.items():
                    val = current_values.get(indicator)
                    if val is None:
                        continue
                    status = ""
                    if low is not None and val < low:
                        status = "⬇️ Ниже нормы"
                    elif high is not None and val > high:
                        status = "⬆️ Выше нормы"
                    else:
                        status = "✅ В норме"
                    ref_range_str = f"{low if low is not None else '—'} – {high if high is not None else '—'}"
                    ref_table.append({
                        "Показатель": indicator,
                        "Значение": f"{val:.2f}" if isinstance(val, float) else str(val),
                        "Референс": ref_range_str,
                        "Статус": status
                    })

                if ref_table:
                    ref_df = pd.DataFrame(ref_table)
                    st.dataframe(ref_df, use_container_width=True, hide_index=True)
                else:
                    st.info("Введите значения анализов и антропометрии, чтобы увидеть сравнение с нормами.")

                # 5. Процентили пациента относительно загруженной выборки
                st.subheader("5. Процентили пациента относительно загруженной выборки")
                st.caption("Если загружен датасет, можно оценить, в каком процентиле находится пациент по сопоставимым показателям.")

                if analysis_df is not None and analysis_df.shape[0] > 0:
                    # Сопоставление имён колонок загруженного датасета с нашими показателями
                    synonyms = {
                        'age': ['age', 'возраст', 'years', 'лет'],
                        'bmi': ['bmi', 'имт', 'body mass index'],
                        'glucose': ['glucose', 'глюкоза', 'fasting glucose', 'глюкоза натощак'],
                        'hba1c': ['hba1c', 'гликированный гемоглобин', 'a1c'],
                        'tsh': ['tsh', 'ттг', 'тиреотропный гормон'],
                        'ft4': ['ft4', 'св.t4', 'свободный t4', 'thyroxine'],
                        'calcium': ['calcium', 'кальций', 'ca'],
                        'waist': ['waist', 'талия', 'окружность талии'],
                        'sleep': ['sleep', 'сон', 'sleep hours'],
                        'height': ['height', 'рост'],
                        'weight': ['weight', 'вес'],
                    }

                    # Функция поиска колонки по синонимам
                    def find_column_by_synonyms(df, synonyms_list):
                        for syn in synonyms_list:
                            for col in df.columns:
                                if syn in col.lower():
                                    return col
                        return None

                    percentile_results = []
                    # Проверяем каждый показатель
                    patient_values = {
                        'age': age,
                        'bmi': bmi,
                        'glucose': fasting_glucose if fasting_glucose > 0 else None,
                        'hba1c': hba1c if hba1c > 0 else None,
                        'tsh': tsh_value if tsh_value > 0 else None,
                        'ft4': ft4_value if ft4_value > 0 else None,
                        'calcium': serum_calcium if serum_calcium > 0 else None,
                        'waist': waist_cm,
                        'sleep': sleep_hours,
                        'height': height_cm,
                        'weight': weight_kg,
                    }

                    for key, value in patient_values.items():
                        if value is None:
                            continue
                        col_name = find_column_by_synonyms(analysis_df, synonyms[key])
                        if col_name is None:
                            continue
                        # Преобразуем колонку к числу, если надо
                        try:
                            data_col = pd.to_numeric(analysis_df[col_name], errors='coerce').dropna()
                        except:
                            continue
                        if len(data_col) == 0:
                            continue
                        # Вычисляем процентиль
                        percentile = (data_col < value).mean() * 100
                        percentile_results.append({
                            "Показатель": key.capitalize(),
                            "Значение пациента": value,
                            "Колонка в датасете": col_name,
                            "Процентиль": f"{percentile:.1f}%",
                            "Интерпретация": "Ниже медианы" if percentile < 50 else "Выше медианы"
                        })

                    if percentile_results:
                        st.write("#### Процентили пациента в загруженной выборке")
                        perc_df = pd.DataFrame(percentile_results)
                        st.dataframe(perc_df, use_container_width=True, hide_index=True)
                        st.caption("Процентиль показывает, какой процент наблюдений в выборке имеет значение меньше, чем у пациента.")
                    else:
                        st.info("Не удалось сопоставить ни один показатель с колонками загруженного датасета. Проверьте названия колонок.")
                else:
                    st.info("💡 Загрузите датасет, чтобы рассчитать процентили пациента.")

    # ======================== ЭКСПАНДЕР 3: МУЛЬТИФРАКТАЛЬНЫЙ АНАЛИЗ ========================
    with st.expander("🧠 Мультифрактальный анализ данных", expanded=st.session_state['expander_states']['mfdfa']):
        st.subheader("🧠 Мультифрактальный анализ временных рядов (MF-DFA)")
        st.caption("Введите один или несколько временных рядов (например, глюкоза, вес, давление). Каждый ряд должен содержать не менее 12 чисел.")

        # Инициализация списка рядов
        if 'mfdfa_series_list' not in st.session_state:
            st.session_state['mfdfa_series_list'] = [{"name": "Глюкоза", "data": ""}]

        # Функция для добавления нового ряда
        def add_series():
            st.session_state['mfdfa_series_list'].append({"name": f"Ряд {len(st.session_state['mfdfa_series_list'])+1}", "data": ""})

        # Загрузка CSV файла
        st.markdown("#### 📁 Загрузка CSV файла с рядами")
        st.caption("Файл должен содержать колонки с числовыми данными. Каждая колонка будет рассматриваться как отдельный временной ряд.")

        mfdfa_file = st.file_uploader(
            "Выберите CSV файл с временными рядами",
            type=["csv"],
            key="mfdfa_file_uploader",
            help="CSV файл с колонками данных. Первая строка - заголовки."
        )

        if mfdfa_file is not None:
            try:
                # Читаем CSV файл
                mfdfa_df = pd.read_csv(mfdfa_file, sep=None, engine='python')
                
                # Если файл не разделился правильно, пробуем другие разделители
                if mfdfa_df.shape[1] == 1:
                    for sep in [';', '\t', '|', ',']:
                        try:
                            mfdfa_file.seek(0)
                            mfdfa_df = pd.read_csv(mfdfa_file, sep=sep, engine='python')
                            if mfdfa_df.shape[1] > 1:
                                st.info(f"Использован разделитель: '{sep}'")
                                break
                        except:
                            continue
                
                st.success(f"✅ Загружен файл: {mfdfa_df.shape[0]} строк, {mfdfa_df.shape[1]} колонок")
                
                # Показываем первые строки
                with st.expander("Просмотр загруженных данных"):
                    st.dataframe(mfdfa_df.head(10))
                
                # Определяем числовые колонки
                numeric_cols = []
                for col in mfdfa_df.columns:
                    # Пробуем преобразовать в числа
                    try:
                        pd.to_numeric(mfdfa_df[col].astype(str).str.replace(',', '.'), errors='raise')
                        numeric_cols.append(col)
                    except:
                        try:
                            pd.to_numeric(mfdfa_df[col], errors='raise')
                            numeric_cols.append(col)
                        except:
                            pass
                
                if numeric_cols:
                    st.info(f"Найдено числовых колонок: {len(numeric_cols)}")
                    
                    # Мультивыбор колонок
                    selected_cols = st.multiselect(
                        "Выберите колонки для анализа",
                        options=numeric_cols,
                        default=numeric_cols[:min(3, len(numeric_cols))] if len(numeric_cols) > 0 else [],
                        key="mfdfa_cols_select"
                    )
                    
                    if selected_cols:
                        # Кнопка для загрузки выбранных колонок
                        if st.button("📥 Загрузить выбранные колонки в анализ", key="load_mfdfa_cols", type="primary"):
                            # Очищаем текущий список
                            st.session_state['mfdfa_series_list'] = []
                            
                            # Добавляем выбранные колонки
                            for col in selected_cols:
                                # Преобразуем в числа
                                try:
                                    series_data = pd.to_numeric(
                                        mfdfa_df[col].astype(str).str.replace(',', '.'), 
                                        errors='coerce'
                                    )
                                except:
                                    series_data = pd.to_numeric(mfdfa_df[col], errors='coerce')
                                
                                # Убираем NaN
                                series_data = series_data.dropna()
                                
                                if len(series_data) >= 12:
                                    # Преобразуем в строку для text_area
                                    data_str = ', '.join(series_data.astype(str).tolist())
                                    st.session_state['mfdfa_series_list'].append({
                                        "name": col,
                                        "data": data_str
                                    })
                                    st.success(f"✅ Колонка '{col}': {len(series_data)} значений загружено")
                                else:
                                    st.warning(f"⚠️ Колонка '{col}' пропущена: только {len(series_data)} значений (нужно ≥12)")
                            
                            # Очищаем file_uploader
                            st.session_state['mfdfa_file_uploader'] = None
                            st.rerun()
                    else:
                        st.info("Выберите хотя бы одну колонку для анализа")
                else:
                    st.warning("В файле не найдены числовые колонки")
                    
            except Exception as e:
                st.error(f"Ошибка при чтении файла: {e}")
                st.info("Проверьте формат файла. Убедитесь, что:")
                st.info("1. Файл в формате CSV")
                st.info("2. Содержит числовые колонки")
                st.info("3. Числовые значения используют точку или запятую как десятичный разделитель")

        st.markdown("---")
        st.markdown("#### ✍️ Ручной ввод рядов")

        # Кнопка добавления ряда вручную
        col_add, _ = st.columns([1, 3])
        with col_add:
            st.button("➕ Добавить ряд вручную", on_click=add_series, key="add_series_button")

        # Отображение полей для ввода
        for i, series_item in enumerate(st.session_state['mfdfa_series_list']):
            col_name, col_data = st.columns([1, 3])
            with col_name:
                series_item['name'] = st.text_input(
                    f"Название ряда {i+1}", 
                    value=series_item['name'], 
                    key=f"mfdfa_name_{i}"
                )
            with col_data:
                series_item['data'] = st.text_area(
                    f"Данные {series_item['name']}",
                    value=series_item['data'],
                    height=80,
                    placeholder="92, 95, 90, 101, ...",
                    key=f"mfdfa_data_{i}"
                )
            # Кнопка удаления (кроме первого)
            if i > 0:
                if st.button(f"🗑️ Удалить {series_item['name']}", key=f"del_mfdfa_{i}"):
                    st.session_state['mfdfa_series_list'].pop(i)
                    st.rerun()

        # Кнопка анализа
        st.markdown("---")
        if st.button("🔍 Анализировать ряды", key="analyze_mfdfa", type="primary"):
            valid_series = []
            for item in st.session_state['mfdfa_series_list']:
                arr = parse_series(item['data'])
                if arr is not None and len(arr) >= 12:
                    valid_series.append({"name": item['name'], "data": arr})
                else:
                    if item['data'].strip():
                        st.warning(f"Ряд '{item['name']}' пропущен: недостаточно данных (нужно ≥12 чисел).")
            
            if not valid_series:
                if st.session_state.get('mfdfa_results'):
                    valid_series = st.session_state['mfdfa_results']
                    st.info("Используются сохраненные результаты анализа.")
                else:
                    st.error("Нет валидных рядов для анализа.")
            
            if valid_series:
                # Сохраняем результаты
                st.session_state['mfdfa_results'] = valid_series
                
                # Рассчитываем MF-DFA для каждого ряда
                results = {}
                for series in valid_series:
                    res = mfdfa(series['data'])
                    if res is not None:
                        results[series['name']] = res
                        st.markdown(f"#### Ряд: {series['name']}")
                        st.write(mfdfa_interpretation(res))
                        comparison = compare_to_reference(res["width"])
                        st.info(f"Сравнение с эталоном (ширина спектра {comparison['reference']:.2f}): {comparison['status']}. Отклонение {comparison['delta']:+.3f}.")
                        st.metric("Интерпретация по ширине спектра", interpret_complexity(res["width"]))
                        sd = np.std(series['data'])
                        cv = sd / np.mean(series['data']) * 100
                        st.write(f"**Среднее:** {np.mean(series['data']):.1f}, **SD:** {sd:.1f}, **CV:** {cv:.1f}%")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            fig1 = plot_mfdfa_scaling(res)
                            if fig1 is not None:
                                st.pyplot(fig1, clear_figure=True, use_container_width=True)
                                plt.close(fig1)
                        with col2:
                            fig2 = plot_mfdfa_spectrum(res)
                            if fig2 is not None:
                                st.pyplot(fig2, clear_figure=True, use_container_width=True)
                                plt.close(fig2)
                    else:
                        st.warning(f"Не удалось рассчитать MF-DFA для ряда '{series['name']}'.")
                
                # Сравнительный график
                if len(results) >= 2:
                    st.markdown("#### Сравнение мультифрактальных спектров")
                    fig_comp, ax_comp = plt.subplots(figsize=(8, 5))
                    for name, res in results.items():
                        alpha = res["alpha"]
                        f_alpha = res["f_alpha"]
                        valid = np.isfinite(alpha) & np.isfinite(f_alpha)
                        if valid.sum() >= 2:
                            ax_comp.plot(alpha[valid], f_alpha[valid], marker='o', label=name, linewidth=1.5)
                    ax_comp.set_xlabel("α")
                    ax_comp.set_ylabel("f(α)")
                    ax_comp.set_title("Сравнение мультифрактальных спектров")
                    ax_comp.legend()
                    ax_comp.grid(True, alpha=0.25)
                    st.pyplot(fig_comp, clear_figure=True, use_container_width=True)
                    plt.close(fig_comp)

    # ======================== PDF ОТЧЕТ ========================

    st.markdown("---")
    st.subheader("📄 Создание PDF отчета")
    st.caption("Создайте подробный PDF отчет со всеми данными пациента.")

    if st.button("📄 Создать PDF отчет по данным", key="create_data_pdf_button"):
        try:
            # Импортируем необходимые библиотеки для создания PDF
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch, cm
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            import io
            import os
            import tempfile
            
            # Регистрируем шрифты с поддержкой кириллицы
            font_registered = False
            
            # Список возможных путей к шрифтам
            font_candidates = [
                ("DejaVuSans", "C:/Windows/Fonts/DejaVuSans.ttf"),
                ("DejaVuSans-Bold", "C:/Windows/Fonts/DejaVuSans-Bold.ttf"),
                ("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                ("DejaVuSans-Bold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
                ("DejaVuSans", "/Library/Fonts/DejaVuSans.ttf"),
                ("DejaVuSans-Bold", "/Library/Fonts/DejaVuSans-Bold.ttf"),
                ("Arial", "C:/Windows/Fonts/arial.ttf"),
                ("Arial-Bold", "C:/Windows/Fonts/arialbd.ttf"),
                ("TimesNewRoman", "C:/Windows/Fonts/times.ttf"),
                ("TimesNewRoman-Bold", "C:/Windows/Fonts/timesbd.ttf"),
            ]
            
            main_font = None
            bold_font = None
            
            regular_candidates = []
            bold_candidates = []
            
            for font_name, font_path in font_candidates:
                if os.path.exists(font_path):
                    if "Bold" in font_name or "bold" in font_name:
                        bold_candidates.append((font_name, font_path))
                    else:
                        regular_candidates.append((font_name, font_path))
            
            if regular_candidates:
                regular_font_name, regular_font_path = regular_candidates[0]
                try:
                    pdfmetrics.registerFont(TTFont(regular_font_name, regular_font_path))
                    main_font = regular_font_name
                    font_registered = True
                except:
                    pass
            
            if bold_candidates:
                bold_font_name, bold_font_path = bold_candidates[0]
                try:
                    pdfmetrics.registerFont(TTFont(bold_font_name, bold_font_path))
                    bold_font = bold_font_name
                except:
                    bold_font = main_font
            
            if not font_registered or main_font is None:
                try:
                    import matplotlib.font_manager as fm
                    font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
                    
                    for font_path in font_list:
                        try:
                            font_name = os.path.basename(font_path).replace('.ttf', '')
                            if any(cyrillic_font in font_name.lower() for cyrillic_font in ['dejavu', 'arial', 'times', 'liberation', 'ubuntu', 'noto']):
                                try:
                                    pdfmetrics.registerFont(TTFont(font_name, font_path))
                                    if main_font is None:
                                        main_font = font_name
                                        bold_font = font_name
                                        font_registered = True
                                        break
                                except:
                                    continue
                        except:
                            continue
                except:
                    pass
            
            if not font_registered or main_font is None:
                main_font = "Helvetica"
                bold_font = "Helvetica-Bold"
                st.warning("⚠️ Не найдены шрифты с поддержкой кириллицы. Русский текст может отображаться некорректно.")
            else:
                if bold_font is None:
                    bold_font = main_font
            
            # Создаем буфер для PDF
            pdf_buffer = io.BytesIO()
            
            # Создаем документ
            doc = SimpleDocTemplate(
                pdf_buffer,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            # Стили
            styles = getSampleStyleSheet()
            
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontName=bold_font,
                fontSize=20,
                textColor=colors.HexColor('#1a237e'),
                spaceAfter=10,
                alignment=TA_CENTER
            )
            
            section_style = ParagraphStyle(
                'CustomSection',
                parent=styles['Heading2'],
                fontName=bold_font,
                fontSize=16,
                textColor=colors.HexColor('#1a237e'),
                spaceAfter=8,
                spaceBefore=16
            )
            
            subsection_style = ParagraphStyle(
                'CustomSubsection',
                parent=styles['Heading3'],
                fontName=bold_font,
                fontSize=13,
                textColor=colors.HexColor('#333333'),
                spaceAfter=6,
                spaceBefore=12
            )
            
            body_style = ParagraphStyle(
                'CustomBody',
                parent=styles['Normal'],
                fontName=main_font,
                fontSize=10,
                leading=14,
                spaceAfter=4
            )
            
            # Временная директория для графиков
            temp_dir = tempfile.mkdtemp()
            
            # Собираем элементы для PDF
            elements = []
            
            # Заголовок
            elements.append(Paragraph("Отчет по анализу данных", title_style))
            elements.append(Paragraph(f"Дата создания: {pd.Timestamp.now().strftime('%d.%m.%Y %H:%M')}", body_style))
            elements.append(Spacer(1, 20))
            
            # 1. Информация о загруженном датасете
            elements.append(Paragraph("1. Информация о датасете", section_style))
            
            if analysis_df is not None and analysis_df.shape[0] > 0:
                elements.append(Paragraph(f"Загруженный датасет содержит:", body_style))
                elements.append(Paragraph(f"• Количество строк: {analysis_df.shape[0]}", body_style))
                elements.append(Paragraph(f"• Количество колонок: {analysis_df.shape[1]}", body_style))
                
                numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = analysis_df.select_dtypes(include=['object']).columns.tolist()
                
                elements.append(Paragraph(f"• Числовые колонки: {len(numeric_cols)}", body_style))
                elements.append(Paragraph(f"• Категориальные колонки: {len(categorical_cols)}", body_style))
                
                # Первые строки данных
                elements.append(Spacer(1, 10))
                elements.append(Paragraph("Первые 5 строк данных:", subsection_style))
                
                head_data = [analysis_df.columns.tolist()] + analysis_df.head(5).values.tolist()
                head_data = [[str(cell)[:15] for cell in row] for row in head_data]
                
                if len(head_data[0]) > 10:
                    head_data = [row[:10] for row in head_data]
                
                head_table = Table(head_data)
                head_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a237e')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), bold_font),
                    ('FONTSIZE', (0, 0), (-1, 0), 8),
                    ('FONTNAME', (0, 1), (-1, -1), main_font),
                    ('FONTSIZE', (0, 1), (-1, -1), 7),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f5f5f5')),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                elements.append(head_table)
                
                # График распределения данных
                if numeric_cols:
                    elements.append(Spacer(1, 15))
                    elements.append(Paragraph("Распределение первых числовых колонок:", subsection_style))
                    
                    # Создаем гистограммы для первых 4 числовых колонок
                    fig_hist, axes = plt.subplots(2, 2, figsize=(10, 8))
                    axes = axes.flatten()
                    
                    for idx, col in enumerate(numeric_cols[:4]):
                        axes[idx].hist(analysis_df[col].dropna(), bins=20, edgecolor='black', alpha=0.7)
                        axes[idx].set_title(f"Гистограмма: {col}", fontsize=10)
                        axes[idx].set_xlabel(col, fontsize=8)
                        axes[idx].set_ylabel("Частота", fontsize=8)
                        axes[idx].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    hist_path = os.path.join(temp_dir, "histograms.png")
                    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    hist_img = Image(hist_path, width=15*cm, height=12*cm)
                    elements.append(hist_img)
                    
            else:
                elements.append(Paragraph("Датасет не был загружен.", body_style))
            
            # 2. Описательная статистика
            elements.append(PageBreak())
            elements.append(Paragraph("2. Описательная статистика", section_style))
            
            if analysis_df is not None and analysis_df.shape[0] > 0:
                numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
                
                if numeric_cols:
                    desc_stats = analysis_df[numeric_cols].describe()
                    
                    if len(numeric_cols) > 10:
                        desc_stats = desc_stats.iloc[:, :10]
                    
                    stats_data = [["Метрика"] + desc_stats.columns.tolist()]
                    for idx, row in desc_stats.iterrows():
                        stats_data.append([idx] + [f"{val:.2f}" if isinstance(val, (int, float)) else str(val) for val in row])
                    
                    stats_table = Table(stats_data)
                    stats_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a237e')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), bold_font),
                        ('FONTSIZE', (0, 0), (-1, 0), 7),
                        ('FONTNAME', (0, 1), (-1, -1), main_font),
                        ('FONTSIZE', (0, 1), (-1, -1), 6),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f5f5f5')),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ]))
                    elements.append(stats_table)
                    
                    # Boxplot для числовых колонок
                    elements.append(Spacer(1, 15))
                    elements.append(Paragraph("Boxplot числовых колонок:", subsection_style))
                    
                    fig_box, ax_box = plt.subplots(figsize=(10, 6))
                    box_data = [analysis_df[col].dropna() for col in numeric_cols[:8]]
                    ax_box.boxplot(box_data, labels=numeric_cols[:8], vert=False)
                    ax_box.set_title("Boxplot числовых колонок", fontsize=12)
                    ax_box.grid(True, alpha=0.3, axis='x')
                    plt.tight_layout()
                    
                    boxplot_path = os.path.join(temp_dir, "boxplot.png")
                    plt.savefig(boxplot_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    boxplot_img = Image(boxplot_path, width=15*cm, height=10*cm)
                    elements.append(boxplot_img)
                    
            else:
                elements.append(Paragraph("Датасет не был загружен.", body_style))
            
            # 3. Корреляционный анализ
            elements.append(PageBreak())
            elements.append(Paragraph("3. Корреляционный анализ", section_style))
            
            if analysis_df is not None and analysis_df.shape[0] >= 5:
                numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
                
                if len(numeric_cols) >= 2:
                    if len(numeric_cols) > 12:
                        numeric_cols = numeric_cols[:12]
                    
                    corr_matrix = analysis_df[numeric_cols].corr()
                    
                    # Тепловая карта корреляции
                    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
                    im = ax_corr.imshow(corr_matrix.values, cmap='coolwarm', vmin=-1, vmax=1)
                    ax_corr.set_xticks(range(len(corr_matrix.columns)))
                    ax_corr.set_yticks(range(len(corr_matrix.columns)))
                    ax_corr.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=8)
                    ax_corr.set_yticklabels(corr_matrix.columns, fontsize=8)
                    ax_corr.set_title("Корреляционная матрица", fontsize=12)
                    
                    for i in range(len(corr_matrix.columns)):
                        for j in range(len(corr_matrix.columns)):
                            ax_corr.text(j, i, f"{corr_matrix.values[i, j]:.2f}",
                                        ha="center", va="center", fontsize=6,
                                        color="white" if abs(corr_matrix.values[i, j]) > 0.5 else "black")
                    
                    plt.colorbar(im, ax=ax_corr)
                    plt.tight_layout()
                    
                    corr_path = os.path.join(temp_dir, "correlation.png")
                    plt.savefig(corr_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    corr_img = Image(corr_path, width=15*cm, height=12*cm)
                    elements.append(corr_img)
                    
                    # Сильные корреляции
                    strong_corr = []
                    for i in range(len(numeric_cols)):
                        for j in range(i+1, len(numeric_cols)):
                            if abs(corr_matrix.iloc[i, j]) > 0.7:
                                strong_corr.append(f"• {numeric_cols[i]} ↔ {numeric_cols[j]}: {corr_matrix.iloc[i, j]:.2f}")
                    
                    if strong_corr:
                        elements.append(Spacer(1, 10))
                        elements.append(Paragraph("Сильные корреляции (|r| > 0.7):", subsection_style))
                        for corr in strong_corr:
                            elements.append(Paragraph(corr, body_style))
                else:
                    elements.append(Paragraph("Недостаточно числовых колонок для корреляционного анализа.", body_style))
            else:
                elements.append(Paragraph("Недостаточно данных для корреляционного анализа.", body_style))
            
            # 4. Выбросы
            elements.append(Paragraph("4. Анализ выбросов", section_style))
            
            if analysis_df is not None and analysis_df.shape[0] > 0:
                numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
                
                if numeric_cols:
                    outlier_details = []
                    for col in numeric_cols:
                        Q1 = analysis_df[col].quantile(0.25)
                        Q3 = analysis_df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        col_outliers = analysis_df[(analysis_df[col] < lower_bound) | (analysis_df[col] > upper_bound)]
                        
                        if not col_outliers.empty:
                            outlier_details.append([col, len(col_outliers), f"{lower_bound:.2f}", f"{upper_bound:.2f}"])
                    
                    if outlier_details:
                        outlier_data = [["Признак", "Кол-во выбросов", "Нижняя граница", "Верхняя граница"]] + outlier_details
                        outlier_table = Table(outlier_data, colWidths=[4*cm, 3*cm, 3*cm, 3*cm])
                        outlier_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a237e')),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, 0), bold_font),
                            ('FONTSIZE', (0, 0), (-1, 0), 9),
                            ('FONTNAME', (0, 1), (-1, -1), main_font),
                            ('FONTSIZE', (0, 1), (-1, -1), 8),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f5f5f5')),
                            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ]))
                        elements.append(outlier_table)
                    else:
                        elements.append(Paragraph("Выбросы не обнаружены.", body_style))
            else:
                elements.append(Paragraph("Датасет не был загружен.", body_style))
            
            # 5. Проверка нормальности
            elements.append(PageBreak())
            elements.append(Paragraph("5. Проверка нормальности распределения", section_style))
            
            if analysis_df is not None and analysis_df.shape[0] >= 3:
                numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
                
                if numeric_cols:
                    from scipy.stats import shapiro
                    
                    normality_results = []
                    for col in numeric_cols:
                        data = analysis_df[col].dropna()
                        if len(data) >= 3:
                            stat, p = shapiro(data)
                            normality_results.append([col, f"{p:.4f}", "Нормальное" if p > 0.05 else "Отличается"])
                    
                    if normality_results:
                        norm_data = [["Признак", "p-value", "Распределение"]] + normality_results
                        norm_table = Table(norm_data, colWidths=[5*cm, 3*cm, 4*cm])
                        norm_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a237e')),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, 0), bold_font),
                            ('FONTSIZE', (0, 0), (-1, 0), 9),
                            ('FONTNAME', (0, 1), (-1, -1), main_font),
                            ('FONTSIZE', (0, 1), (-1, -1), 8),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f5f5f5')),
                            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ]))
                        elements.append(norm_table)
                        
                        # QQ plot для первой числовой колонки
                        if len(numeric_cols) > 0:
                            from scipy import stats as scipy_stats
                            
                            elements.append(Spacer(1, 15))
                            elements.append(Paragraph(f"QQ plot для колонки '{numeric_cols[0]}':", subsection_style))
                            
                            fig_qq, ax_qq = plt.subplots(figsize=(8, 6))
                            scipy_stats.probplot(analysis_df[numeric_cols[0]].dropna(), dist="norm", plot=ax_qq)
                            ax_qq.set_title(f"QQ plot: {numeric_cols[0]}", fontsize=12)
                            ax_qq.grid(True, alpha=0.3)
                            plt.tight_layout()
                            
                            qq_path = os.path.join(temp_dir, "qq_plot.png")
                            plt.savefig(qq_path, dpi=150, bbox_inches='tight')
                            plt.close()
                            
                            qq_img = Image(qq_path, width=12*cm, height=9*cm)
                            elements.append(qq_img)
                else:
                    elements.append(Paragraph("В датасете нет числовых колонок.", body_style))
            else:
                elements.append(Paragraph("Недостаточно данных для проверки нормальности.", body_style))
            
            # 6. MF-DFA анализ
            elements.append(Paragraph("6. Мультифрактальный анализ (MF-DFA)", section_style))
            
            if st.session_state.get('mfdfa_results'):
                mfdfa_results = st.session_state['mfdfa_results']
                
                for series_result in mfdfa_results:
                    elements.append(Paragraph(f"Ряд: {series_result['name']}", subsection_style))
                    
                    res = mfdfa(series_result['data'])
                    if res is not None:
                        width = res['width']
                        mean_h = res['mean_h']
                        
                        elements.append(Paragraph(f"• Ширина спектра: {width:.3f}", body_style))
                        elements.append(Paragraph(f"• Средний H(q): {mean_h:.3f}", body_style))
                        elements.append(Paragraph(f"• Интерпретация: {interpret_complexity(width)}", body_style))
                        
                        comparison = compare_to_reference(width)
                        elements.append(Paragraph(f"• Сравнение с эталоном: {comparison['status']}", body_style))
                        
                        data = series_result['data']
                        elements.append(Paragraph(f"• Среднее: {np.mean(data):.2f}", body_style))
                        elements.append(Paragraph(f"• Стандартное отклонение: {np.std(data):.2f}", body_style))
                        elements.append(Paragraph(f"• Коэффициент вариации: {np.std(data) / np.mean(data) * 100:.1f}%", body_style))
                        
                        # График временного ряда
                        elements.append(Spacer(1, 10))
                        fig_series, ax_series = plt.subplots(figsize=(10, 4))
                        ax_series.plot(data, marker='o', markersize=3, linewidth=1)
                        ax_series.set_title(f"Временной ряд: {series_result['name']}", fontsize=12)
                        ax_series.set_xlabel("Индекс", fontsize=10)
                        ax_series.set_ylabel("Значение", fontsize=10)
                        ax_series.grid(True, alpha=0.3)
                        plt.tight_layout()
                        
                        series_path = os.path.join(temp_dir, f"series_{series_result['name']}.png")
                        plt.savefig(series_path, dpi=150, bbox_inches='tight')
                        plt.close()
                        
                        series_img = Image(series_path, width=15*cm, height=6*cm)
                        elements.append(series_img)
                        
                        # Графики MF-DFA
                        fig_mfdfa1 = plot_mfdfa_scaling(res)
                        if fig_mfdfa1 is not None:
                            mfdfa1_path = os.path.join(temp_dir, f"mfdfa_scaling_{series_result['name']}.png")
                            fig_mfdfa1.savefig(mfdfa1_path, dpi=150, bbox_inches='tight')
                            plt.close(fig_mfdfa1)
                            
                            mfdfa1_img = Image(mfdfa1_path, width=12*cm, height=9*cm)
                            elements.append(mfdfa1_img)
                        
                        fig_mfdfa2 = plot_mfdfa_spectrum(res)
                        if fig_mfdfa2 is not None:
                            mfdfa2_path = os.path.join(temp_dir, f"mfdfa_spectrum_{series_result['name']}.png")
                            fig_mfdfa2.savefig(mfdfa2_path, dpi=150, bbox_inches='tight')
                            plt.close(fig_mfdfa2)
                            
                            mfdfa2_img = Image(mfdfa2_path, width=12*cm, height=9*cm)
                            elements.append(mfdfa2_img)
                        
                        elements.append(Spacer(1, 8))
            else:
                elements.append(Paragraph("MF-DFA анализ не выполнялся.", body_style))
            
            # 7. Результаты ML моделей
            if st.session_state.get('model_trained', False):
                elements.append(PageBreak())
                elements.append(Paragraph("7. Результаты обучения ML моделей", section_style))
                
                if 'metrics_df' in st.session_state:
                    metrics_df = st.session_state['metrics_df']
                    
                    elements.append(Paragraph("Метрики качества моделей:", body_style))
                    
                    metrics_data = [["Модель"] + metrics_df.columns.tolist()]
                    for idx, row in metrics_df.iterrows():
                        metrics_data.append([idx] + [f"{val:.4f}" for val in row])
                    
                    metrics_table = Table(metrics_data)
                    metrics_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a237e')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), bold_font),
                        ('FONTSIZE', (0, 0), (-1, 0), 7),
                        ('FONTNAME', (0, 1), (-1, -1), main_font),
                        ('FONTSIZE', (0, 1), (-1, -1), 6),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f5f5f5')),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ]))
                    elements.append(metrics_table)
                    
                    # График сравнения метрик
                    elements.append(Spacer(1, 15))
                    elements.append(Paragraph("Сравнение метрик моделей:", subsection_style))
                    
                    fig_metrics, ax_metrics = plt.subplots(figsize=(10, 6))
                    metrics_to_plot = ['ROC-AUC', 'PR-AUC', 'F1', 'MCC']
                    metrics_to_plot = [m for m in metrics_to_plot if m in metrics_df.columns]
                    
                    x = range(len(metrics_df.index))
                    width = 0.8 / len(metrics_to_plot)
                    
                    for i, metric in enumerate(metrics_to_plot):
                        ax_metrics.bar([xi + i*width for xi in x], metrics_df[metric], width, label=metric)
                    
                    ax_metrics.set_xticks([xi + width*(len(metrics_to_plot)-1)/2 for xi in x])
                    ax_metrics.set_xticklabels(metrics_df.index, fontsize=9)
                    ax_metrics.set_ylabel("Значение метрики")
                    ax_metrics.set_title("Сравнение метрик моделей")
                    ax_metrics.legend()
                    ax_metrics.grid(True, alpha=0.3, axis='y')
                    plt.tight_layout()
                    
                    metrics_path = os.path.join(temp_dir, "metrics_comparison.png")
                    plt.savefig(metrics_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    metrics_img = Image(metrics_path, width=15*cm, height=9*cm)
                    elements.append(metrics_img)
                    
                    # Калибровочная кривая для лучшей модели
                    if 'best_metric' in st.session_state and 'trained_models' in st.session_state:
                        best_metric = st.session_state['best_metric']
                        best_model_name = metrics_df[best_metric].idxmax()
                        
                        if best_model_name in st.session_state['trained_models']:
                            best_model = st.session_state['trained_models'][best_model_name]
                            
                            # Создаем калибровочную кривую
                            if hasattr(best_model, 'predict_proba'):
                                # Используем тестовые данные из session_state
                                if 'X_test_scaled' in st.session_state and 'y_test' in st.session_state:
                                    X_test_scaled = st.session_state.get('X_test_scaled')
                                    y_test = st.session_state.get('y_test')
                                    
                                    if X_test_scaled is not None and y_test is not None:
                                        y_proba = best_model.predict_proba(X_test_scaled)[:, 1]
                                        
                                        fig_cal, ax_cal = plt.subplots(figsize=(8, 6))
                                        
                                        from sklearn.calibration import calibration_curve
                                        fraction_of_positives, mean_predicted_value = calibration_curve(
                                            y_test, y_proba, n_bins=10
                                        )
                                        
                                        ax_cal.plot(mean_predicted_value, fraction_of_positives, "s-", label=best_model_name)
                                        ax_cal.plot([0, 1], [0, 1], "k--", label="Идеальная калибровка")
                                        ax_cal.set_xlabel("Средняя предсказанная вероятность")
                                        ax_cal.set_ylabel("Доля положительных исходов")
                                        ax_cal.set_title(f"Калибровочная кривая: {best_model_name}")
                                        ax_cal.legend()
                                        ax_cal.grid(True, alpha=0.3)
                                        plt.tight_layout()
                                        
                                        cal_path = os.path.join(temp_dir, "calibration_curve.png")
                                        plt.savefig(cal_path, dpi=150, bbox_inches='tight')
                                        plt.close()
                                        
                                        cal_img = Image(cal_path, width=12*cm, height=9*cm)
                                        elements.append(cal_img)
            
            # Добавляем предупреждение
            elements.append(Spacer(1, 20))
            elements.append(Paragraph(
                "<i>Данный отчет создан автоматически и носит информационный характер. "
                "Результаты анализа данных требуют профессиональной интерпретации.</i>",
                body_style
            ))
            
            # Создаем PDF
            doc.build(elements)
            
            # Получаем байты PDF
            pdf_bytes = pdf_buffer.getvalue()
            pdf_buffer.close()
            
            # Очищаем временные файлы
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            # Сохраняем в session_state
            st.session_state['data_pdf_report'] = pdf_bytes
            
            st.success("✅ PDF отчет по данным успешно создан!")
            
            # Кнопка для скачивания PDF
            st.download_button(
                "📥 Скачать PDF отчет по данным",
                data=pdf_bytes,
                file_name=f"data_analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                key="download_data_pdf_button"
            )
            
        except ImportError:
            st.error("❌ Библиотека reportlab не установлена. Установите её с помощью: pip install reportlab")
        except Exception as e:
            st.error(f"❌ Ошибка при создании PDF: {e}")
            st.info("Попробуйте перезапустить приложение и повторить попытку.")


# ======================== ПОДВАЛ ========================
st.markdown("---")
st.caption(
    "Прототип создан в образовательных целях. Диагностические решения и назначения должен подтверждать врач. "
    "MF-DFA блок является экспериментальным исследовательским модулем; ряд можно вводить вручную или загружать файлом."
)

