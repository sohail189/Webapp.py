"""
app.py — Production-ready Streamlit Disease Prediction App
Heart Disease & Liver Disease Risk Prediction with EDA & Fuzzy Logic

Author  : Muhammad Sohail
Version : 2.0.0
"""

# ─────────────────────────────────────────────
# 1. STANDARD IMPORTS
# ─────────────────────────────────────────────
import os
import pickle
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import streamlit as st
from streamlit_option_menu import option_menu

# ─────────────────────────────────────────────
# 2. LOGGING CONFIGURATION
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 3. CONSTANTS
# ─────────────────────────────────────────────
PAGE_TITLE  = "Disease Prediction System"
PAGE_ICON   = "🏥"
LAYOUT      = "wide"

HEART_MODEL_PATH = "heart_disease_model (3).sav"
LIVER_MODEL_PATH = "liver_disease_model (2).sav"
HEART_DATA_PATH  = "heart (1).csv"
LIVER_DATA_PATH  = "Liver_disease_data.csv"

RISK_HIGH_THRESHOLD   = 0.70
RISK_MEDIUM_THRESHOLD = 0.40

SIDEBAR_PAGES = [
    "Combined Risk Prediction",
    "Heart Disease Prediction",
    "Liver Disease Prediction",
    "Exploratory Data Analysis",
    "Plots and Charts",
    "Histogram Marker",
]

SIDEBAR_ICONS = [
    "heart-pulse",
    "heart",
    "activity",
    "bar-chart-line",
    "bar-chart",
    "graph-up",
]

# ─────────────────────────────────────────────
# 4. PAGE CONFIGURATION  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
)

# ─────────────────────────────────────────────
# 5. HELPER UTILITIES
# ─────────────────────────────────────────────
def get_risk_label(probability: float) -> str:
    """Return an emoji-tagged risk label from a raw probability (0–1)."""
    if probability >= RISK_HIGH_THRESHOLD:
        return "High Risk 🚨"
    if probability >= RISK_MEDIUM_THRESHOLD:
        return "Moderate Risk ⚠️"
    return "Low Risk ✅"


def get_risk_label_from_score(score: float) -> str:
    """Return an emoji-tagged risk label from a 0–100 risk score."""
    if score >= 70:
        return "High 🚨"
    if score >= 40:
        return "Moderate ⚠️"
    return "Low ✅"


# ─────────────────────────────────────────────
# 6. MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ML models…")
def load_model(model_path: str):
    """
    Load a pickled scikit-learn model from disk.
    Returns None and shows an st.error on failure.
    """
    if not os.path.exists(model_path):
        st.error(
            f"Model file not found: **{model_path}**  \n"
            "Please ensure it lives in the same directory as app.py."
        )
        logger.error("Model file missing: %s", model_path)
        return None
    try:
        with open(model_path, "rb") as fh:
            model = pickle.load(fh)
        logger.info("Model loaded successfully: %s", model_path)
        return model
    except Exception as exc:
        st.error(f"Failed to load model `{model_path}`: {exc}")
        logger.exception("Error loading model %s", model_path)
        return None


heart_model = load_model(HEART_MODEL_PATH)
liver_model = load_model(LIVER_MODEL_PATH)


# ─────────────────────────────────────────────
# 7. DATASET LOADING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_dataset(data_path: str) -> pd.DataFrame | None:
    """
    Load a CSV dataset into a DataFrame.
    Returns None and shows an st.error on failure.
    """
    if not os.path.exists(data_path):
        st.error(f"Dataset not found: **{data_path}**")
        logger.error("Dataset missing: %s", data_path)
        return None
    try:
        df = pd.read_csv(data_path)
        logger.info("Dataset loaded: %s  (%d rows, %d cols)", data_path, *df.shape)
        return df
    except Exception as exc:
        st.error(f"Failed to read dataset `{data_path}`: {exc}")
        logger.exception("Error reading dataset %s", data_path)
        return None


# ─────────────────────────────────────────────
# 8. FUZZY LOGIC SYSTEM
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Initialising fuzzy logic engine…")
def build_fuzzy_simulator() -> ctrl.ControlSystemSimulation:
    """
    Build and return a cached fuzzy logic control system that
    combines heart and liver disease probabilities into a single
    overall risk score (0–100).
    """
    heart_prob = ctrl.Antecedent(np.arange(0, 101, 1), "heart_prob")
    liver_prob = ctrl.Antecedent(np.arange(0, 101, 1), "liver_prob")
    risk       = ctrl.Consequent(np.arange(0, 101, 1), "risk")

    heart_prob.automf(3, variable_type="quant")
    liver_prob.automf(3, variable_type="quant")

    risk["low"]    = fuzz.trimf(risk.universe, [0,   0,  40])
    risk["medium"] = fuzz.trimf(risk.universe, [30, 50,  70])
    risk["high"]   = fuzz.trimf(risk.universe, [60, 100, 100])

    rules = [
        ctrl.Rule(heart_prob["high"]    | liver_prob["high"],    risk["high"]),
        ctrl.Rule(heart_prob["average"] | liver_prob["average"], risk["medium"]),
        ctrl.Rule(heart_prob["low"]     & liver_prob["low"],     risk["low"]),
    ]

    system = ctrl.ControlSystem(rules)
    logger.info("Fuzzy logic system initialised.")
    return ctrl.ControlSystemSimulation(system)


fuzzy_sim = build_fuzzy_simulator()


# ─────────────────────────────────────────────
# 9. PREDICTION HELPERS
# ─────────────────────────────────────────────
def run_heart_prediction(user_input: list):
    """Return (prediction, probability) tuple or raise on error."""
    prediction  = heart_model.predict([user_input])[0]
    probability = heart_model.predict_proba([user_input])[0][1]
    return prediction, probability


def run_liver_prediction(user_input: list):
    """Return (prediction, probability) tuple or raise on error."""
    prediction  = liver_model.predict([user_input])[0]
    probability = liver_model.predict_proba([user_input])[0][1]
    return prediction, probability


def run_fuzzy_combined(heart_pct: float, liver_pct: float) -> float:
    """Feed percentages (0–100) into the fuzzy sim and return risk score."""
    fuzzy_sim.input["heart_prob"] = heart_pct
    fuzzy_sim.input["liver_prob"] = liver_pct
    fuzzy_sim.compute()
    return fuzzy_sim.output["risk"]


# ─────────────────────────────────────────────
# 10. SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    selected = option_menu(
        menu_title="Multiple Disease Prediction System",
        options=SIDEBAR_PAGES,
        icons=SIDEBAR_ICONS,
        menu_icon="hospital-fill",
        default_index=0,
    )


# ─────────────────────────────────────────────
# 11. PAGE: COMBINED RISK PREDICTION
# ─────────────────────────────────────────────
def page_combined_risk():
    st.title("❤️🫀 Combined Heart & Liver Risk Prediction")
    st.caption(
        "Uses a **Fuzzy Logic** engine to merge individual disease probabilities "
        "into a single overall risk score."
    )
    st.divider()

    if not heart_model or not liver_model:
        st.error("Both models must be loaded before running combined prediction.")
        return

    with st.form("fuzzy_form"):
        st.subheader("Patient Details")
        c1, c2, c3 = st.columns(3)

        with c1:
            age          = st.number_input("Age",              min_value=1,   max_value=120, value=30)
            sex          = st.selectbox("Sex (Heart)",         ["Male", "Female"])
            cp           = st.number_input("Chest Pain Type (0–3)", 0, 3, 0)
            chol         = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)

        with c2:
            thalach      = st.number_input("Max Heart Rate",   60, 220, 150)
            bmi          = st.number_input("BMI",              10.0, 50.0, 24.0)
            alcohol      = st.number_input("Alcohol Consumption (units/week)", 0.0, 10.0, 1.0)

        with c3:
            gender_liver = st.selectbox("Gender (Liver)",      ["Male", "Female"])
            liver_fn     = st.number_input("Liver Function Test", 0.0, 100.0, 40.0)

        submitted = st.form_submit_button("🔍 Predict Combined Risk", use_container_width=True)

    if submitted:
        sex_num    = 1 if sex == "Male" else 0
        gender_num = 1 if gender_liver == "Male" else 0

        heart_input = [age, sex_num, cp, 120, chol, 0, 1, thalach, 0, 0.0, 1]
        liver_input = [age, gender_num, bmi, alcohol, 0, 0, 2.0, 0, 0, liver_fn, 0]

        try:
            heart_pct = heart_model.predict_proba([heart_input])[0][1] * 100
            liver_pct = liver_model.predict_proba([liver_input])[0][1] * 100
            risk_score = run_fuzzy_combined(heart_pct, liver_pct)

            st.divider()
            st.subheader("Results")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Heart Disease Probability", f"{heart_pct:.1f}%")
            col_b.metric("Liver Disease Probability", f"{liver_pct:.1f}%")
            col_c.metric("Combined Risk Score",       f"{risk_score:.1f}%")

            st.info(f"**Overall Risk Level:** {get_risk_label_from_score(risk_score)}")

        except Exception as exc:
            st.error(f"Prediction error: {exc}")
            logger.exception("Combined prediction failed.")


# ─────────────────────────────────────────────
# 12. PAGE: HEART DISEASE PREDICTION
# ─────────────────────────────────────────────
def page_heart_disease():
    st.title("❤️ Heart Disease Prediction")
    st.caption("Input 11 clinical features to receive a risk prediction.")
    st.divider()

    if not heart_model:
        st.error("Heart disease model is not loaded.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        age       = st.number_input("Age",                         1, 120, step=1)
        trestbps  = st.number_input("Resting Blood Pressure",      50, 250, step=1)
        restecg   = st.number_input("Resting ECG (0, 1, 2)",       0, 2, step=1)
        oldpeak   = st.number_input("ST Depression (Oldpeak)",     0.0, step=0.1)

    with c2:
        sex       = st.selectbox("Sex",                            ["Male", "Female"])
        cp        = st.number_input("Chest Pain Type (0–3)",       0, 3, step=1)
        chol      = st.number_input("Serum Cholesterol (mg/dL)",  100, 600, step=1)
        thalach   = st.number_input("Maximum Heart Rate",          60, 220, step=1)

    with c3:
        fbs       = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1], format_func=lambda x: "Yes" if x else "No")
        exang     = st.selectbox("Exercise Induced Angina",          [0, 1], format_func=lambda x: "Yes" if x else "No")
        slope     = st.number_input("Slope of Peak ST Segment (0–2)", 0, 2, step=1)

    if st.button("🔍 Predict Heart Disease", use_container_width=True):
        sex_num    = 1 if sex == "Male" else 0
        user_input = [age, sex_num, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope]
        try:
            prediction, probability = run_heart_prediction(user_input)
            st.divider()
            diagnosis = "⚠️ The patient **has** heart disease." if prediction == 1 else "✅ The patient does **not** have heart disease."
            st.subheader(diagnosis)
            st.metric("Probability of Heart Disease", f"{probability * 100:.1f}%")
            st.info(f"**Risk Level:** {get_risk_label(probability)}")
        except Exception as exc:
            st.error(f"Prediction error: {exc}")
            logger.exception("Heart prediction failed.")


# ─────────────────────────────────────────────
# 13. PAGE: LIVER DISEASE PREDICTION
# ─────────────────────────────────────────────
def page_liver_disease():
    st.title("🫀 Liver Disease Prediction")
    st.caption("Input patient liver-function data to receive a risk prediction.")
    st.divider()

    if not liver_model:
        st.error("Liver disease model is not loaded.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        age             = st.number_input("Age",                        1, 120, step=1)
        bmi             = st.number_input("BMI",                        10.0, step=0.1)
        alcohol         = st.number_input("Alcohol Consumption",        0.0, step=0.1)

    with c2:
        gender          = st.selectbox("Gender",                        ["Male", "Female"])
        smoking         = st.selectbox("Smoking",                       [0, 1], format_func=lambda x: "Yes" if x else "No")
        genetic_risk    = st.selectbox("Genetic Risk",                  [0, 1], format_func=lambda x: "Yes" if x else "No")

    with c3:
        physical_act    = st.number_input("Physical Activity (hrs/wk)", 0.0, step=0.1)
        diabetes        = st.selectbox("Diabetes",                      [0, 1], format_func=lambda x: "Yes" if x else "No")
        hypertension    = st.selectbox("Hypertension",                  [0, 1], format_func=lambda x: "Yes" if x else "No")
        liver_fn_test   = st.number_input("Liver Function Test",        0.0, step=0.1)

    diagnosis_code = st.selectbox("Diagnosis (0 = None, 1 = Mild, 2 = Severe)", [0, 1, 2])

    if st.button("🔍 Predict Liver Disease", use_container_width=True):
        gender_num = 1 if gender == "Male" else 0
        user_input = [age, gender_num, bmi, alcohol, smoking, genetic_risk,
                      physical_act, diabetes, hypertension, liver_fn_test, diagnosis_code]
        try:
            prediction, probability = run_liver_prediction(user_input)
            st.divider()
            diagnosis = "⚠️ The patient **has** liver disease." if prediction == 1 else "✅ The patient does **not** have liver disease."
            st.subheader(diagnosis)
            st.metric("Probability of Liver Disease", f"{probability * 100:.1f}%")
            st.info(f"**Risk Level:** {get_risk_label(probability)}")
        except Exception as exc:
            st.error(f"Prediction error: {exc}")
            logger.exception("Liver prediction failed.")


# ─────────────────────────────────────────────
# 14. PAGE: EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
def page_eda():
    st.title("📊 Exploratory Data Analysis")
    st.divider()

    dataset_label = st.selectbox("Select Dataset", ["Heart Disease", "Liver Disease"])
    data_path     = HEART_DATA_PATH if dataset_label == "Heart Disease" else LIVER_DATA_PATH
    df            = load_dataset(data_path)

    if df is None:
        return

    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)
    with c2:
        st.subheader("Missing Values")
        missing = df.isnull().sum().rename("Missing Count").to_frame()
        missing["% Missing"] = (missing["Missing Count"] / len(df) * 100).round(2)
        st.dataframe(missing, use_container_width=True)

    st.subheader("Data Types")
    st.dataframe(df.dtypes.rename("Dtype").to_frame(), use_container_width=True)


# ─────────────────────────────────────────────
# 15. PAGE: PLOTS AND CHARTS
# ─────────────────────────────────────────────
def page_plots():
    st.title("📈 Plots and Charts")
    st.divider()

    dataset_label = st.selectbox("Select Dataset", ["Heart Disease", "Liver Disease"])
    data_path     = HEART_DATA_PATH if dataset_label == "Heart Disease" else LIVER_DATA_PATH
    df            = load_dataset(data_path)

    if df is None:
        return

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if not numeric_cols:
        st.warning("No numeric columns found in the selected dataset.")
        return

    # --- Histogram ---
    st.subheader("Histogram")
    hist_col = st.selectbox("Column", numeric_cols, key="hist_col")
    fig, ax  = plt.subplots(figsize=(8, 4))
    sns.histplot(df[hist_col], kde=True, bins=30, ax=ax)
    ax.set_title(f"Histogram of {hist_col}")
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # --- Correlation Heatmap ---
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # --- Line Plot ---
    st.subheader("Line Plot")
    lc1, lc2 = st.columns(2)
    x_axis   = lc1.selectbox("X-axis", numeric_cols, key="line_x")
    y_axis   = lc2.selectbox("Y-axis", numeric_cols, key="line_y")
    fig, ax  = plt.subplots(figsize=(8, 4))
    sns.lineplot(x=df[x_axis], y=df[y_axis], marker="o", ax=ax)
    ax.set_title(f"{y_axis} vs {x_axis}")
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # --- Box Plot ---
    st.subheader("Box Plot")
    box_col = st.selectbox("Column", numeric_cols, key="box_col")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(x=df[box_col], ax=ax)
    ax.set_title(f"Box Plot of {box_col}")
    st.pyplot(fig)
    plt.close(fig)


# ─────────────────────────────────────────────
# 16. PAGE: HISTOGRAM MARKER
# ─────────────────────────────────────────────
def page_histogram_marker():
    st.title("📌 Histogram Marker Tool")
    st.caption("Mark your personal value on the dataset distribution to see where you stand.")
    st.divider()

    dataset_label = st.selectbox("Select Dataset", ["Heart Disease", "Liver Disease"], key="marker_ds")
    data_path     = HEART_DATA_PATH if dataset_label == "Heart Disease" else LIVER_DATA_PATH
    df            = load_dataset(data_path)

    if df is None:
        return

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if not numeric_cols:
        st.warning("No numeric columns found.")
        return

    col       = st.selectbox("Select Column", numeric_cols)
    col_mean  = float(df[col].mean())
    marker    = st.number_input(f"Mark a value on `{col}`", value=col_mean)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df[col], bins=30, kde=True, ax=ax, color="steelblue", alpha=0.7)
    ax.axvline(marker, color="red", linestyle="--", linewidth=2, label=f"Your Value: {marker:.2f}")
    ax.axvline(col_mean, color="green", linestyle=":", linewidth=1.5, label=f"Mean: {col_mean:.2f}")
    ax.set_title(f"Distribution of {col}")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    percentile = (df[col] < marker).mean() * 100
    st.info(f"Your marked value **{marker:.2f}** is higher than **{percentile:.1f}%** of the population in this dataset.")


# ─────────────────────────────────────────────
# 17. FOOTER
# ─────────────────────────────────────────────
def render_footer():
    st.divider()
    st.caption(
        "**Disclaimer:** This application is for educational purposes only and is **not** a substitute "
        "for professional medical advice. Always consult a qualified healthcare provider."
    )


# ─────────────────────────────────────────────
# 18. ROUTER — dispatch to correct page
# ─────────────────────────────────────────────
PAGE_DISPATCH = {
    "Combined Risk Prediction":  page_combined_risk,
    "Heart Disease Prediction":  page_heart_disease,
    "Liver Disease Prediction":  page_liver_disease,
    "Exploratory Data Analysis": page_eda,
    "Plots and Charts":          page_plots,
    "Histogram Marker":          page_histogram_marker,
}

PAGE_DISPATCH[selected]()
render_footer()
