# disease_prediction_app.py
import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Liver and Heart Disease Prediction",
    layout="wide",
    page_icon="🏥"
)

# --- MODEL AND DATA PATHS ---
HEART_MODEL = 'heart_disease_model (3).sav'
LIVER_MODEL = 'liver_disease_model (2).sav'
HEART_DATA = 'heart (1).csv'
LIVER_DATA = 'Liver_disease_data.csv'

# --- MODEL LOADING ---
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Model not found: {path}")
        return None
    with open(path, 'rb') as file:
        return pickle.load(file)

heart_model = load_model(HEART_MODEL)
liver_model = load_model(LIVER_MODEL)

# --- FUZZY SYSTEM ---
@st.cache_resource
def setup_fuzzy():
    heart_prob = ctrl.Antecedent(np.arange(0, 101, 1), 'heart_prob')
    liver_prob = ctrl.Antecedent(np.arange(0, 101, 1), 'liver_prob')
    risk = ctrl.Consequent(np.arange(0, 101, 1), 'risk')

    heart_prob.automf(3)
    liver_prob.automf(3)

    risk['low'] = fuzz.trimf(risk.universe, [0, 0, 40])
    risk['medium'] = fuzz.trimf(risk.universe, [30, 50, 70])
    risk['high'] = fuzz.trimf(risk.universe, [60, 100, 100])

    rules = [
        ctrl.Rule(heart_prob['high'] | liver_prob['high'], risk['high']),
        ctrl.Rule(heart_prob['average'] | liver_prob['average'], risk['medium']),
        ctrl.Rule(heart_prob['low'] & liver_prob['low'], risk['low'])
    ]

    return ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))

fuzzy_sim = setup_fuzzy()

# --- SIDEBAR MENU ---
with st.sidebar:
    menu = option_menu(
        "Disease Prediction",
        ["Combined Risk Prediction", "Heart Disease Prediction", "Liver Disease Prediction", "Exploratory Data Analysis", "Plots and Charts", "Histogram Marker"],
        icons=["activity", "heart", "droplet", "bar-chart", "pie-chart", "graph-up"],
        default_index=0
    )

# --- COMBINED RISK PREDICTION ---
if menu == "Combined Risk Prediction":
    st.title("Heart & Liver Combined Risk")
    with st.form("combined_form"):
        st.subheader("Heart Inputs")
        age = st.number_input("Age", 1, 120, 30)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.number_input("Chest Pain Type", 0, 3, 0)
        chol = st.number_input("Cholesterol", 100, 600, 200)
        thalach = st.number_input("Max Heart Rate", 60, 220, 150)

        st.subheader("Liver Inputs")
        bmi = st.number_input("BMI", 10.0, 50.0, 24.0)
        alcohol = st.number_input("Alcohol Consumption", 0.0, 10.0, 1.0)
        gender_liver = st.selectbox("Gender (Liver)", ["Male", "Female"])
        liver_func = st.number_input("Liver Function Test", 0.0, 100.0, 40.0)

        submit = st.form_submit_button("Predict")

    if submit:
        if heart_model and liver_model:
            heart_input = [age, 1 if sex == "Male" else 0, cp, 120, chol, 0, 1, thalach, 0, 0.0, 1]
            liver_input = [age, 1 if gender_liver == "Male" else 0, bmi, alcohol, 0, 0, 2.0, 0, 0, liver_func, 0]

            heart_prob = heart_model.predict_proba([heart_input])[0][1] * 100
            liver_prob = liver_model.predict_proba([liver_input])[0][1] * 100

            fuzzy_sim.input['heart_prob'] = heart_prob
            fuzzy_sim.input['liver_prob'] = liver_prob
            fuzzy_sim.compute()
            risk = fuzzy_sim.output['risk']

            st.metric("Heart Disease Probability", f"{heart_prob:.2f}%")
            st.metric("Liver Disease Probability", f"{liver_prob:.2f}%")
            st.metric("Combined Risk Score", f"{risk:.2f}%")
        else:
            st.error("Model loading failed.")

# --- HEART DISEASE PREDICTION ---
elif menu == "Heart Disease Prediction":
    st.title("Heart Disease Prediction")
    age = st.number_input("Age", 1, 120, 30)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.number_input("Chest Pain Type", 0, 3, 0)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)

    if st.button("Predict Heart Disease"):
        if heart_model:
            sex_num = 1 if sex == "Male" else 0
            features = [age, sex_num, cp, 120, chol, 0, 1, thalach, 0, 0.0, 1]
            prediction = heart_model.predict([features])[0]
            prob = heart_model.predict_proba([features])[0][1] * 100
            st.success("Has Heart Disease" if prediction else "No Heart Disease")
            st.info(f"Probability: {prob:.2f}%")
        else:
            st.error("Model not loaded")

# --- LIVER DISEASE PREDICTION ---
elif menu == "Liver Disease Prediction":
    st.title("Liver Disease Prediction")
    age = st.number_input("Age", 1, 120, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    bmi = st.number_input("BMI", 10.0, 50.0, 24.0)
    alcohol = st.number_input("Alcohol", 0.0, 10.0, 1.0)
    liver_func = st.number_input("Liver Function Test", 0.0, 100.0, 40.0)

    if st.button("Predict Liver Disease"):
        if liver_model:
            g_num = 1 if gender == "Male" else 0
            features = [age, g_num, bmi, alcohol, 0, 0, 2.0, 0, 0, liver_func, 0]
            prediction = liver_model.predict([features])[0]
            prob = liver_model.predict_proba([features])[0][1] * 100
            st.success("Has Liver Disease" if prediction else "No Liver Disease")
            st.info(f"Probability: {prob:.2f}%")
        else:
            st.error("Model not loaded")

# --- EDA PAGE ---
elif menu == "Exploratory Data Analysis":
    st.title("EDA")
    choice = st.selectbox("Choose Dataset", ["Heart", "Liver"])
    path = HEART_DATA if choice == "Heart" else LIVER_DATA

    if os.path.exists(path):
        df = pd.read_csv(path)
        st.write(df.head())
        st.write(df.describe())
        st.write(df.isnull().sum())

        numeric = df.select_dtypes(include=[np.number])
        if len(numeric.columns) > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(numeric.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
    else:
        st.error("Dataset not found")

# --- PLOTS PAGE ---
elif menu == "Plots and Charts":
    st.title("Charts")
    choice = st.selectbox("Dataset", ["Heart", "Liver"])
    path = HEART_DATA if choice == "Heart" else LIVER_DATA

    if os.path.exists(path):
        df = pd.read_csv(path)
        numeric = df.select_dtypes(include=[np.number])
        if numeric.empty:
            st.warning("No numeric data found")
        else:
            col = st.selectbox("Histogram Column", numeric.columns)
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            st.pyplot(fig)

            st.write("Correlation Heatmap")
            fig, ax = plt.subplots()
            sns.heatmap(numeric.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

            x = st.selectbox("X", numeric.columns)
            y = st.selectbox("Y", numeric.columns)
            fig, ax = plt.subplots()
            sns.scatterplot(x=df[x], y=df[y], ax=ax)
            st.pyplot(fig)

# --- HISTOGRAM MARKER ---
elif menu == "Histogram Marker":
    st.title("Histogram Marker")
    choice = st.selectbox("Dataset", ["Heart", "Liver"])
    path = HEART_DATA if choice == "Heart" else LIVER_DATA

    if os.path.exists(path):
        df = pd.read_csv(path)
        numeric = df.select_dtypes(include=[np.number])
        col = st.selectbox("Column", numeric.columns)
        val = st.number_input("Value to mark", float(df[col].min()), float(df[col].max()), float(df[col].mean()))

        fig, ax = plt.subplots()
        sns.histplot(df[col], bins=30, kde=True, ax=ax)
        ax.axvline(val, color='red', linestyle='--')
        st.pyplot(fig)
