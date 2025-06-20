import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Page Configuration
st.set_page_config(page_title="Liver and Heart Disease Prediction", layout="wide", page_icon="🏥")
HEART_MODEL_FILENAME = 'heart_disease_model (3).sav'
LIVER_MODEL_FILENAME = 'liver_disease_model (2).sav'
HEART_DATA_FILENAME = 'heart (1).csv'
LIVER_DATA_FILENAME = 'Liver_disease_data.csv'

# Function to Load Models
def load_model(model_path):
    try:
        # Before attempting to open, check if the file actually exists
        if not os.path.exists(model_path):
            st.error(f"Error: Model file not found at '{model_path}'. Please ensure it's in the repository root and the name matches exactly.")
            return None
        with open(model_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        st.error(f"Error loading model from '{model_path}': {e}")
        return None

# Load heart and liver disease models
heart_disease_model = load_model(HEART_MODEL_FILENAME)
liver_disease_model = load_model(LIVER_MODEL_FILENAME)
# --- FUZZY LOGIC SYSTEM ---
@st.cache_resource
def setup_fuzzy_system():
    """
    Sets up the fuzzy logic control system for combined risk prediction.
    Caches the system to avoid re-creation on every rerun.
    """
    heart_prob = ctrl.Antecedent(np.arange(0, 101, 1), 'heart_prob')
    liver_prob = ctrl.Antecedent(np.arange(0, 101, 1), 'liver_prob')
    risk = ctrl.Consequent(np.arange(0, 101, 1), 'risk')

    heart_prob.automf(3, variable_type='quant')
    liver_prob.automf(3, variable_type='quant')

    risk['low'] = fuzz.trimf(risk.universe, [0, 0, 40])
    risk['medium'] = fuzz.trimf(risk.universe, [30, 50, 70])
    risk['high'] = fuzz.trimf(risk.universe, [60, 100, 100])

    rule1 = ctrl.Rule(heart_prob['high'] | liver_prob['high'], risk['high'])
    rule2 = ctrl.Rule(heart_prob['average'] | liver_prob['average'], risk['medium'])
    rule3 = ctrl.Rule(heart_prob['low'] & liver_prob['low'], risk['low'])

    risk_ctrl_system = ctrl.ControlSystem([rule1, rule2, rule3])
    return ctrl.ControlSystemSimulation(risk_ctrl_system)

fuzzy_simulator = setup_fuzzy_system()

# --- SIDEBAR ---
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Combined Risk Prediction', 'Heart Disease Prediction', 'Liver Disease Prediction',
         'Exploratory Data Analysis', 'Plots and Charts', 'Histogram Marker'],
        icons=['heart-pulse', 'heart', 'activity', 'bar-chart-line', 'bar-chart', 'graph-up'],
        menu_icon='hospital-fill',
        default_index=0
    )

# --- Combined Fuzzy Logic Page ---
if selected == 'Combined Risk Prediction':
    st.title("Heart & Liver Disease Risk Prediction")
    # Removed the red note st.markdown block from here
    with st.form("fuzzy_input_form"):
        st.subheader("Enter Patient Details for Combined Risk")

        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", 1, 120, 30)
            sex = st.selectbox("Sex (Heart)", ["Male", "Female"])
            cp = st.number_input("Chest Pain Type (0-3)", 0, 3, 0)
            chol = st.number_input("Cholesterol", 100, 600, 200)
        with col2:
            thalach = st.number_input("Max Heart Rate", 60, 220, 150)
            bmi = st.number_input("BMI", 10.0, 50.0, 24.0)
            alcohol = st.number_input("Alcohol Consumption", 0.0, 10.0, 1.0)
        with col3:
            gender_liver = st.selectbox("Gender (Liver)", ["Male", "Female"])
            liver_function = st.number_input("Liver Function Test", 0.0, 100.0, 40.0)

        if st.form_submit_button("Predict Combined Risk"):
            if not heart_disease_model or not liver_disease_model:
                st.error("Models not loaded properly.")
            else:
                sex_num = 1 if sex == "Male" else 0
                gender_num = 1 if gender_liver == "Male" else 0

                heart_input = [age, sex_num, cp, 120, chol, 0, 1, thalach, 0, 0.0, 1]
                liver_input = [age, gender_num, bmi, alcohol, 0, 0, 2.0, 0, 0, liver_function, 0]

                heart_proba = heart_disease_model.predict_proba([heart_input])[0][1] * 100
                liver_proba = liver_disease_model.predict_proba([liver_input])[0][1] * 100

                fuzzy_simulator.input['heart_prob'] = heart_proba
                fuzzy_simulator.input['liver_prob'] = liver_proba
                fuzzy_simulator.compute()
                risk_score = fuzzy_simulator.output['risk']

                st.subheader("Prediction Results:")
                st.write(f"Heart Disease Probability: **{heart_proba:.2f}%**")
                st.write(f"Liver Disease Probability: **{liver_proba:.2f}%**")
                st.write(f"Combined Risk Score: **{risk_score:.2f}%**")

                if risk_score >= 70:
                    st.write("Overall Risk: High 🚨")
                elif risk_score >= 40:
                    st.write("Overall Risk: Moderate ⚠️")
                else:
                    st.write("Overall Risk: Low ✅")


elif selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input('Age', 1, 120, step=1)
        trestbps = st.number_input('Resting Blood Pressure', 50, 250, step=1)
        restecg = st.number_input('Resting ECG (0, 1, 2)', 0, 2, step=1)
        oldpeak = st.number_input('ST Depression', 0.0, step=0.1)
    with col2:
        sex = st.selectbox('Sex', ['Male', 'Female'])
        cp = st.number_input('Chest Pain Type (0-3)', 0, 3, step=1)
        chol = st.number_input('Serum Cholesterol', 100, 600, step=1)
        thalach = st.number_input('Maximum Heart Rate', 60, 220, step=1)
    with col3:
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)', [0, 1])
        exang = st.selectbox('Exercise Induced Angina (1 = Yes, 0 = No)', [0, 1])
        slope = st.number_input('Slope of Peak Exercise ST Segment (0, 1, 2)', 0, 2, step=1)

    if st.button('Heart Disease Test Result'):
        sex_num = 1 if sex == 'Male' else 0
        user_input = [age, sex_num, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope]
        if heart_disease_model:
            try:
                prediction = heart_disease_model.predict([user_input])
                result = 'The person has heart disease' if prediction[0] == 1 else 'The person does not have heart disease'
                st.write(result)
                probability = heart_disease_model.predict_proba([user_input])[0][1]
                percentage = round(probability * 100, 2)
                if probability >= 0.7:
                    risk_level = "High Risk 🚨"
                elif probability >= 0.4:
                    risk_level = "Moderate Risk ⚠️"
                else:
                    risk_level = "Low Risk ✅"
                st.write(f"Probability of Heart Disease: {percentage}%")
                st.write(f"Risk Level: {risk_level}")
            except Exception as e:
                st.error(f"Prediction Error: {e}")

elif selected == 'Liver Disease Prediction':
    st.title('Liver Disease Prediction using ML')
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input('Age', 1, 120, step=1)
        BMI = st.number_input('Body Mass Index', 10.0, step=0.1)
        Alcoholcons = st.number_input('Alcohol Consumption', 0.0, step=0.1)
    with col2:
        gender = st.selectbox('Gender', ['Male', 'Female'])
        Smoking = st.selectbox('Smoking (1 = Yes, 0 = No)', [0, 1])
        Genetic_Risk = st.selectbox('Genetic Risk (1 = Yes, 0 = No)', [0, 1])
    with col3:
        PhysicalActivity = st.number_input('Physical Activity', 0.0, step=0.1)
        Diabetes = st.selectbox('Diabetes (1 = Yes, 0 = No)', [0, 1])
        Hypertension = st.selectbox('Hypertension (1 = Yes, 0 = No)', [0, 1])
        liverfunctiontest = st.number_input('Liver Function Test', 0.0, step=0.1)
    Diagnosis = st.selectbox('Diagnosis (0, 1, 2)', [0, 1, 2])
    if st.button('Liver Disease Test Result'):
        gender_num = 1 if gender == 'Male' else 0
        user_input = [age, gender_num, BMI, Alcoholcons, Smoking, Genetic_Risk, PhysicalActivity, Diabetes, Hypertension, liverfunctiontest, Diagnosis]
        if liver_disease_model:
            try:
                prediction = liver_disease_model.predict([user_input])
                result = 'The person has liver disease' if prediction[0] == 1 else 'The person does not have liver disease'
                st.write(result)
                probability = liver_disease_model.predict_proba([user_input])[0][1]
                percentage = round(probability * 100, 2)
                if probability >= 0.7:
                    risk_level = "High Risk 🚨"
                elif probability >= 0.4:
                    risk_level = "Moderate Risk ⚠️"
                else:
                    risk_level = "Low Risk ✅"
                st.write(f"Probability of Liver Disease: {percentage}%")
                st.write(f"Risk Level: {risk_level}")
            except Exception as e:
                st.error(f"Prediction Error for Liver Disease: {e}")

elif selected == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    dataset_choice = st.selectbox("Select Dataset", ["Heart Disease", "Liver Disease"])
    # Adjusted paths to use the defined constants
    data_path = HEART_DATA_FILENAME if dataset_choice == "Heart Disease" else LIVER_DATA_FILENAME
    
    if os.path.exists(data_path):
        try:
            data = pd.read_csv(data_path)
            st.write("### Data Preview:")
            st.dataframe(data.head())
            st.write("### Summary Statistics:")
            st.write(data.describe())
            st.write("### Missing Values:")
            st.write(data.isnull().sum())
        except Exception as e:
            st.error(f"Error loading or processing data from '{data_path}': {e}")
    else:
        st.error("Dataset not found! Please check the path: " + data_path)

elif selected == "Plots and Charts":
    st.title("Plots and Charts")
    dataset_choice = st.selectbox("Select Dataset", ["Heart Disease", "Liver Disease"])
    # Adjusted paths to use the defined constants
    data_path = HEART_DATA_FILENAME if dataset_choice == "Heart Disease" else LIVER_DATA_FILENAME
    
    if os.path.exists(data_path):
        try:
            data = pd.read_csv(data_path)
            numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if numeric_cols:
                st.write("### Histogram")
                selected_hist_col = st.selectbox("Select column for Histogram", numeric_cols, key='hist_col')
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.histplot(data[selected_hist_col], kde=True, bins=30, ax=ax)
                ax.set_title(f'Histogram of {selected_hist_col}')
                st.pyplot(fig)

                st.write("### Correlation Heatmap")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
                st.pyplot(fig)

                st.write("### Line Plot")
                x_axis = st.selectbox("Select X-axis", numeric_cols, key='line_x')
                y_axis = st.selectbox("Select Y-axis", numeric_cols, key='line_y')
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.lineplot(x=data[x_axis], y=data[y_axis], marker='o', ax=ax)
                ax.set_title(f'Line Plot of {y_axis} vs {x_axis}')
                st.pyplot(fig)

                st.write("### Box Plot")
                selected_box_col = st.selectbox("Select column for Box Plot", numeric_cols, key='box_col')
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.boxplot(x=data[selected_box_col], ax=ax)
                ax.set_title(f'Box Plot of {selected_box_col}')
                st.pyplot(fig)
            else:
                st.warning("No numeric columns found in dataset.")
        except Exception as e:
            st.error(f"Error generating plots or charts from '{data_path}': {e}")
    else:
        st.error("Dataset not found! Please check the path: " + data_path)

elif selected == "Histogram Marker":
    st.title("Histogram Marker Tool")
    dataset_choice = st.selectbox("Select Dataset", ["Heart Disease", "Liver Disease"], key="marker_dataset")
    data_path = HEART_DATA_FILENAME if dataset_choice == "Heart Disease" else LIVER_DATA_FILENAME
    
    if os.path.exists(data_path):
        try:
            data = pd.read_csv(data_path)
            numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if numeric_cols:
                selected_col = st.selectbox("Select Column for Histogram", numeric_cols)
                marker_value = st.number_input(f"Enter a value to mark on the histogram of {selected_col}", value=float(data[selected_col].mean()))
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(data[selected_col], bins=30, kde=True, ax=ax)
                ax.axvline(marker_value, color='red', linestyle='--', linewidth=2, label=f'Marked Value: {marker_value}')
                ax.set_title(f"Histogram of {selected_col} with Marker")
                ax.legend()
                st.pyplot(fig)
            else:
                st.warning("No numeric columns found in dataset.")
        except Exception as e:
            st.error(f"Error generating histogram with marker from '{data_path}': {e}")
    else:
        st.error("Dataset not found! Please check the path.")


