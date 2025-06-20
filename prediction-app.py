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

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Liver and Heart Disease Prediction",
    layout="wide",
    page_icon="🏥"
)

# --- MODEL LOADING ---
def load_model(model_name):
    """
    Loads a pickled machine learning model from a specified relative path.
    Displays an error if loading fails.
    """
    script_dir = os.path.dirname(__file__) # Get the directory of the current script
    model_path = os.path.join(script_dir, model_name)
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
            st.success(f"Model '{model_name}' loaded successfully!")
            return model
    except FileNotFoundError:
        st.error(f"Error: Model file '{model_name}' not found at '{model_path}'. "
                 "Please ensure the model file is in the same directory as the script.")
        return None
    except Exception as e:
        st.error(f"Error loading model '{model_name}': {e}")
        return None

# Load heart and liver disease models
# Ensure these .sav files are in the same directory as your Streamlit script.
heart_disease_model = load_model('heart_disease_model (3).sav')
liver_disease_model = load_model('liver_disease_model (2).sav')

# --- FUZZY LOGIC SETUP ---
@st.cache_resource
def setup_fuzzy_system():
    """
    Sets up the fuzzy logic control system for combined risk prediction.
    Caches the system to avoid re-creation on every rerun.
    """
    # Antecedents (inputs)
    heart_prob = ctrl.Antecedent(np.arange(0, 101, 1), 'heart_prob')
    liver_prob = ctrl.Antecedent(np.arange(0, 101, 1), 'liver_prob')

    # Consequent (output)
    risk = ctrl.Consequent(np.arange(0, 101, 1), 'risk')

    # Membership functions for inputs (automatically generated for simplicity)
    heart_prob.automf(3, variable_type='quant') # Creates 'low', 'average', 'high'
    liver_prob.automf(3, variable_type='quant') # Creates 'low', 'average', 'high'

    # Custom membership functions for output (risk)
    risk['low'] = fuzz.trimf(risk.universe, [0, 0, 40])
    risk['medium'] = fuzz.trimf(risk.universe, [30, 50, 70])
    risk['high'] = fuzz.trimf(risk.universe, [60, 100, 100])

    # Fuzzy Rules (Corrected for logical risk assessment)
    # Rule 1: If heart probability is high OR liver probability is high, then risk is high.
    rule1 = ctrl.Rule(heart_prob['high'] | liver_prob['high'], risk['high'])
    # Rule 2: If heart probability is average OR liver probability is average, then risk is medium.
    rule2 = ctrl.Rule(heart_prob['average'] | liver_prob['average'], risk['medium'])
    # Rule 3: If heart probability is low AND liver probability is low, then risk is low.
    rule3 = ctrl.Rule(heart_prob['low'] & liver_prob['low'], risk['low'])

    # Control System creation and simulation setup
    risk_ctrl_system = ctrl.ControlSystem([rule1, rule2, rule3])
    return ctrl.ControlSystemSimulation(risk_ctrl_system)

fuzzy_simulator = setup_fuzzy_system()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        [
            'Combined Risk Prediction',
            'Heart Disease Prediction',
            'Liver Disease Prediction',
            'Exploratory Data Analysis',
            'Plots and Charts',
            'Histogram Marker'
        ],
        icons=['heart-pulse', 'heart', 'activity', 'bar-chart-line', 'bar-chart', 'graph-up'],
        menu_icon='hospital-fill',
        default_index=0
    )

# --- PAGE FUNCTIONS ---

def run_combined_risk_prediction_page():
    """
    Displays the Combined Heart & Liver Disease Risk Prediction page using Fuzzy Logic.
    This page uses a simplified set of 4-5 inputs for each disease
    and uses default values for the remaining features required by the models.
    """
    st.title("Heart & Liver Disease Risk Prediction")
    st.markdown(
        """
        <p style="color:red; font-weight:bold;">
        Note: For this combined prediction, only a few key inputs are displayed for simplicity. 
        The underlying machine learning models may require more features, for which common default 
        values are used. This might affect the precision of the probability predictions from the ML models.
        </p>
        """, unsafe_allow_html=True
    )

    with st.form("fuzzy_input_form"):
        st.subheader("Enter Patient Details for Combined Risk")

        st.markdown("<h5>Heart Disease Key Inputs (5 features)</h5>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", 1, 120, 30, help="Age in years.")
            sex = st.selectbox("Sex (for Heart Disease Model)", ["Male", "Female"], help="Gender of the patient.")
        with col2:
            cp = st.number_input("Chest Pain Type (0-3)", 0, 3, 0, help="Type of chest pain experienced: 0=typical angina, 1=atypical angina, 2=non-anginal pain, 3=asymptomatic.")
            chol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200, help="Serum Cholestoral in mg/dl.")
        with col3:
            thalach = st.number_input("Max Heart Rate", 60, 220, 150, help="Maximum heart rate achieved during exercise.")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h5>Liver Disease Key Inputs (5 features)</h5>", unsafe_allow_html=True)
        col4, col5, col6 = st.columns(3)
        with col4:
            bmi = st.number_input("BMI", 10.0, 50.0, 24.0, help="Body Mass Index (BMI).")
            alcohol = st.number_input("Alcohol Consumption", 0.0, 10.0, 1.0, help="Level of alcohol consumption (e.g., units per day/week).")
        with col5:
            gender_liver = st.selectbox("Gender (for Liver Disease Model)", ["Male", "Female"], help="Gender of the patient.")
        with col6:
            liver_function = st.number_input("Liver Function Test Result", 0.0, 100.0, 40.0, help="Numerical result from a liver function test (e.g., AST/ALT levels).")
            # Diagnosis is assumed to be 0 by default for simplicity as per requirements for 5 inputs
            # diagnosis = st.selectbox("Diagnosis Code (0, 1, 2)", [0, 1, 2])

        submitted = st.form_submit_button("Predict Combined Risk")

    if submitted:
        # Check if models are loaded
        if heart_disease_model is None or liver_disease_model is None:
            st.error("One or both prediction models could not be loaded. Cannot proceed with prediction.")
            return

        # Convert sex/gender to numerical (1 for Male, 0 for Female)
        sex_num_heart = 1 if sex == "Male" else 0
        gender_num_liver = 1 if gender_liver == "Male" else 0

        # --- Constructing full input arrays for models ---
        # These arrays must match the exact number and order of features
        # that the pre-trained models expect.
        # Features not taken from UI are assigned default values.

        # Heart Disease Model Input (11 features expected by original model based on your `user_input` list)
        # Original order: [age, sex_num, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope]
        heart_input_features = [
            age,                         # UI Input
            sex_num_heart,               # UI Input
            cp,                          # UI Input
            120,                         # Default for trestbps (Resting Blood Pressure)
            chol,                        # UI Input
            0,                           # Default for fbs (Fasting Blood Sugar >120 mg/dl)
            1,                           # Default for restecg (Resting ECG)
            thalach,                     # UI Input
            0,                           # Default for exang (Exercise Induced Angina)
            0.0,                         # Default for oldpeak (ST Depression)
            1                            # Default for slope (Slope of Peak Exercise ST Segment)
        ]

        # Liver Disease Model Input (11 features expected by original model based on your `user_input` list)
        # Original order: [age, gender_num, bmi, alcohol, smoking, genetics, physical_activity, diabetes, hypertension, liver_function, diagnosis]
        liver_input_features = [
            age,                         # UI Input (age for liver model)
            gender_num_liver,            # UI Input
            bmi,                         # UI Input
            alcohol,                     # UI Input
            0,                           # Default for smoking (0=No)
            0,                           # Default for genetics (0=No)
            2.0,                         # Default for physical_activity (as per your original code's default)
            0,                           # Default for diabetes (0=No)
            0,                           # Default for hypertension (0=No)
            liver_function,              # UI Input
            0                            # Default for diagnosis (code 0, as not in reduced UI)
        ]

        try:
            # Predict probabilities using the loaded models
            heart_proba = heart_disease_model.predict_proba([heart_input_features])[0][1] * 100
            liver_proba = liver_disease_model.predict_proba([liver_input_features])[0][1] * 100

            # Feed probabilities into the fuzzy logic system
            fuzzy_simulator.input['heart_prob'] = heart_proba
            fuzzy_simulator.input['liver_prob'] = liver_proba
            fuzzy_simulator.compute()
            final_risk = fuzzy_simulator.output['risk']

            st.subheader("Prediction Results:")
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.info(f"Heart Disease Probability: **{heart_proba:.2f}%**")
                st.info(f"Liver Disease Probability: **{liver_proba:.2f}%**")
            with col_res2:
                st.warning(f"Combined Fuzzy Logic Risk Score: **{final_risk:.2f}**")
                if final_risk >= 70:
                    st.error("High Overall Risk Detected 🚨")
                elif final_risk >= 40:
                    st.warning("Moderate Overall Risk Detected ⚠️")
                else:
                    st.success("Low Overall Risk Detected ✅")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}. "
                     "Please ensure the models are correctly loaded and inputs match expected format.")


def run_heart_disease_prediction_page():
    """
    Displays the individual Heart Disease Prediction page with all relevant inputs.
    """
    st.title('Heart Disease Prediction using ML')
    st.markdown("<h5>Enter All Heart Disease Specific Parameters</h5>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input('Age', 1, 120, 30, help="Age in years.")
        trestbps = st.number_input('Resting Blood Pressure (mm Hg)', 50, 250, 120, help="Resting blood pressure (in mm Hg on admission to the hospital).")
        restecg = st.number_input('Resting ECG (0, 1, 2)', 0, 2, 1, help="Resting electrocardiographic results (0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy).")
        oldpeak = st.number_input('ST Depression (oldpeak)', 0.0, 6.0, 1.0, step=0.1, help="ST depression induced by exercise relative to rest.")
    with col2:
        sex = st.selectbox('Sex', ['Male', 'Female'], help="Gender (Male/Female).")
        cp = st.number_input('Chest Pain Type (0-3)', 0, 3, 0, help="Chest pain type (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic).")
        chol = st.number_input('Serum Cholesterol (mg/dL)', 100, 600, 200, help="Serum Cholestoral in mg/dl.")
        thalach = st.number_input('Maximum Heart Rate', 60, 220, 150, help="Maximum heart rate achieved during exercise.")
    with col3:
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (1=Yes, 0=No)', [0, 1], help="Fasting blood sugar > 120 mg/dl (1 = true; 0 = false).")
        exang = st.selectbox('Exercise Induced Angina (1=Yes, 0=No)', [0, 1], help="Exercise induced angina (1 = yes; 0 = no).")
        slope = st.number_input('Slope of Peak Exercise ST Segment (0, 1, 2)', 0, 2, 1, help="The slope of the peak exercise ST segment (0: upsloping, 1: flat, 2: downsloping).")
    
    if st.button('Predict Heart Disease'):
        if heart_disease_model is None:
            st.error("Heart disease model not loaded correctly! Cannot perform prediction.")
            return

        sex_num = 1 if sex == 'Male' else 0
        
        # Ensure this list matches the features and order the model was trained on.
        user_input_heart = [age, sex_num, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope]
        
        try:
            prediction = heart_disease_model.predict([user_input_heart])
            
            # Prediction result display
            if prediction[0] == 1:
                st.success('Prediction: The person is likely to have heart disease.')
            else:
                st.success('Prediction: The person is unlikely to have heart disease.')
            
            # Probability and Risk Level display
            probability = heart_disease_model.predict_proba([user_input_heart])[0][1]
            percentage = round(probability * 100, 2)
            
            st.info(f"Probability of Heart Disease: {percentage}%")
            if percentage >= 70:
                st.error("Risk Level: High Risk 🚨")
            elif percentage >= 40:
                st.warning("Risk Level: Moderate Risk ⚠️")
            else:
                st.success("Risk Level: Low Risk ✅")

        except Exception as e:
            st.error(f"An error occurred during heart disease prediction: {e}. "
                     "Please check inputs and model compatibility.")


def run_liver_disease_prediction_page():
    """
    Displays the individual Liver Disease Prediction page with all relevant inputs.
    """
    st.title('Liver Disease Prediction using ML')
    st.markdown("<h5>Enter All Liver Disease Specific Parameters</h5>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input('Age', 1, 120, 30, help="Age in years.")
        bmi = st.number_input('Body Mass Index (BMI)', 10.0, 50.0, 24.0, step=0.1, help="Body Mass Index.")
        alcohol_consumption = st.number_input('Alcohol Consumption', 0.0, 10.0, 1.0, step=0.1, help="Level of alcohol consumption (e.g., units per day/week).")
    with col2:
        gender = st.selectbox('Gender', ['Male', 'Female'], help="Gender of the patient.")
        smoking = st.selectbox('Smoking (1 = Yes, 0 = No)', [0, 1], help="Smoking status (1 = Yes, 0 = No).")
        genetic_risk = st.selectbox('Genetic Risk (1 = Yes, 0 = No)', [0, 1], help="Indicates if there's a genetic predisposition (1 = Yes, 0 = No).")
    with col3:
        physical_activity = st.number_input('Physical Activity', 0.0, 10.0, 2.0, step=0.1, help="Physical activity level (e.g., hours per week).")
        diabetes = st.selectbox('Diabetes (1 = Yes, 0 = No)', [0, 1], help="Diabetes status (1 = Yes, 0 = No).")
        hypertension = st.selectbox('Hypertension (1 = Yes, 0 = No)', [0, 1], help="Hypertension status (1 = Yes, 0 = No).")
    
    liver_function_test = st.number_input('Liver Function Test Result', 0.0, 100.0, 40.0, step=0.1, help="Numerical result from a liver function test (e.g., AST/ALT levels).")
    diagnosis_code = st.selectbox('Diagnosis Code (0, 1, 2)', [0, 1, 2], help="Specific diagnosis code (e.g., disease stage or type).")

    if st.button('Predict Liver Disease'):
        if liver_disease_model is None:
            st.error("Liver disease model not loaded correctly! Cannot perform prediction.")
            return

        gender_num = 1 if gender == 'Male' else 0
        
        # Ensure this list matches the features and order the model was trained on.
        user_input_liver = [age, gender_num, bmi, alcohol_consumption, smoking, genetic_risk, physical_activity, diabetes, hypertension, liver_function_test, diagnosis_code]
        
        try:
            prediction = liver_disease_model.predict([user_input_liver])
            
            # Prediction result display
            if prediction[0] == 1:
                st.success('Prediction: The person is likely to have liver disease.')
            else:
                st.success('Prediction: The person is unlikely to have liver disease.')
            
            # Probability and Risk Level display
            probability = liver_disease_model.predict_proba([user_input_liver])[0][1]
            percentage = round(probability * 100, 2)
            
            st.info(f"Probability of Liver Disease: {percentage}%")
            if percentage >= 70:
                st.error("Risk Level: High Risk 🚨")
            elif percentage >= 40:
                st.warning("Risk Level: Moderate Risk ⚠️")
            else:
                st.success("Risk Level: Low Risk ✅")
        
        except Exception as e:
            st.error(f"An error occurred during liver disease prediction: {e}. "
                     "Please check inputs and model compatibility.")

# --- EXPLORATORY DATA ANALYSIS (EDA) PAGE ---
def run_eda_page():
    """Displays the Exploratory Data Analysis page."""
    st.title("Exploratory Data Analysis (EDA)")
    st.info("This section allows you to explore the datasets used for model training.")
    
    dataset_choice = st.selectbox("Select Dataset", ["Heart Disease", "Liver Disease"])
    data_path = ('heart (1).csv' if dataset_choice == "Heart Disease" else 'Liver_disease_data.csv')

    if os.path.exists(data_path):
        try:
            data = pd.read_csv(data_path)
            st.write("### Data Preview:")
            st.dataframe(data.head())
            st.write("### Summary Statistics:")
            st.write(data.describe())
            st.write("### Missing Values:")
            st.write(data.isnull().sum())
            
            # Add correlation matrix visualization for EDA
            st.write("### Correlation Matrix:")
            numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) > 1:
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(data[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
                st.pyplot(fig)
            else:
                st.warning("Not enough numeric columns for a correlation heatmap.")

        except Exception as e:
            st.error(f"Error loading or processing dataset: {e}. Ensure the CSV is correctly formatted.")
    else:
        st.error(f"Dataset not found! Please ensure '{data_path}' is in the same directory as the script.")

# --- PLOTS AND CHARTS PAGE ---
def run_plots_page():
    """Displays the Plots and Charts page."""
    st.title("Plots and Charts")
    st.info("Visualize distributions and relationships within your selected dataset.")

    dataset_choice = st.selectbox("Select Dataset", ["Heart Disease", "Liver Disease"])
    data_path = ('heart (1).csv' if dataset_choice == "Heart Disease" else 'Liver_disease_data.csv')

    if os.path.exists(data_path):
        try:
            data = pd.read_csv(data_path)
            numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

            if not numeric_cols:
                st.warning("No numeric columns found in the selected dataset to plot.")
                return

            st.write("### Histogram")
            selected_hist_col = st.selectbox("Select column for Histogram", numeric_cols, key='hist_col_plot_page')
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(data[selected_hist_col], kde=True, bins=30, ax=ax)
            ax.set_title(f'Histogram of {selected_hist_col}')
            ax.set_xlabel(selected_hist_col)
            ax.set_ylabel('Frequency')
            st.pyplot(fig)

            st.write("### Correlation Heatmap")
            # Ensure there are at least two numeric columns to compute correlation
            if len(numeric_cols) >= 2:
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(data[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
                ax.set_title('Correlation Heatmap')
                st.pyplot(fig)
            else:
                st.info("Not enough numeric columns for a correlation heatmap.")


            st.write("### Scatter Plot")
            x_axis = st.selectbox("Select X-axis for Scatter Plot", numeric_cols, key='scatter_x')
            y_axis = st.selectbox("Select Y-axis for Scatter Plot", numeric_cols, key='scatter_y')
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x=data[x_axis], y=data[y_axis], ax=ax)
            ax.set_title(f'Scatter Plot of {y_axis} vs {x_axis}')
            ax.set_xlabel(x_axis)
            ax.set_ylabel(y_axis)
            st.pyplot(fig)

            st.write("### Box Plot")
            selected_box_col = st.selectbox("Select column for Box Plot", numeric_cols, key='box_col_plot_page')
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(x=data[selected_box_col], ax=ax)
            ax.set_title(f'Box Plot of {selected_box_col}')
            ax.set_xlabel(selected_box_col)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error generating plots: {e}. Please check dataset integrity and column types.")
    else:
        st.error(f"Dataset not found! Please ensure '{data_path}' is in the same directory as the script.")

# --- HISTOGRAM MARKER PAGE ---
def run_histogram_marker_page():
    """Allows users to select a column and mark a value on its histogram."""
    st.title("Histogram Marker Tool")
    st.info("Visualize the distribution of a numeric column and mark a specific value.")

    dataset_choice = st.selectbox("Select Dataset", ["Heart Disease", "Liver Disease"], key="marker_dataset")
    data_path = ('heart (1).csv' if dataset_choice == "Heart Disease" else 'Liver_disease_data.csv')

    if os.path.exists(data_path):
        try:
            data = pd.read_csv(data_path)
            numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

            if not numeric_cols:
                st.warning("No numeric columns found in the selected dataset to create a histogram.")
                return
            
            selected_col = st.selectbox("Select Column for Histogram", numeric_cols, key='hist_marker_col')
            
            # Ensure the marker value input is within the min/max range of the selected column
            min_val = float(data[selected_col].min())
            max_val = float(data[selected_col].max())
            default_val = float(data[selected_col].mean()) # Use mean as default marker

            marker_value = st.number_input(
                f"Enter a value to mark on the histogram of '{selected_col}'", 
                value=default_val, 
                min_value=min_val, 
                max_value=max_val, 
                step=(max_val - min_val) / 100 # Adjust step dynamically
            )
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data[selected_col], bins=30, kde=True, ax=ax)
            ax.axvline(marker_value, color='red', linestyle='--', linewidth=2, label=f'Marked Value: {marker_value:.2f}')
            ax.set_title(f"Histogram of {selected_col} with Marker")
            ax.set_xlabel(selected_col)
            ax.set_ylabel('Frequency')
            ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error in Histogram Marker Tool: {e}. Please check dataset and column selection.")
    else:
        st.error(f"Dataset not found! Please ensure '{data_path}' is in the same directory as the script.")


# --- MAIN PAGE ROUTING ---
# Routes the application based on the sidebar selection.
if selected == 'Combined Risk Prediction':
    run_combined_risk_prediction_page()
elif selected == 'Heart Disease Prediction':
    run_heart_disease_prediction_page()
elif selected == 'Liver Disease Prediction':
    run_liver_disease_prediction_page()
elif selected == 'Exploratory Data Analysis':
    run_eda_page()
elif selected == 'Plots and Charts':
    run_plots_page()
elif selected == 'Histogram Marker':
    run_histogram_marker_page()
