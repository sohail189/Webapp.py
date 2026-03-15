# 🏥 Disease Prediction Web App

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Stars](https://img.shields.io/github/stars/sohail189/Webapp.py?style=for-the-badge)

**A Streamlit web application that predicts Heart Disease and Liver Disease risk using trained Machine Learning models — with full EDA and interactive visualizations.**

[🚀 Run Locally](#-installation) · [📊 Features](#-features) · [🔍 How It Works](#-how-it-works) · [🐛 Report Bug](https://github.com/sohail189/Webapp.py/issues)

</div>

---

## 📝 Description

The **Disease Prediction Web App** is an AI-powered health tool that lets users input their medical data and instantly receive a **risk prediction** for:

- ❤️ **Heart Disease** — using a trained classification model on the UCI Heart Disease dataset
- 🫀 **Liver Disease** — using a trained model on the Liver Disease dataset

Beyond predictions, the app includes a full **Exploratory Data Analysis (EDA)** suite with interactive charts, correlation heatmaps, histograms, and a unique **Histogram Marker Tool** that lets users visualise where their personal data falls within the dataset distribution.

> ⚠️ **Disclaimer:** This app is for educational and informational purposes only. It is **not** a replacement for professional medical advice. Always consult a qualified doctor.

---

## ✨ Features

- ❤️ **Heart Disease Prediction** — Input 13 clinical features, get risk level + probability score
- 🫀 **Liver Disease Prediction** — Input patient liver function test values, get instant prediction
- 📊 **Interactive EDA Dashboard** — Explore dataset distributions, correlations, and statistics
- 🔥 **Correlation Heatmap** — Visualise feature relationships with an interactive heatmap
- 📈 **Histogram Viewer** — Explore distribution of any feature in the dataset
- 📌 **Histogram Marker Tool** — Mark your personal value on any histogram to compare with the population
- 📉 **Box Plots & Line Plots** — Understand spread and trends in the medical data
- 🎯 **Risk Level Output** — Results displayed as Low / Medium / High risk with probability percentage
- 🖥️ **Clean Streamlit UI** — Simple, beginner-friendly web interface
- 🔬 **Jupyter Notebooks Included** — Full model training workflow for both diseases

---

## 🛠️ Technologies Used

| Category | Technology |
|---|---|
| **Frontend / UI** | Streamlit |
| **Machine Learning** | Scikit-learn, CatBoost, Random Forest |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Model Persistence** | Pickle (`.sav` files) |
| **Development** | Python 3.12, VS Code, Anaconda |
| **Notebooks** | Jupyter Notebook |
| **Language** | Python (96.9%), Jupyter Notebook |

---

## 📁 Repository Structure

```
Webapp.py/
│
├── app2.py                        ← Main Streamlit application
│
├── 📓 Notebooks (Model Training)
│   ├── final_heart (1).ipynb      ← Heart disease model training notebook
│   └── liver_disease (1).ipynb    ← Liver disease model training notebook
│
├── 🤖 Trained Models
│   ├── heart_disease_model (3).sav  ← Saved heart disease ML model
│   └── liver_disease_model (2).sav  ← Saved liver disease ML model
│
├── 📂 Datasets
│   ├── heart (1).csv              ← Heart Disease UCI dataset
│   └── Liver_disease_data.csv     ← Liver Disease dataset
│
├── requirements.txt               ← Python dependencies
└── README.md                      ← Project documentation
```

---

## 🔍 How It Works

```
User Input → Feature Validation → Pre-trained ML Model → Prediction
                                                              ↓
                                              Risk Level + Probability Score
                                                              ↓
                                         Visualizations & EDA Comparison
```

1. User enters their health data through the Streamlit sidebar form
2. Input is validated and passed to the pre-trained `.sav` model
3. Model returns a binary prediction (Disease / No Disease)
4. App displays risk level, probability, and comparison charts

---

## 📋 Input Features

### ❤️ Heart Disease Inputs

| Feature | Description | Type |
|---|---|---|
| `age` | Age in years | Integer |
| `sex` | Sex (1 = Male, 0 = Female) | Binary |
| `cp` | Chest pain type (0–3) | Integer |
| `trestbps` | Resting blood pressure (mm Hg) | Integer |
| `chol` | Serum cholesterol (mg/dl) | Integer |
| `fbs` | Fasting blood sugar > 120 mg/dl | Binary |
| `restecg` | Resting ECG results (0–2) | Integer |
| `thalach` | Maximum heart rate achieved | Integer |
| `exang` | Exercise-induced angina (1 = Yes) | Binary |
| `oldpeak` | ST depression (exercise vs rest) | Float |
| `slope` | Slope of peak exercise ST segment | Integer |
| `ca` | Number of major vessels (0–3) | Integer |
| `thal` | Thalassemia type (1–3) | Integer |

### 🫀 Liver Disease Inputs

| Feature | Description |
|---|---|
| `Age` | Patient age |
| `Gender` | Male / Female |
| `Total Bilirubin` | Liver function marker |
| `Direct Bilirubin` | Liver function marker |
| `Alkaline Phosphotase` | Enzyme level |
| `Alamine Aminotransferase` | Liver enzyme |
| `Aspartate Aminotransferase` | Liver enzyme |
| `Total Proteins` | Protein level in blood |
| `Albumin` | Protein produced by liver |
| `Albumin / Globulin Ratio` | Liver health indicator |

---

## 📦 Installation

### Step 1 — Clone the Repository

```bash
git clone https://github.com/sohail189/Webapp.py.git
cd Webapp.py
```

### Step 2 — Create Virtual Environment (Recommended)

```bash
# Windows (Anaconda)
conda create -n disease-app python=3.12
conda activate disease-app

# Or with venv
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Run the App

```bash
streamlit run app2.py
```

App opens at **http://localhost:8501** 🎉

---

## 🚀 Usage

### Making a Prediction

1. Open the app in your browser
2. Select **Heart Disease** or **Liver Disease** from the sidebar
3. Enter your health values in the input fields
4. Click **Predict**
5. View your **risk level** and **probability score**

### Using the EDA Tools

- Navigate to the **EDA** section
- Select any feature from the dropdown
- View histogram, box plot, or correlation heatmap
- Use the **Histogram Marker Tool** — enter your value to see where you fall in the distribution

### Running the Training Notebooks

```bash
jupyter notebook "final_heart (1).ipynb"
# or
jupyter notebook "liver_disease (1).ipynb"
```

---

## 📊 Model Information

| Model | Disease | Algorithm | Dataset Size |
|---|---|---|---|
| `heart_disease_model (3).sav` | Heart Disease | Random Forest / CatBoost | 303 records |
| `liver_disease_model (2).sav` | Liver Disease | Random Forest / CatBoost | 583 records |

> Both models are serialized using Python's `pickle` library and loaded directly into the Streamlit app at runtime.

---

## 🖥️ Demo

> 💡 *Add screenshots of your app here by uploading images to the repo*

```markdown
![Heart Disease Prediction](screenshots/heart_prediction.png)
![Liver Disease Prediction](screenshots/liver_prediction.png)
![EDA Dashboard](screenshots/eda_dashboard.png)
![Histogram Marker Tool](screenshots/histogram_marker.png)
```

---

## ⚙️ Requirements

```txt
streamlit
scikit-learn
catboost
pandas
numpy
matplotlib
seaborn
pickle-mixin
```

Install all:
```bash
pip install -r requirements.txt
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: catboost` | Run `pip install catboost` |
| `File not found: .sav model` | Make sure all `.sav` files are in the same folder as `app2.py` |
| `streamlit: command not found` | Run `pip install streamlit` first |
| App won't open in browser | Go to `http://localhost:8501` manually |
| `Error: Invalid value: File does not exist` | Run `cd Webapp.py` then `streamlit run app2.py` |

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. **Fork** the repository
2. **Create** a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Commit** your changes:
   ```bash
   git commit -m "Add: your feature description"
   ```
4. **Push** and open a **Pull Request**

### 💡 Ideas for Contribution

- [ ] Add Diabetes prediction model
- [ ] Add Kidney Disease prediction
- [ ] Integrate a medical AI chatbot (Gemini / GPT)
- [ ] Add patient history saving with SQLite
- [ ] Deploy to Streamlit Community Cloud
- [ ] Add SHAP values for model explainability
- [ ] Add confidence intervals to predictions

---

## 📄 License

This project is licensed under the **MIT License** — free to use, modify, and distribute with attribution.

---

## 📬 Contact

**Muhammad Sohail**
*Data Scientist | Machine Learning Engineer | Streamlit Developer*

[![GitHub](https://img.shields.io/badge/GitHub-sohail189-181717?style=flat-square&logo=github)](https://github.com/sohail189)
[![Email](https://img.shields.io/badge/Email-sm7933294@gmail.com-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:sm7933294@gmail.com)

🔗 **Project Link:** [https://github.com/sohail189/Webapp.py](https://github.com/sohail189/Webapp.py)

---

<div align="center">

Built with ❤️ using Python, Streamlit & Machine Learning

⭐ **Found this useful? Give it a star!** ⭐

</div>
