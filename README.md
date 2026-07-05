
## About

A machine learning-powered web application that assesses diabetes risk from key health parameters, built with a focus on both predictive accuracy and interpretability. The system uses a **CatBoost** classifier — selected after comparative evaluation against multiple models for its superior recall and F1-score — trained on the Pima Indians Diabetes dataset, with **SMOTE** applied to address class imbalance.

Beyond just returning a prediction, the app integrates **SHAP (SHapley Additive exPlanations)** to generate a visual, feature-level breakdown of *why* the model arrived at its result, making the prediction transparent rather than a black box. Input validation and median-based imputation (for zero/missing clinical values like Glucose, BMI, and Blood Pressure) ensure realistic, clinically sound predictions. Confidence-based messaging further guides users on when a result may warrant professional consultation.

## Tech Stack

**Backend:** Python, Flask

**Machine Learning:** CatBoost, SHAP, scikit-learn, SMOTE (imbalanced-learn), NumPy

**Visualization:** Matplotlib

**Model Persistence:** Joblib

**Frontend:** HTML, Bootstrap


### Prerequisites
- Python 3.8+

### Installation

```bash
git clone https://github.com/iroy25/diabetes-risk-assessment.git
cd diabetes-risk-assessment
pip install -r requirements.txt
```

### Run Locally

```bash
python app.py
```

Visit `http://127.0.0.1:5000` in your browser.

## 📊 How It Works

1. User submits 8 clinical parameters via the form
2. Backend validates each input against medically plausible ranges
3. Invalid zero-entries are imputed with precomputed medians
4. CatBoost model returns a prediction + class probabilities
5. SHAP TreeExplainer computes feature-level contributions
6. A force plot is rendered, encoded to base64, and displayed alongside the prediction

## Screenshots

<img width="1477" height="857" alt="Screenshot 2026-07-05 192147" src="https://github.com/user-attachments/assets/a6c5b740-36a0-48b5-a02b-1b3ddb65d5f1" />
<img width="1476" height="842" alt="Screenshot 2026-07-05 192158" src="https://github.com/user-attachments/assets/d2ab8b17-ce43-4ed9-9519-d06a49dd3e02" />
<img width="1378" height="647" alt="Screenshot 2026-07-05 192229" src="https://github.com/user-attachments/assets/bc45d378-258f-4b4c-ae82-830947fa1784" />




## ⚠️ Disclaimer

This tool is intended for educational and demonstrative purposes only. It is **not** a substitute for professional medical diagnosis. Predictions — especially borderline ones — should always be followed up with clinical consultation.

