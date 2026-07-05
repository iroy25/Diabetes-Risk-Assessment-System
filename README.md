

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

## ⚠️ Disclaimer

This tool is intended for educational and demonstrative purposes only. It is **not** a substitute for professional medical diagnosis. Predictions — especially borderline ones — should always be followed up with clinical consultation.

