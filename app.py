from flask import Flask, render_template, request
import joblib
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')  # non-interactive backend, must be before pyplot import
import matplotlib.pyplot as plt
import base64
import io

app = Flask(__name__)

model        = joblib.load("models/catboost_final.pkl")
explainer    = shap.TreeExplainer(model)
preprocessing = joblib.load("models/preprocessing.pkl")
medians      = preprocessing["medians"]

FEATURE_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure',
    'SkinThickness', 'Insulin', 'BMI',
    'DiabetesPedigreeFunction', 'Age'
]

def generate_shap_image(shap_values, sample):
    fig, ax = plt.subplots(figsize=(10, 3))
    shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        sample[0],
        feature_names=FEATURE_NAMES,
        matplotlib=True,
        show=False
    )
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        pregnancies = float(request.form["pregnancies"])
        glucose     = float(request.form["glucose"])
        bp          = float(request.form["bp"])
        skin        = float(request.form["skin"])
        insulin     = float(request.form["insulin"])
        bmi         = float(request.form["bmi"])
        dpf         = float(request.form["dpf"])
        age         = float(request.form["age"])

        # Validation
        if not (0  <= pregnancies <= 17):   raise ValueError("Pregnancies must be between 0 and 17.")
        if not (0  <= glucose     <= 199):  raise ValueError("Glucose must be between 0 and 199.")
        if not (0  <= bp          <= 122):  raise ValueError("Blood Pressure must be between 0 and 122.")
        if not (0  <= skin        <= 99):   raise ValueError("Skin Thickness must be between 0 and 99.")
        if not (0  <= insulin     <= 846):  raise ValueError("Insulin must be between 0 and 846.")
        if not (0  <= bmi         <= 67.1): raise ValueError("BMI must be between 0 and 67.1.")
        if not (0.078 <= dpf      <= 2.42): raise ValueError("Diabetes Pedigree Function must be between 0.078 and 2.42.")
        if not (21 <= age         <= 81):   raise ValueError("Age must be between 21 and 81.")

        # Median imputation (mirrors preprocessing.pkl)
        if glucose  == 0: glucose  = medians["Glucose"]
        if bp       == 0: bp       = medians["BloodPressure"]
        if skin     == 0: skin     = medians["SkinThickness"]
        if insulin  == 0: insulin  = medians["Insulin"]
        if bmi      == 0: bmi      = medians["BMI"]

        sample       = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        prediction   = model.predict(sample)[0]
        probabilities = model.predict_proba(sample)[0]
        shap_values  = explainer.shap_values(sample)

        confidence   = round(max(probabilities) * 100, 1)
        result       = "DIABETIC" if prediction == 1 else "NON-DIABETIC"

        if confidence >= 80:
            assessment = "High confidence prediction."
        elif confidence >= 60:
            assessment = "Moderate confidence prediction."
        else:
            assessment = (
                "Borderline prediction. "
                "Clinical consultation and further diagnostic testing are recommended."
            )

        shap_image = generate_shap_image(shap_values, sample)

        return render_template(
            "index.html",
            prediction   = result,
            confidence   = confidence,
            non_diabetic = round(probabilities[0] * 100, 1),
            diabetic     = round(probabilities[1] * 100, 1),
            assessment   = assessment,
            shap_image   = shap_image
        )

    except Exception as e:
        return render_template("index.html", error=str(e))


if __name__ == "__main__":
    app.run(debug=True)