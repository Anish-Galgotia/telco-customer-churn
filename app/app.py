import joblib
import pandas as pd
from flask import Flask, request, jsonify

# initialize flask app
app = Flask(__name__)

# load trained model
model = joblib.load("models/churn_model.pkl")


@app.route("/")
def home():
    return "Telco Customer Churn Prediction API is running"


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict churn for a single customer.
    Input: JSON
    Output: churn prediction + probability
    """

    data = request.get_json()

    # convert input json to dataframe
    input_df = pd.DataFrame([data])

    # one-hot encode (same as training)
    input_df = pd.get_dummies(input_df)

    # align columns with training data
    model_features = model.feature_names_in_
    input_df = input_df.reindex(columns=model_features, fill_value=0)

    # prediction
    churn_pred = model.predict(input_df)[0]
    churn_prob = model.predict_proba(input_df)[0][1]

    result = {
        "churn_prediction": "Yes" if churn_pred == 1 else "No",
        "churn_probability": round(float(churn_prob), 4)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
