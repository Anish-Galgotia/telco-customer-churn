import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from data_preprocessing import preprocess_pipeline


def train_model(csv_path, model_output_path):
    """
    Train Logistic Regression churn model and save it.
    """

    # preprocess data
    X, y = preprocess_pipeline(csv_path)

    # train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # initialize model
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    # train model
    model.fit(X_train, y_train)

    # predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # evaluation
    print("Model evaluation metrics:")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))
    print("ROC AUC  :", roc_auc_score(y_test, y_prob))

    # create models directory if not exists
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)

    # save model
    joblib.dump(model, model_output_path)
    print(f"Model saved at: {model_output_path}")


if __name__ == "__main__":
    csv_path = "data/raw/telco_churn.csv"
    model_output_path = "models/churn_model.pkl"

    train_model(csv_path, model_output_path)
