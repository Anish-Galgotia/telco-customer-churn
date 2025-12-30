import pandas as pd


def load_data(csv_path):
    #Load raw telco churn dataset from CSV
    df = pd.read_csv(csv_path)
    return df


def clean_total_charges(df):
    # Fix TotalCharges column and handle missing values
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["tenure"] == 0, "TotalCharges"] = 0

    return df


def drop_unnecessary_columns(df):
    # Drop columns that are not useful for prediction
    df = df.copy()

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    return df


def encode_features(df):
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, drop_first=True)
    return df_encoded


def prepare_features_and_target(df):
    # Split dataframe into X (features) and y (target)
    X = df.drop("Churn_Yes", axis=1)
    y = df["Churn_Yes"]

    return X, y


def preprocess_pipeline(csv_path):
 
    df = load_data(csv_path)
    df = clean_total_charges(df)
    df = drop_unnecessary_columns(df)
    df = encode_features(df)

    X, y = prepare_features_and_target(df)

    return X, y


