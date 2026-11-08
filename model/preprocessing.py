import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


def load_data(path):
    return pd.read_csv(path)


def preprocess_data(df):

    df = df.copy()

    # Remove rows where target is missing
    df = df.dropna(subset=["RainTomorrow"]).copy()

    # Convert target into numerical values
    df["RainTomorrow"] = df["RainTomorrow"].map({
        "Yes": 1,
        "No": 0
    })

    df = df.dropna(subset=["RainTomorrow"])
    df["RainTomorrow"] = df["RainTomorrow"].astype(int)

    # Convert RainToday into numerical values
    df["RainToday"] = df["RainToday"].map({
        "Yes": 1,
        "No": 0
    })

    df["RainToday"] = df["RainToday"].fillna(0).astype(int)

    # Separate features and target
    y = df["RainTomorrow"]
    X = df.drop("RainTomorrow", axis=1)

    # Extract date features
    X["Date"] = pd.to_datetime(X["Date"])

    X["Year"] = X["Date"].dt.year
    X["Month"] = X["Date"].dt.month
    X["Day"] = X["Date"].dt.day

    X = X.drop("Date", axis=1)

    # Identify numerical and categorical columns
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    # Handle missing numerical values
    num_imputer = SimpleImputer(strategy="median")
    X[num_cols] = num_imputer.fit_transform(X[num_cols])

    # Handle missing categorical values
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

    # Encode categorical columns
    encoders = {}

    for col in cat_cols:

        le = LabelEncoder()

        X[col] = le.fit_transform(X[col].astype(str))

        encoders[col] = le

    feature_columns = X.columns.tolist()

    return (
        X,
        y,
        encoders,
        num_imputer,
        cat_imputer,
        feature_columns
    )