import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from preprocessing import load_data, preprocess_data


os.makedirs("trained_models", exist_ok=True)

print("Loading dataset...")

df = load_data("dataset/weatherAUS.csv")

print("Preprocessing dataset...")

(
    X,
    y,
    encoders,
    num_imputer,
    cat_imputer,
    feature_columns
) = preprocess_data(df)


# Save preprocessing objects

joblib.dump(encoders, "trained_models/encoders.pkl")
joblib.dump(num_imputer, "trained_models/num_imputer.pkl")
joblib.dump(cat_imputer, "trained_models/cat_imputer.pkl")
joblib.dump(feature_columns, "trained_models/feature_columns.pkl")


# Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

joblib.dump(
    (X_test, y_test),
    "trained_models/test_data.pkl"
)


# Standardize features

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

joblib.dump(scaler, "trained_models/scaler.pkl")


# Define five models

models = {

    "logistic": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "decision_tree": DecisionTreeClassifier(
        random_state=42
    ),

    "knn": KNeighborsClassifier(
        n_neighbors=5
    ),

    "naive_bayes": GaussianNB(),

    "svm": CalibratedClassifierCV(
        estimator=LinearSVC(random_state=42),
        cv=3
    )

}


# Train models

for name, model in models.items():

    print(f"Training {name}...")

    if name in ["logistic", "knn", "svm"]:

        model.fit(X_train_scaled, y_train)

    else:

        model.fit(X_train, y_train)

    joblib.dump(
        model,
        f"trained_models/{name}_model.pkl"
    )

    print(f"{name} saved successfully!")


print("\nAll five models trained successfully!")