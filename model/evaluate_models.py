import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)


# Load test data

X_test, y_test = joblib.load(
    "trained_models/test_data.pkl"
)

scaler = joblib.load(
    "trained_models/scaler.pkl"
)


# Model dictionary

models = {

    "Logistic Regression": "logistic_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "KNN": "knn_model.pkl",
    "Naive Bayes": "naive_bayes_model.pkl",
    "Support Vector Machine": "svm_model.pkl"

}


results = []


for model_name, file_name in models.items():

    print(f"\nEvaluating {model_name}...")

    model = joblib.load(
        f"trained_models/{file_name}"
    )

    if model_name in [
        "Logistic Regression",
        "KNN",
        "Support Vector Machine"
    ]:

        X_input = scaler.transform(X_test)

    else:

        X_input = X_test

    y_pred = model.predict(X_input)

    y_prob = model.predict_proba(X_input)[:, 1]

    results.append({

        "Model": model_name,

        "Accuracy": accuracy_score(
            y_test,
            y_pred
        ),

        "Precision": precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "AUC": roc_auc_score(
            y_test,
            y_prob
        ),

        "MCC": matthews_corrcoef(
            y_test,
            y_pred
        )

    })


results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\nMODEL PERFORMANCE COMPARISON")

print(results_df.to_string(index=False))

results_df.to_csv(
    "trained_models/model_comparison.csv",
    index=False
)

print("\nResults saved successfully!")