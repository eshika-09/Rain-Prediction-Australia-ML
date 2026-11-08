import streamlit as st
import pandas as pd
import numpy as np
import joblib

from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Rain Prediction Dashboard",
    page_icon="🌦️",
    layout="wide"
)


# =====================================================
# LOAD MODELS AND ARTIFACTS
# =====================================================

@st.cache_resource
def load_artifacts():

    models = {

        "Logistic Regression": joblib.load(
            "trained_models/logistic_model.pkl"
        ),

        "Decision Tree": joblib.load(
            "trained_models/decision_tree_model.pkl"
        ),

        "KNN": joblib.load(
            "trained_models/knn_model.pkl"
        ),

        "Naive Bayes": joblib.load(
            "trained_models/naive_bayes_model.pkl"
        ),

        "Support Vector Machine": joblib.load(
            "trained_models/svm_model.pkl"
        )

    }

    scaler = joblib.load(
        "trained_models/scaler.pkl"
    )

    encoders = joblib.load(
        "trained_models/encoders.pkl"
    )

    num_imputer = joblib.load(
        "trained_models/num_imputer.pkl"
    )

    cat_imputer = joblib.load(
        "trained_models/cat_imputer.pkl"
    )

    feature_columns = joblib.load(
        "trained_models/feature_columns.pkl"
    )

    X_test, y_test = joblib.load(
        "trained_models/test_data.pkl"
    )

    return (
        models,
        scaler,
        encoders,
        num_imputer,
        cat_imputer,
        feature_columns,
        X_test,
        y_test
    )


try:

    (
        models,
        scaler,
        encoders,
        num_imputer,
        cat_imputer,
        feature_columns,
        X_test,
        y_test
    ) = load_artifacts()

except Exception as e:

    st.error(
        f"Unable to load trained models: {e}"
    )

    st.info(
        "Run python model/train_models.py from the project root first."
    )

    st.stop()


# =====================================================
# PREPROCESSING FUNCTION
# =====================================================

def prepare_input_data(input_df):

    input_df = input_df.copy()

    # Ensure all expected columns are present
    for col in feature_columns:

        if col not in input_df.columns:

            input_df[col] = np.nan

    input_df = input_df[feature_columns]

    num_cols = list(num_imputer.feature_names_in_)

    cat_cols = list(cat_imputer.feature_names_in_)

    # Impute numerical columns
    input_df[num_cols] = num_imputer.transform(
        input_df[num_cols]
    )

    # Impute categorical columns
    input_df[cat_cols] = cat_imputer.transform(
        input_df[cat_cols]
    )

    # Encode categorical columns
    for col in cat_cols:

        encoder = encoders[col]

        known_values = set(encoder.classes_)

        input_df[col] = input_df[col].astype(str)

        input_df[col] = input_df[col].apply(
            lambda value: value
            if value in known_values
            else encoder.classes_[0]
        )

        input_df[col] = encoder.transform(
            input_df[col]
        )

    return input_df


# =====================================================
# MODEL INPUT FUNCTION
# =====================================================

def get_model_input(model_name, data):

    if model_name in [
        "Logistic Regression",
        "KNN",
        "Support Vector Machine"
    ]:

        return scaler.transform(data)

    return data


# =====================================================
# METRICS FUNCTION
# =====================================================

def calculate_metrics(model, X, y, model_name):

    X_input = get_model_input(
        model_name,
        X
    )

    predictions = model.predict(X_input)

    probabilities = model.predict_proba(
        X_input
    )[:, 1]

    metrics = {

        "Accuracy": accuracy_score(y, predictions),

        "AUC": roc_auc_score(
            y,
            probabilities
        ),

        "Precision": precision_score(
            y,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y,
            predictions,
            zero_division=0
        ),

        "F1 Score": f1_score(
            y,
            predictions,
            zero_division=0
        ),

        "MCC": matthews_corrcoef(
            y,
            predictions
        )

    }

    return metrics, predictions


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🌦️ Rain Prediction")

page = st.sidebar.radio(
    "Navigation",
    [
        "Prediction Dashboard",
        "Model Evaluation"
    ]
)

st.sidebar.markdown("---")

selected_model_name = st.sidebar.selectbox(
    "Select Machine Learning Model",
    list(models.keys())
)

selected_model = models[selected_model_name]


# =====================================================
# PAGE 1: PREDICTION DASHBOARD
# =====================================================

if page == "Prediction Dashboard":

    st.title("🌦️ Rain Prediction Dashboard")

    st.write(
        "Predict whether it will rain tomorrow in Australia "
        "using machine learning."
    )

    st.markdown("---")

    # MODEL PERFORMANCE

    st.subheader(
        f"Model Performance: {selected_model_name}"
    )

    metrics, _ = calculate_metrics(
        selected_model,
        X_test,
        y_test,
        selected_model_name
    )

    cols = st.columns(6)

    for col, (name, value) in zip(
        cols,
        metrics.items()
    ):

        col.metric(
            name,
            f"{value:.4f}"
        )

    metric_df = pd.DataFrame(
        metrics.items(),
        columns=["Metric", "Score"]
    )

    fig, ax = plt.subplots(figsize=(9, 4))

    sns.barplot(
        data=metric_df,
        x="Metric",
        y="Score",
        ax=ax
    )

    ax.set_ylim(0, 1)
    ax.set_title("Model Performance")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")

    plt.xticks(rotation=20)

    st.pyplot(fig)

    plt.close(fig)

    st.markdown("---")

    # USER INPUT

    st.subheader("Enter Weather Details")

    col1, col2, col3 = st.columns(3)

    with col1:

        MinTemp = st.number_input(
            "Minimum Temperature (°C)",
            value=10.0
        )

        MaxTemp = st.number_input(
            "Maximum Temperature (°C)",
            value=20.0
        )

        Rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            value=0.0
        )

        WindSpeed9am = st.number_input(
            "Wind Speed at 9am",
            min_value=0.0,
            value=10.0
        )

    with col2:

        WindSpeed3pm = st.number_input(
            "Wind Speed at 3pm",
            min_value=0.0,
            value=15.0
        )

        Humidity9am = st.slider(
            "Humidity at 9am (%)",
            0,
            100,
            50
        )

        Humidity3pm = st.slider(
            "Humidity at 3pm (%)",
            0,
            100,
            50
        )

        Pressure9am = st.number_input(
            "Pressure at 9am",
            value=1010.0
        )

    with col3:

        Pressure3pm = st.number_input(
            "Pressure at 3pm",
            value=1010.0
        )

        Temp9am = st.number_input(
            "Temperature at 9am (°C)",
            value=15.0
        )

        Temp3pm = st.number_input(
            "Temperature at 3pm (°C)",
            value=18.0
        )

        RainToday = st.selectbox(
            "Rain Today?",
            ["No", "Yes"]
        )

    RainToday = 1 if RainToday == "Yes" else 0

    # DEFAULT VALUES

    today = datetime.today()

    default_values = {

        "Year": today.year,
        "Month": today.month,
        "Day": today.day,

        "Location": "Sydney",

        "Evaporation": 5.0,
        "Sunshine": 7.0,

        "WindGustDir": "N",
        "WindGustSpeed": 30.0,

        "WindDir9am": "N",
        "WindDir3pm": "N",

        "Cloud9am": 5.0,
        "Cloud3pm": 5.0

    }

    input_dict = {

        "MinTemp": MinTemp,
        "MaxTemp": MaxTemp,
        "Rainfall": Rainfall,

        "WindSpeed9am": WindSpeed9am,
        "WindSpeed3pm": WindSpeed3pm,

        "Humidity9am": Humidity9am,
        "Humidity3pm": Humidity3pm,

        "Pressure9am": Pressure9am,
        "Pressure3pm": Pressure3pm,

        "Temp9am": Temp9am,
        "Temp3pm": Temp3pm,

        "RainToday": RainToday

    }

    input_dict.update(default_values)

    input_df = pd.DataFrame([input_dict])

    with st.expander("View Input Details"):

        st.dataframe(
            input_df,
            use_container_width=True
        )

    # PREDICTION

    if st.button(
        "Predict Rain Tomorrow",
        type="primary",
        use_container_width=True
    ):

        try:

            input_processed = prepare_input_data(
                input_df
            )

            model_input = get_model_input(
                selected_model_name,
                input_processed
            )

            prediction = selected_model.predict(
                model_input
            )[0]

            probability = selected_model.predict_proba(
                model_input
            )[0][1]

            st.markdown("---")

            st.subheader("Prediction Result")

            col1, col2 = st.columns(2)

            with col1:

                if prediction == 1:

                    st.error(
                        "🌧️ Rain is predicted tomorrow."
                    )

                else:

                    st.success(
                        "☀️ No rain is predicted tomorrow."
                    )

            with col2:

                st.metric(
                    "Rain Probability",
                    f"{probability * 100:.2f}%"
                )

            st.progress(
                float(probability)
            )

            st.caption(
                f"Prediction generated using {selected_model_name}"
            )

        except Exception as e:

            st.error(
                f"Prediction error: {e}"
            )


# =====================================================
# PAGE 2: MODEL EVALUATION
# =====================================================

elif page == "Model Evaluation":

    st.title("📊 Model Evaluation")

    st.write(
        "Upload a CSV file containing weather observations "
        "and the actual RainTomorrow column."
    )

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload CSV Dataset",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(uploaded_file)

            st.subheader("Dataset Preview")

            st.dataframe(
                df.head(),
                use_container_width=True
            )

            if "RainTomorrow" not in df.columns:

                st.error(
                    "RainTomorrow column is missing."
                )

                st.stop()

            # Convert target values
            target = (
                df["RainTomorrow"]
                .astype("string")
                .str.strip()
                .map({
                    "Yes": 1,
                    "No": 0
                })
            )

            valid_rows = target.notna()

            df = df.loc[valid_rows].copy()

            y_true = target.loc[
                valid_rows
            ].astype(int)

            if len(df) == 0:

                st.error(
                    "No valid target values found."
                )

                st.stop()

            # Separate features
            X_input = df.drop(
                "RainTomorrow",
                axis=1
            )

            # Extract date features
            if "Date" in X_input.columns:

                dates = pd.to_datetime(
                    X_input["Date"],
                    errors="coerce"
                )

                X_input["Year"] = dates.dt.year
                X_input["Month"] = dates.dt.month
                X_input["Day"] = dates.dt.day

                X_input = X_input.drop(
                    "Date",
                    axis=1
                )

            # Convert RainToday
            if "RainToday" in X_input.columns:

                X_input["RainToday"] = (
                    X_input["RainToday"]
                    .astype("string")
                    .str.strip()
                    .map({
                        "Yes": 1,
                        "No": 0
                    })
                )

            # Prepare features
            for col in feature_columns:

                if col not in X_input.columns:

                    X_input[col] = np.nan

            X_input = X_input[feature_columns]

            X_processed = prepare_input_data(
                X_input
            )

            model_input = get_model_input(
                selected_model_name,
                X_processed
            )

            # Predictions
            y_pred = selected_model.predict(
                model_input
            )

            y_prob = selected_model.predict_proba(
                model_input
            )[:, 1]

            # Metrics
            st.markdown("---")

            st.subheader(
                f"Results: {selected_model_name}"
            )

            metrics = {

                "Accuracy": accuracy_score(
                    y_true,
                    y_pred
                ),

                "AUC": roc_auc_score(
                    y_true,
                    y_prob
                ) if len(np.unique(y_true)) == 2 else np.nan,

                "Precision": precision_score(
                    y_true,
                    y_pred,
                    zero_division=0
                ),

                "Recall": recall_score(
                    y_true,
                    y_pred,
                    zero_division=0
                ),

                "F1 Score": f1_score(
                    y_true,
                    y_pred,
                    zero_division=0
                ),

                "MCC": matthews_corrcoef(
                    y_true,
                    y_pred
                )

            }

            metric_df = pd.DataFrame(
                metrics.items(),
                columns=["Metric", "Score"]
            )

            st.dataframe(
                metric_df,
                use_container_width=True
            )

            # Confusion matrix
            st.markdown("---")

            st.subheader("Confusion Matrix")

            cm = confusion_matrix(
                y_true,
                y_pred,
                labels=[0, 1]
            )

            fig, ax = plt.subplots(
                figsize=(6, 4)
            )

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=[
                    "No Rain",
                    "Rain"
                ],
                yticklabels=[
                    "No Rain",
                    "Rain"
                ],
                ax=ax
            )

            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")

            st.pyplot(fig)

            plt.close(fig)

            # Classification report
            st.markdown("---")

            st.subheader("Classification Report")

            report = classification_report(
                y_true,
                y_pred,
                labels=[0, 1],
                target_names=[
                    "No Rain",
                    "Rain"
                ],
                output_dict=True,
                zero_division=0
            )

            report_df = pd.DataFrame(
                report
            ).transpose()

            st.dataframe(
                report_df,
                use_container_width=True
            )

            # Download predictions
            output_df = df.copy()

            output_df["Actual RainTomorrow"] = y_true.values

            output_df["Predicted RainTomorrow"] = y_pred

            output_df["Rain Probability"] = y_prob

            st.download_button(
                "Download Prediction Results",
                data=output_df.to_csv(index=False),
                file_name="rain_prediction_results.csv",
                mime="text/csv"
            )

        except Exception as e:

            st.error(
                f"Evaluation failed: {e}"
            )


# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;">
        <p>Rain Prediction in Australia</p>
        <p>Machine Learning Project</p>
        <p>
        Logistic Regression | Decision Tree | KNN |
        Naive Bayes | Support Vector Machine
        </p>
    </div>
    """,
    unsafe_allow_html=True
)