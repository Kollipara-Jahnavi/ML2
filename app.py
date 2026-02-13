import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

st.set_page_config(page_title="ML Assignment 2", layout="wide")
st.title("Machine Learning Assignment 2 - Mobile Price Classification")

TARGET_COL = "price_range"

MODEL_FILES = {
    "Logistic Regression": "model/saved_models/logistic_regression.pkl",
    "Decision Tree": "model/saved_models/decision_tree.pkl",
    "KNN": "model/saved_models/knn.pkl",
    "Naive Bayes": "model/saved_models/naive_bayes.pkl",
    "Random Forest": "model/saved_models/random_forest.pkl",
    "XGBoost": "model/saved_models/xgboost.pkl",
}

# ---------- Sidebar ----------
st.sidebar.header("Controls")
model_name = st.sidebar.selectbox("Select Model", list(MODEL_FILES.keys()))

# ---------- Load metrics.csv ----------
st.subheader("Model Comparison Table")
try:
    metrics_df = pd.read_csv("model/metrics.csv")
    st.dataframe(metrics_df)
except Exception as e:
    st.error(f"Could not load model/metrics.csv: {e}")

# ---------- Upload test data ----------
st.subheader("Upload Test CSV")
uploaded = st.file_uploader("Upload CSV file (with or without price_range)", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV file to run predictions.")
    st.stop()

df = pd.read_csv(uploaded)
st.write("Uploaded data shape:", df.shape)
st.dataframe(df.head(10))

# ---------- Load model + scaler ----------
model = joblib.load(MODEL_FILES[model_name])
scaler = joblib.load("model/saved_models/scaler.pkl")

# Separate X and y (if y exists)
if TARGET_COL in df.columns:
    X = df.drop(columns=[TARGET_COL])
    y_true = df[TARGET_COL]
    has_labels = True
else:
    X = df.copy()
    y_true = None
    has_labels = False

# Scale features
X_scaled = scaler.transform(X)

# Predict
y_pred = model.predict(X_scaled)

st.subheader("Predictions")
pred_df = pd.DataFrame({"prediction": y_pred})
st.dataframe(pred_df.head(20))

# ---------- Evaluation (if labels available) ----------
if has_labels:
    st.subheader("Evaluation")
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)

    col1, col2 = st.columns(2)

    with col1:
        st.write("Confusion Matrix")
        fig, ax = plt.subplots()
        ax.imshow(cm)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with col2:
        st.write("Classification Report")
        st.text(report)
else:
    st.warning(
        f"No `{TARGET_COL}` column found in uploaded CSV. "
        "Upload test data with labels to see confusion matrix and report."
    )