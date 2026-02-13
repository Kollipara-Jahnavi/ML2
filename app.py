import streamlit as st
import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

st.set_page_config(page_title="ML Assignment 2", layout="wide")
st.title("Machine Learning Assignment 2 - Classification Models")

st.markdown(
    """
This app will:
- Let you upload a CSV file (test data)
- Let you choose a model
- Show metrics table (from `model/metrics.csv`)
- Show confusion matrix / classification report (once labels are available)
"""
)

# ---------- Load metrics table if available ----------
st.sidebar.header("Model Selection")
model_name = st.sidebar.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

metrics_path = "model/metrics.csv"
try:
    metrics_df = pd.read_csv(metrics_path)
    st.subheader("Model Comparison (metrics.csv)")
    st.dataframe(metrics_df)
except Exception:
    st.warning("metrics.csv not found yet. Run training first to generate model/metrics.csv.")

# ---------- Upload test data ----------
st.subheader("Upload Test CSV")
uploaded = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV to run predictions.")
    st.stop()

df = pd.read_csv(uploaded)
st.write("Shape:", df.shape)
st.dataframe(df.head(10))

# ---------- Predict placeholder ----------
st.subheader("Predictions / Evaluation")
st.info(
    "Next step: connect this to trained models saved in model/saved_models/.\n\n"
    "Your training script will save 6 models + (optional) scaler + label encoder."
)

# If your test CSV contains the label column, you can compute confusion matrix/report AFTER model connection
label_col_guess = "price_range"  # change later if your dataset uses a different target column
if label_col_guess in df.columns:
    st.write(f"Detected label column: `{label_col_guess}` (evaluation will work after model connection).")
else:
    st.write(
        f"No label column `{label_col_guess}` found in uploaded CSV. "
        "Once model is connected, app will still show predictions."
    )