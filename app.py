import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
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
#model_name = st.sidebar.selectbox("Select Model", list(MODEL_FILES.keys()))
model_name = st.sidebar.radio(
    "Select Model",
    list(MODEL_FILES.keys())
)

# ---------- Load metrics.csv ----------
st.subheader("Model Comparison Table")
try:
    metrics_df = pd.read_csv("model/metrics.csv")
    st.dataframe(metrics_df)
except Exception as e:
    st.error(f"Could not load model/metrics.csv: {e}")

# ---------- Load data (default or upload) ----------
st.subheader("Dataset Input")

use_default = st.checkbox("Use default dataset from repository (mobile_price.csv)", value=True)

if use_default:
    df = pd.read_csv("data/mobile_price.csv")
    st.success("Loaded default dataset from repository: data/mobile_price.csv")
else:
    uploaded = st.file_uploader("Upload CSV file (with or without price_range)", type=["csv"])

    if uploaded is None:
        st.info("Upload a CSV file to run predictions.")
        st.stop()

    df = pd.read_csv(uploaded)
    st.success("Uploaded dataset successfully!")

st.write("Dataset shape:", df.shape)
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

# ---- SAFETY CHECK ----
expected_cols = list(pd.read_csv("data/mobile_price.csv").drop(columns=[TARGET_COL]).columns)

missing = [c for c in expected_cols if c not in X.columns]
extra = [c for c in X.columns if c not in expected_cols]

if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# reorder columns to match training
X = X[expected_cols]

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
        ax.set_xticks(range(cm.shape[1]))
        ax.set_yticks(range(cm.shape[0]))
        ax.set_title(model_name)
        st.pyplot(fig)

    with col2:
        st.write("Classification Report")
        st.text(report)
else:
    st.warning(
        f"No `{TARGET_COL}` column found in uploaded CSV. "
        "Upload test data with labels to see confusion matrix and report."
    )

# ---------- ROC Curve (Multi-class OvR) ----------
st.subheader("ROC Curve (One-vs-Rest)")

y_proba = model.predict_proba(X_scaled)
classes = np.unique(y_true)

y_true_bin = label_binarize(y_true, classes=classes)

fig2, ax2 = plt.subplots()

for i, c in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, label=f"Class {c} (AUC={roc_auc:.3f})")

ax2.plot([0, 1], [0, 1], linestyle="--", label="Random")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title(f"ROC Curve - {model_name}")
ax2.legend()

st.pyplot(fig2)    