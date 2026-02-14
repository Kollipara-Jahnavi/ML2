import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

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
st.subheader("Model Comparison Table (Test vs Train)")

try:
    metrics_df = pd.read_csv("model/metrics.csv").round(2)

    html = """
    <style>
    table {width:100%; border-collapse: collapse;}
    th, td {border: 1px solid #ddd; padding: 8px; text-align: center;}
    th {background-color: #f2f2f2; font-weight: bold;}
    </style>
    """

    html += "<table>"
    html += """
        <tr>
            <th rowspan="2">Model</th>
            <th colspan="6">Test</th>
            <th colspan="6">Train</th>
        </tr>
        <tr>
            <th>Accuracy</th><th>AUC</th><th>Precision</th><th>Recall</th><th>F1 Score</th><th>MCC</th>
            <th>Accuracy</th><th>AUC</th><th>Precision</th><th>Recall</th><th>F1 Score</th><th>MCC</th>
        </tr>
    """

    for _, row in metrics_df.iterrows():
        html += f"""
        <tr>
            <td><b>{row['Model']}</b></td>
            <td>{row['Test Accuracy']}</td>
            <td>{row['Test AUC']}</td>
            <td>{row['Test Precision']}</td>
            <td>{row['Test Recall']}</td>
            <td>{row['Test F1']}</td>
            <td>{row['Test MCC']}</td>

            <td>{row['Train Accuracy']}</td>
            <td>{row['Train AUC']}</td>
            <td>{row['Train Precision']}</td>
            <td>{row['Train Recall']}</td>
            <td>{row['Train F1']}</td>
            <td>{row['Train MCC']}</td>
        </tr>
        """

    html += "</table>"

    components.html(html, height=420, scrolling=True)

except Exception as e:
    st.error(f"Could not load model/metrics.csv: {e}")

# ---------- Load data (default or upload) ----------
st.subheader("Dataset Input (Test Data)")

data_mode = st.radio(
    "Choose input method:",
    ["Upload CSV", "Use built-in test.csv (data/test.csv)"],
    index=0
)

if data_mode == "Upload CSV":
    uploaded = st.file_uploader("Upload CSV file (with or without price_range)", type=["csv"])
    if uploaded is None:
        st.warning("Please upload a CSV file to continue.")
        st.stop()
    df = pd.read_csv(uploaded)
    st.success("Uploaded dataset successfully!")

else:
    df = pd.read_csv("data/test.csv")
    st.success("Loaded built-in dataset: data/test.csv")

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
expected_cols = list(pd.read_csv("data/train.csv").drop(columns=[TARGET_COL]).columns)

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
st.write(f"Total predictions generated: **{len(pred_df)}** rows")

# Show compact preview
st.dataframe(pred_df.head(20), use_container_width=False)

# Add class distribution (very useful)
st.markdown("**Prediction distribution:**")
st.dataframe(pred_df["prediction"].value_counts().rename_axis("class").reset_index(name="count"))

# ---------- Evaluation (if labels available) ----------
if has_labels:
    st.subheader("Evaluation (Test Data)")

    # --- Core metrics (macro averaged for multiclass) ---
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    # AUC (multiclass OvR) requires probabilities
    try:
        y_proba = model.predict_proba(X_scaled)
        auc_ovr = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except Exception:
        auc_ovr = np.nan

    # --- Show 6 metrics nicely ---
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Accuracy", f"{acc:.2f}")
    c2.metric("AUC (OvR)", f"{auc_ovr:.2f}" if not np.isnan(auc_ovr) else "NA")
    c3.metric("Precision (Macro)", f"{prec:.2f}")
    c4.metric("Recall (Macro)", f"{rec:.2f}")
    c5.metric("F1 (Macro)", f"{f1:.2f}")
    c6.metric("MCC", f"{mcc:.2f}")

    # --- Confusion Matrix + Report table ---
    cm = confusion_matrix(y_true, y_pred)

    left, right = st.columns([1, 1])

    with left:
        st.markdown("### Confusion Matrix")
        fig, ax = plt.subplots()
        ax.imshow(cm)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(model_name)
        ax.set_xticks(range(cm.shape[1]))
        ax.set_yticks(range(cm.shape[0]))
        # show numbers inside cells
        threshold = cm.max() / 2

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] < threshold else "black"
                ax.text(j, i, cm[i, j], ha="center", va="center", color=color, fontweight="bold")
        
        st.pyplot(fig)

    with right:
        st.markdown("### Classification Report (Per Class)")

        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report_dict).T

        # Remove accuracy row
        report_df = report_df.drop(index="accuracy", errors="ignore")

        # Keep only useful columns
        keep_cols = ["precision", "recall", "f1-score", "support"]
        report_df = report_df[keep_cols].round(3)

        # Style with HTML
        styled_html = report_df.to_html(classes="report-table")

        st.markdown(
        """
        <style>
        .report-table {
        width: 100%;
        border-collapse: collapse;
        text-align: center;
        font-size: 14px;
        }
        .report-table th {
        background-color: #f2f2f2;
        font-weight: bold;
        border: 1px solid #ddd;
        padding: 8px;
        }
        .report-table td {
        border: 1px solid #ddd;
        padding: 8px;
        }
        .report-table tbody th {
        font-weight: bold;
        background-color: #fafafa;
        }
        </style>
        """,
        unsafe_allow_html=True
        )

        st.markdown(styled_html, unsafe_allow_html=True)

else:
    st.warning(
        f"No `{TARGET_COL}` column found. "
        "Upload test data with labels (price_range) to see metrics, confusion matrix, and report."
    )
# ---------- ROC Curve (Multi-class OvR) ----------
if has_labels:
    st.subheader("ROC Curve (One-vs-Rest)")

    y_proba = model.predict_proba(X_scaled)
    classes = np.unique(y_true)
    y_true_bin = label_binarize(y_true, classes=classes)

    fig2, ax2 = plt.subplots(figsize=(3, 2))
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, label=f"Class {c} (AUC={roc_auc:.3f})")

    ax2.plot([0, 1], [0, 1], linestyle="--", label="Random")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title(f"ROC Curve - {model_name}")
    ax2.legend(fontsize=6)
    st.pyplot(fig2)
else:
    st.info("ROC curve requires true labels (price_range). Upload a CSV that includes price_range to view ROC.")