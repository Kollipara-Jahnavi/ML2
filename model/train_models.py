import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


# -------------------------
# 1. Load dataset
# -------------------------
DATA_PATH = "data/mobile_price.csv"
df = pd.read_csv(DATA_PATH)

TARGET_COL = "price_range"

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]


# -------------------------
# 2. Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -------------------------
# 3. Scaling
# -------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -------------------------
# 4. Define models
# -------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(
        objective="multi:softprob",
        num_class=len(np.unique(y)),
        eval_metric="mlogloss",
        random_state=42
    )
}


# -------------------------
# 5. Metrics function
# -------------------------
def evaluate_model(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)

    # Probability scores for AUC
    y_proba = model.predict_proba(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Multi-class AUC (One-vs-Rest)
    auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")

    return acc, auc, prec, rec, f1, mcc


# -------------------------
# 6. Train models + store results
# -------------------------
results = []

os.makedirs("model/saved_models", exist_ok=True)

for name, model in models.items():
    print(f"Training: {name}")
    model.fit(X_train_scaled, y_train)

    acc, auc, prec, rec, f1, mcc = evaluate_model(model, X_test_scaled, y_test)

    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
        "MCC": round(mcc, 4)
    })

    joblib.dump(model, f"model/saved_models/{name.replace(' ', '_').lower()}.pkl")


# Save scaler
joblib.dump(scaler, "model/saved_models/scaler.pkl")

# Save metrics table
metrics_df = pd.DataFrame(results)
metrics_df.to_csv("model/metrics.csv", index=False)

print("\nTraining complete. Saved:")
print("- model/metrics.csv")
print("- model/saved_models/*.pkl")