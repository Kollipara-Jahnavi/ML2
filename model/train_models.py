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
def evaluate_model(model, X_scaled, y_true):
    y_pred = model.predict(X_scaled)

    # AUC needs probabilities
    y_proba = model.predict_proba(X_scaled)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")

    return acc, auc, prec, rec, f1, mcc

# -------------------------
# 6. Train models + store results
# -------------------------
results = []

os.makedirs("model/saved_models", exist_ok=True)

for name, model in models.items():
    print(f"Training: {name}")
    model.fit(X_train_scaled, y_train)

    train_acc, train_auc, train_prec, train_rec, train_f1, train_mcc = evaluate_model(model, X_train_scaled, y_train)
    test_acc, test_auc, test_prec, test_rec, test_f1, test_mcc = evaluate_model(model, X_test_scaled, y_test)

    results.append({
        "Model": name,

        "Train Accuracy": round(train_acc, 4),
        "Train AUC": round(train_auc, 4),
        "Train Precision": round(train_prec, 4),
        "Train Recall": round(train_rec, 4),
        "Train F1": round(train_f1, 4),
        "Train MCC": round(train_mcc, 4),

        "Test Accuracy": round(test_acc, 4),
        "Test AUC": round(test_auc, 4),
        "Test Precision": round(test_prec, 4),
        "Test Recall": round(test_rec, 4),
        "Test F1": round(test_f1, 4),
        "Test MCC": round(test_mcc, 4),
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