import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import roc_curve, auc

DATA_PATH = "data/mobile_price.csv"
TARGET_COL = "price_range"

# Choose which model ROC to plot (must exist in saved_models)
MODEL_FILE = "model/saved_models/xgboost.pkl"  # or logistic_regression.pkl

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Load model (or train again if you prefer)
model = joblib.load(MODEL_FILE)

# Predict probabilities
y_score = model.predict_proba(X_test_s)

# One-vs-Rest binarization
classes = np.unique(y)
y_test_bin = label_binarize(y_test, classes=classes)

# Plot ROC curve for each class
plt.figure()
for i, c in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {c} (AUC={roc_auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multi-class ROC (One-vs-Rest)")
plt.legend()
plt.tight_layout()

# Save image for README/PDF
plt.savefig("model/roc_curve.png", dpi=200)
print("Saved: model/roc_curve.png")

