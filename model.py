import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, classification_report

import joblib

# =========================
# CONFIG
# =========================
CSV_PATH = r"first_feature_df_17_12_2025.csv"
MODEL_PATH = r"mortality_xgboost_pipeline.joblib"

TARGET_COL = "MORTALITY_INHOSPITAL"
DROP_COLS = ["MORTALITY_INHOSPITAL", "SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"]

RANDOM_STATE = 42
TEST_SIZE = 0.2
THRESHOLD = 0.5

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(CSV_PATH)

X = df.drop(DROP_COLS, axis=1)
y = df[TARGET_COL]

# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Train size: {len(X_train)}")
print(f"Test size:  {len(X_test)}")

# =========================
# PIPELINE (IMPUTE + XGBOOST)
# =========================
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("model", xgb.XGBClassifier(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])

pipeline.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
y_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= THRESHOLD).astype(int)

auc_score = roc_auc_score(y_test, y_proba)
print(f"\nROC AUC Score: {auc_score:.4f}")

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"XGBoost Pipeline (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - XGBoost Pipeline")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix (threshold=0.5):")
print(cm)

print("\nClassification Report (threshold=0.5):")
print(classification_report(y_test, y_pred, digits=4))

# =========================
# SAVE PIPELINE
# =========================
joblib.dump(
    {"pipeline": pipeline, "threshold": THRESHOLD, "feature_cols": list(X.columns)},
    MODEL_PATH
)
print(f"\nSaved pipeline: {MODEL_PATH}")
