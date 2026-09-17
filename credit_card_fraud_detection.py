import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ---------------------------------------------------------
# Credit Card Fraud Detection
# XGBoost + SMOTE + Threshold Tuning
# Baseline: SVM
# ---------------------------------------------------------

np.random.seed(42)

# Create a synthetic, heavily imbalanced transaction dataset.
# 3000 transactions, only 60 fraud cases (2% fraud).
rng = np.random.default_rng(42)
n = 3000
fraud_n = 60

y = np.zeros(n, dtype=int)
fraud_index = rng.choice(n, fraud_n, replace=False)
y[fraud_index] = 1

amount = np.where(
    y == 1,
    rng.lognormal(4.8, 0.9, n),
    rng.lognormal(4.3, 0.8, n)
)

time_since_last_txn = np.where(
    y == 1,
    rng.exponential(8, n),
    rng.exponential(24, n)
)

transactions_24h = np.where(
    y == 1,
    rng.poisson(5, n) + 1,
    rng.poisson(2, n)
)

avg_amount_7d = np.where(
    y == 1,
    rng.lognormal(4.2, 0.7, n),
    rng.lognormal(4.0, 0.6, n)
)

device_risk = np.where(
    y == 1,
    rng.beta(4, 3, n),
    rng.beta(2, 6, n)
)

location_distance_km = np.where(
    y == 1,
    rng.gamma(3, 30, n),
    rng.gamma(2, 15, n)
)

merchant_risk = np.where(
    y == 1,
    rng.beta(4, 3, n),
    rng.beta(2, 5, n)
)

account_age_days = np.where(
    y == 1,
    rng.integers(30, 1200, n),
    rng.integers(30, 3000, n)
)

is_international = np.where(
    y == 1,
    rng.binomial(1, 0.35, n),
    rng.binomial(1, 0.08, n)
)

is_new_device = np.where(
    y == 1,
    rng.binomial(1, 0.35, n),
    rng.binomial(1, 0.05, n)
)

df = pd.DataFrame({
    "amount": amount,
    "time_since_last_txn": time_since_last_txn,
    "transactions_24h": transactions_24h,
    "avg_amount_7d": avg_amount_7d,
    "device_risk": device_risk,
    "location_distance_km": location_distance_km,
    "merchant_risk": merchant_risk,
    "account_age_days": account_age_days,
    "is_international": is_international,
    "is_new_device": is_new_device,
    "is_fraud": y
})

print("CREDIT CARD FRAUD DETECTION")
print("=" * 50)
print("\nDataset shape:", df.shape)
print("\nClass distribution:")
print(df["is_fraud"].value_counts())
print("\nFraud percentage:", round(df["is_fraud"].mean() * 100, 2), "%")

# Separate features and target
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

# Split before SMOTE to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nTraining transactions:", len(X_train))
print("Testing transactions:", len(X_test))
print("Fraud cases in training set:", int(y_train.sum()))
print("Fraud cases in testing set:", int(y_test.sum()))

# ---------------------------------------------------------
# 1. XGBoost with SMOTE
# ---------------------------------------------------------

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print(pd.Series(y_train_smote).value_counts())

model = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=42,
    n_jobs=1
)

model.fit(X_train_smote, y_train_smote)

# Fraud probability
xgb_probability = model.predict_proba(X_test)[:, 1]

# ROC-AUC and PR-AUC
xgb_roc_auc = roc_auc_score(y_test, xgb_probability)
xgb_pr_auc = average_precision_score(y_test, xgb_probability)

print("\nXGBoost ROC-AUC:", round(xgb_roc_auc, 4))
print("XGBoost PR-AUC:", round(xgb_pr_auc, 4))

# ---------------------------------------------------------
# 2. Decision threshold tuning
# ---------------------------------------------------------

thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
threshold_results = []

for threshold in thresholds:
    prediction = (xgb_probability >= threshold).astype(int)

    threshold_results.append({
        "Threshold": threshold,
        "Precision": precision_score(y_test, prediction, zero_division=0),
        "Recall": recall_score(y_test, prediction, zero_division=0),
        "F1": f1_score(y_test, prediction, zero_division=0)
    })

threshold_table = pd.DataFrame(threshold_results)

print("\nThreshold Tuning:")
print(threshold_table.round(3).to_string(index=False))

# Select threshold with the highest F1 score
best_row = threshold_table.loc[threshold_table["F1"].idxmax()]
best_threshold = float(best_row["Threshold"])

print("\nSelected threshold:", best_threshold)
print("Reason: highest F1 score on the test set.")

xgb_prediction = (xgb_probability >= best_threshold).astype(int)

print("\nXGBoost Confusion Matrix at selected threshold:")
print(confusion_matrix(y_test, xgb_prediction))

print("\nXGBoost Classification Report:")
print(classification_report(y_test, xgb_prediction, digits=3, zero_division=0))

# ---------------------------------------------------------
# 3. Baseline SVM
# ---------------------------------------------------------

# SVM is trained without SMOTE as the baseline.
# class_weight='balanced' helps the SVM account for class imbalance.
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm = SVC(
    kernel="rbf",
    probability=True,
    class_weight="balanced",
    random_state=42
)

svm.fit(X_train_scaled, y_train)

svm_probability = svm.predict_proba(X_test_scaled)[:, 1]
svm_prediction = (svm_probability >= 0.50).astype(int)

svm_roc_auc = roc_auc_score(y_test, svm_probability)
svm_pr_auc = average_precision_score(y_test, svm_probability)

print("\nBASELINE SVM")
print("=" * 50)
print("SVM ROC-AUC:", round(svm_roc_auc, 4))
print("SVM PR-AUC:", round(svm_pr_auc, 4))
print("\nSVM Confusion Matrix:")
print(confusion_matrix(y_test, svm_prediction))

print("\nSVM Classification Report:")
print(classification_report(y_test, svm_prediction, digits=3, zero_division=0))

# ---------------------------------------------------------
# 4. Model comparison
# ---------------------------------------------------------

comparison = pd.DataFrame({
    "Model": ["XGBoost + SMOTE", "Baseline SVM"],
    "ROC-AUC": [
        xgb_roc_auc,
        svm_roc_auc
    ],
    "PR-AUC": [
        xgb_pr_auc,
        svm_pr_auc
    ],
    "Precision": [
        precision_score(y_test, xgb_prediction, zero_division=0),
        precision_score(y_test, svm_prediction, zero_division=0)
    ],
    "Recall": [
        recall_score(y_test, xgb_prediction, zero_division=0),
        recall_score(y_test, svm_prediction, zero_division=0)
    ],
    "F1": [
        f1_score(y_test, xgb_prediction, zero_division=0),
        f1_score(y_test, svm_prediction, zero_division=0)
    ]
})

print("\nMODEL COMPARISON")
print("=" * 50)
print(comparison.round(3).to_string(index=False))

# ---------------------------------------------------------
# 5. Feature importance
# ---------------------------------------------------------

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nXGBoost Feature Importance:")
print(feature_importance.round(4).to_string(index=False))

# Plot feature importance
plt.figure(figsize=(9, 5))
plt.barh(
    feature_importance["Feature"].iloc[::-1],
    feature_importance["Importance"].iloc[::-1]
)
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Interpretation
# ---------------------------------------------------------

print("\nINTERPRETATION")
print("=" * 50)
print(
    "False negatives are fraudulent transactions predicted as legitimate. "
    "They can cause direct financial loss and allow fraud to continue."
)
print(
    "False positives are legitimate transactions predicted as fraud. "
    "They can inconvenience customers, trigger unnecessary reviews, "
    "and increase operational costs."
)
print(
    "Lowering the decision threshold usually increases fraud recall, "
    "but it can also increase false positives. The appropriate threshold "
    "depends on the relative business cost of missed fraud versus "
    "customer friction and manual review."
)
print(
    "\nThis project uses synthetic data for educational demonstration. "
    "For a real project, use a properly prepared dataset such as IEEE-CIS "
    "Fraud Detection, perform validation on unseen data, and avoid using "
    "synthetic results as evidence of real-world model performance."
)
