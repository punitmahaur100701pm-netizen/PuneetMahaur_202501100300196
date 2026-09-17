import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

# Data of 10 patients
data = {
    "Age": [65, 45, 70, 30, 55, 80, 40, 60, 72, 35],
    "Heart_Rate": [95, 78, 102, 70, 88, 110, 75, 92, 105, 72],
    "Blood_Pressure": [150, 120, 160, 110, 135, 170, 118, 145, 155, 115],
    "Previous_Visits": [4, 1, 5, 0, 2, 6, 1, 3, 5, 0],
    "Length_of_Stay": [8, 3, 10, 2, 5, 12, 3, 7, 9, 2],
    "Diagnosis_Code": [1, 0, 1, 0, 1, 1, 0, 1, 1, 0],
    "Readmitted": [1, 0, 1, 0, 1, 1, 0, 1, 1, 0]
}

df = pd.DataFrame(data)

print("=== Hospital Readmission Prediction ===")
print("\nPatient Data:")
print(df)

# Separate input features and target
X = df.drop("Readmitted", axis=1)
y = df["Readmitted"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression with L2 regularization
model = LogisticRegression(
    penalty="l2",
    C=1.0,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Predict probabilities
y_probability = model.predict_proba(X_test)[:, 1]

# Convert probabilities into 0/1 predictions
y_prediction = (y_probability >= 0.5).astype(int)

# Evaluate using ROC-AUC
auc = roc_auc_score(y_test, y_probability)

print("\nROC-AUC Score:", round(auc, 4))

# Confusion Matrix
cm = confusion_matrix(y_test, y_prediction)

print("\nConfusion Matrix:")
print(cm)

# Get TN, FP, FN, TP
tn, fp, fn, tp = cm.ravel()

print("\nFalse Positives:", fp)
print("False Negatives:", fn)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_prediction, zero_division=0))

print("Clinical Cost Discussion:")
print("- False Negative: A high-risk patient is predicted as low-risk.")
print("  This may lead to missed follow-up or monitoring.")
print("- False Positive: A low-risk patient is predicted as high-risk.")
print("  This may lead to unnecessary monitoring, tests, or resource use.")
