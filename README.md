# Credit Card Fraud Detection

## Case Study

**Credit Card Fraud Detection — XGBoost + SMOTE + Decision Threshold Tuning**

This project demonstrates how to detect fraudulent credit-card transactions when the fraud class is highly imbalanced.

### Techniques Used

1. **XGBoost** for classification
2. **SMOTE** (Synthetic Minority Over-sampling Technique) for the training data
3. **Decision-threshold tuning** to study the precision/recall trade-off
4. **Feature importance** scores from XGBoost
5. **Baseline SVM** for comparison
6. **ROC-AUC and PR-AUC** for evaluation

## Dataset

For a self-contained GitHub demonstration, this project generates **3,000 synthetic transactions**, including only **60 fraud cases (2%)**.

The synthetic data contains:

- `amount`
- `time_since_last_txn`
- `transactions_24h`
- `avg_amount_7d`
- `device_risk`
- `location_distance_km`
- `merchant_risk`
- `account_age_days`
- `is_international`
- `is_new_device`

Target:

- `is_fraud = 0` → legitimate transaction
- `is_fraud = 1` → fraudulent transaction

This is an educational demonstration. It is **not the Kaggle IEEE-CIS dataset** and its model scores should not be interpreted as real-world fraud-detection performance.

## Why SMOTE?

Fraud detection is usually highly imbalanced: legitimate transactions greatly outnumber fraudulent ones.

SMOTE creates synthetic minority-class training examples so the XGBoost model gets more fraud examples during training.

**Important:** SMOTE is applied only to the training set. The test set keeps the original class distribution.

## Why Tune the Decision Threshold?

The default classification threshold is usually 0.50.

For fraud detection:

- A **false negative** means fraud is missed and classified as legitimate.
- A **false positive** means a legitimate transaction is flagged as fraud.

A lower threshold can catch more fraud, increasing recall, but may also create more false positives. The threshold should therefore reflect the relative business cost of missed fraud and customer friction.

This project tests thresholds from **0.20 to 0.70** and selects the threshold with the highest F1 score on the demonstration test set.

## XGBoost Feature Importance

The project prints and plots XGBoost feature-importance scores.

The importance values indicate how much each feature contributed to the trained tree model. They are useful for interpretation, but they should not be treated as causal explanations.

## Baseline SVM

An RBF-kernel SVM is included as a baseline.

The SVM uses:

- StandardScaler
- `class_weight="balanced"`
- the original imbalanced training data
- a default decision threshold of 0.50

The XGBoost model and SVM are compared using ROC-AUC, PR-AUC, precision, recall, and F1 score.

## How to Run

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd credit-card-fraud-detection
```

### 2. Install the required libraries

```bash
pip install -r requirements.txt
```

### 3. Run the Python program

```bash
python credit_card_fraud_detection.py
```

The program displays:

- class distribution
- SMOTE class distribution
- XGBoost ROC-AUC and PR-AUC
- threshold-tuning table
- selected decision threshold
- confusion matrix
- classification report
- SVM results
- model comparison
- XGBoost feature-importance scores
- feature-importance chart

## Notebook

`Credit_Card_Fraud_Detection.ipynb` contains the same analysis in notebook format with the output already executed.

It can be opened in:

- Jupyter Notebook
- JupyterLab
- Google Colab

## Important Note

The dataset in this repository is synthetic and designed only for a college/educational demonstration.

For a real fraud-detection project, use an appropriate de-identified transaction dataset such as the **IEEE-CIS Fraud Detection** dataset, perform proper train/validation/test evaluation, and consider the financial and operational costs of false negatives and false positives.
