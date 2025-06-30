# 💳 Credit Card Fraud Detection using Machine Learning

This project applies supervised learning techniques to detect fraudulent credit card transactions. We focus on handling extreme class imbalance and improving recall to catch as many fraud cases as possible.

---

## 📄 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Rows**: ~284,807 transactions
- **Fraud cases**: Only 492 (highly imbalanced!)

> **CSV File Link**:  
> [Download creditcard.csv](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## 🎯 Problem Statement

Financial fraud is rare but costly. The goal is to build a model that accurately detects fraud while minimizing false negatives (missed frauds).

---

## 📊 Steps Performed

1️⃣ **Exploratory Data Analysis (EDA)**
- Confirmed no missing values
- Verified data types and distributions
- Observed extreme class imbalance (fraud cases ~0.17%)

2️⃣ **Feature Engineering**
- Scaled `Amount` and `Time` columns using StandardScaler
- PCA-transformed columns V1–V28 left as is

3️⃣ **Train-Test Split**
- Stratified split to maintain fraud ratio

4️⃣ **Modeling**
- Logistic Regression with `class_weight='balanced'`
- Random Forest with `class_weight='balanced'`

5️⃣ **Evaluation**
- Confusion matrix
- Accuracy, Precision, Recall, F1-score
- ROC Curve and AUC

---

## ✅ Key Results (Logistic Regression)

| Metric            | Value               |
|-------------------|---------------------|
| Accuracy          | 97.6%               |
| Recall (fraud)    | 91.8% ✅          |
| Precision (fraud) | 6.3% ⚠️          |
| F1-Score (fraud)  | 11.8%             |

- Very high recall: most fraud cases are caught.
- Low precision: many false alarms, expected in fraud detection.

---

## 🔬 Insights

- In fraud detection, **recall is prioritized** to avoid missing frauds.
- Precision can be improved further using:
  - Adjusted probability thresholds
  - Ensemble models (e.g., XGBoost, LightGBM)
  - SMOTE or other sampling techniques

---

## 💡 Future Work

- Apply SMOTE to further handle imbalance
- Tune model hyperparameters
- Deploy as an API for real-time fraud detection
- Explainable AI (SHAP values) to understand fraud predictions

---

## ⚙️ Requirements

```bash
pandas
numpy
scikit-learn
matplotlib
seaborn

