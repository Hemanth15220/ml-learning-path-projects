# 🧠 HR Attrition Prediction using Machine Learning

This project aims to predict whether an employee is likely to leave the company, using supervised classification algorithms on HR data. We explore techniques to handle class imbalance and improve recall for the minority (attrition) class.

---

## 📊 Dataset

- **Source**: [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- **Size**: ~1500 rows, 35 features
- **Target Variable**: `Attrition` (`Yes` or `No`)

---

## 🔍 Problem Statement

Attrition prediction helps HR departments proactively address employee churn. Given employee information (age, department, job role, performance, etc.), can we predict whether they are at risk of leaving?

---

## ✅ Steps Performed

1. **EDA**
   - Class imbalance: ~16% employees left
   - No missing values
2. **Preprocessing**
   - One-Hot Encoding of categorical features
   - Standard scaling of numeric features
3. **Train-Test Split**
   - Stratified to preserve class balance
4. **Models Built**
   - Logistic Regression (with and without `class_weight='balanced'`)
   - Random Forest Classifier (`class_weight='balanced'`)
5. **Evaluation**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix

---

## 🧠 Key Learnings

| Model                 | Accuracy | Recall (Attrition) | Precision (Attrition) | F1 Score |
|----------------------|----------|---------------------|------------------------|----------|
| Logistic Regression  | 86.4%    | 0.34                | 0.64                   | 0.44     |
| Logistic (Balanced)  | 74.8%    | **0.66**            | 0.35                   | 0.46     |
| Random Forest         | 84.0%    | 0.06                | 0.50                   | 0.11     |

- Class weighting improves recall, especially for Logistic Regression
- Random Forest favored the majority class even with balanced weights
- Class imbalance needs special handling for effective minority detection

---

## 📌 Requirements

```bash
pandas
numpy
scikit-learn
matplotlib
seaborn

