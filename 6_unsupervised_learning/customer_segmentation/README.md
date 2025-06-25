# 🛍️ Customer Segmentation using K-Means Clustering

This project applies unsupervised learning (K-Means) to segment mall customers into distinct behavioral groups based on age, income, and spending score. It helps businesses understand and target customer types effectively.

---

## 📊 Dataset Info

- **Source**: [Mall Customer Segmentation Dataset - Kaggle](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial)
- **Rows**: 200 customers
- **Columns**: CustomerID, Gender, Age, Annual Income (k$), Spending Score (1–100)

---

## 🎯 Objective

Group mall customers into clusters based on:
- Age
- Annual Income
- Spending Score

These clusters represent distinct customer segments for targeted marketing.

---

## 🔧 Techniques Used

- Feature selection: `Age`, `Annual Income`, `Spending Score`
- Feature scaling: `StandardScaler`
- KMeans Clustering
- Elbow Method for optimal k
- Cluster visualization with `seaborn`

---

## 🔍 Cluster Insights

Based on cluster centroids, we assigned human-readable labels:

| Cluster | Label                  | Profile Description                                          |
|---------|------------------------|---------------------------------------------------------------|
| 0       | 🎯 Target Customers       | High income, high spenders — top revenue segment             |
| 1       | ⚖️ Cautious Spenders     | Low income, high spending — price-sensitive but active       |
| 2       | 🧊 Conservative Earners  | High income, low spenders — room for upselling               |
| 3       | 💸 Budget Shoppers       | Low income, low spenders — value-focused customers           |

> These segments help in tailored offers, pricing strategies, and product positioning.

---

## 🧠 Key Learnings

- KMeans requires feature scaling to perform optimally.
- The Elbow Method is useful for selecting the best `k` value.
- Customer segmentation helps visualize business value beyond raw data.

---

## ✅ Requirements

```bash
pandas
numpy
scikit-learn
matplotlib
seaborn
