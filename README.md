# 🩺 Diabetes Prediction with Machine Learning

This project focuses on predicting diabetes risk using the **Pima Indians Diabetes Dataset**, with an emphasis on **robust evaluation, domain-aware decision making, and deployability**.

The goal is not only to build an accurate model, but also to ensure **clinical relevance**, **model stability**, and **transparent decision logic**.

---

## 📊 Dataset
- Source: UCI / Kaggle – Pima Indians Diabetes Database
- Binary target: `Outcome` (0 = No Diabetes, 1 = Diabetes)
- Known issue: zero values in medical features (handled explicitly)

---

## 🧹 Data Preprocessing
- Exploratory Data Analysis (EDA)
- Zero-value handling using **KNN Imputation**
- Feature scaling with **StandardScaler**
- Pipeline-based preprocessing to avoid data leakage

---

## 🤖 Modeling Approach

### Baseline Model
- Logistic Regression with class balancing

### Final Model
- **Random Forest Classifier**
- Hyperparameter tuning using **RandomizedSearchCV**
- Evaluation via **5-Fold Stratified Cross-Validation**

**Cross-validation performance (mean ± std):**
- ROC-AUC: **0.836 ± 0.019**
- Recall: **0.754 ± 0.041**
- F1-score: **0.700 ± 0.031**

---

## 🎯 Threshold Optimization

Instead of using the default probability threshold (0.5), a **threshold sweep analysis** was performed.

- Thresholds evaluated from **0.10 to 0.90**
- Precision, Recall, and F1-score analyzed for each threshold
- Decision driven by **healthcare domain requirements**

📌 **Final threshold selected: 0.40**
- Prioritizes recall to minimize false negatives
- Accepts higher false positives, which is preferable in medical screening

---

## 🔍 Model Explainability
- Feature importance endpoint exposed via API
- Most influential features:
  - Glucose
  - BMI
  - Age
  - Diabetes Pedigree Function

---

## 🚀 API Deployment (FastAPI)

Endpoints:
- `POST /predict`  
  Returns prediction, probability, risk level, model metadata, and decision threshold.

- `GET /feature-importance`  
  Returns ranked feature importance from the trained Random Forest model.

The API is designed for easy integration and experimentation.

---

## 🧠 Key Takeaways
- Model decisions are **data-driven and domain-aware**
- Threshold selection is justified analytically
- Cross-validation ensures robustness
- The pipeline is production-ready and extensible

