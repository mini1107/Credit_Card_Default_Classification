# 📘 Machine Learning Assignment 2  
## Credit Card Default Prediction  

---

## 1️⃣ Problem Statement

Credit card default prediction is an important financial risk modeling problem.  
Banks and financial institutions aim to predict whether a customer will default on their next credit card payment.

Early prediction helps in:

- Reducing financial risk  
- Improving credit approval strategies  
- Identifying high-risk customers  
- Enhancing decision-making processes  

This project formulates the problem as a **binary classification task**:

- `0` → No Default  
- `1` → Default  

---

## 2️⃣ Dataset Description

### 📊 Dataset: Credit Card Default Dataset

**Source:** UCI Machine Learning Repository / Kaggle  

**Total Instances:** 30,000  

**Number of Features:** 23  

### Important Features:

- `LIMIT_BAL` – Credit limit  
- `SEX`  
- `EDUCATION`  
- `MARRIAGE`  
- `AGE`  
- `PAY_0` to `PAY_6` – Repayment status  
- `BILL_AMT1` to `BILL_AMT6` – Bill amounts  
- `PAY_AMT1` to `PAY_AMT6` – Payment amounts  

### Target Variable:

`default.payment.next.month`

- 0 → No Default  
- 1 → Default  

---

## 3️⃣ Models Implemented

The following classification models were implemented from scratch:

1. Logistic Regression  
2. Decision Tree  
3. k-Nearest Neighbors (kNN)  
4. Gaussian Naive Bayes  
5. Random Forest (Ensemble)  
6. XGBoost (Gradient Boosting Ensemble)  

All models were implemented manually without using scikit-learn classifiers.

---

## 4️⃣ Evaluation Metrics

The following evaluation metrics were implemented from scratch:

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- AUC Score  
- Matthews Correlation Coefficient (MCC)  

### Example: Accuracy Implementation

```python
def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

| ML Model            | Accuracy | AUC  | Precision | Recall | F1   | MCC  |
| ------------------- | -------- | ---- | --------- | ------ | ---- | ---- |
| Logistic Regression | 0.81     | 0.78 | 0.66      | 0.54   | 0.59 | 0.45 |
| Decision Tree       | 0.79     | 0.75 | 0.60      | 0.58   | 0.59 | 0.43 |
| kNN                 | 0.77     | 0.72 | 0.56      | 0.50   | 0.53 | 0.36 |
| Naive Bayes         | 0.76     | 0.73 | 0.52      | 0.62   | 0.56 | 0.38 |
| Random Forest       | 0.84     | 0.83 | 0.72      | 0.63   | 0.67 | 0.55 |
| XGBoost             | 0.86     | 0.86 | 0.75      | 0.68   | 0.71 | 0.60 |

## 6️⃣ Observations

### 🔹 Logistic Regression (Baseline Linear Model)

- Provides stable baseline performance  
- Assumes linear relationship between features and target  
- Limited in capturing nonlinear financial behavior  

### 🔹 Decision Tree (Moderate Performance)

- Captures nonlinear relationships  
- Easy to interpret  
- Prone to overfitting  
- Moderate generalization performance  

### 🔹 kNN (Sensitive to Scaling)

- Distance-based model  
- Performance depends on feature scaling  
- Less effective in high-dimensional datasets  

### 🔹 Naive Bayes (Independence Assumption)

- Fast and computationally efficient  
- Assumes features are independent  
- Financial variables are correlated, limiting performance  

### 🔹 Random Forest (Better Generalization)

- Ensemble of multiple decision trees  
- Reduces variance  
- Improves stability and accuracy  
- Better generalization than single tree  

### 🔹 XGBoost (Best Overall Performance)

- Gradient boosting technique  
- Sequentially reduces prediction errors  
- Achieved highest AUC and MCC  
- Best suited for complex financial risk modeling

Streamlit Application Features

The developed Streamlit web application provides an interactive interface to evaluate and compare classification models.

The application includes:

 - Dataset upload functionality (CSV format)
 - Model selection dropdown for all six implemented models
 - Display of evaluation metrics including Accuracy, Precision, Recall, F1 Score, AUC, and MCC
 - Confusion matrix visualization
 - ROC curve visualization
 - Clean and structured dashboard layout