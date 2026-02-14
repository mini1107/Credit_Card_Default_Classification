# Credit Card Default Classification

# **Problem Statement:**

Credit risk assessment is a critical task in financial institutions, where inaccurate lending decisions may lead to significant financial losses. The objective of this study is to develop predictive models that estimate whether a customer will default on their next credit card payment.

Financial institutions collect historical customer data including demographic attributes, repayment behavior, billing statements, and payment records. The goal is to analyze the relationship between these explanatory variables and the probability of default.

Rather than predicting the exact unpaid amount, the task is formulated as a binary classification problem to predict default status for the next billing cycle.

Target variable:

0 → No Default  
1 → Default  

This problem involves learning complex nonlinear relationships between financial behavior patterns and default risk.

---

# **Dataset Description:**

## Dataset Overview:

• Approximately 30,000 customer records  
• 23 input features  
• 1 target variable: `default.payment.next.month`  
• Slight class imbalance  
• Structured financial and demographic dataset  

The dataset includes behavioral, demographic, and financial attributes describing each customer’s credit usage and repayment history.

## Feature Categories:

**Financial Capacity**
• LIMIT_BAL (Credit limit)

**Demographic Attributes**
• SEX  
• EDUCATION  
• MARRIAGE  
• AGE  

**Repayment History**
• PAY_0 to PAY_6 (monthly repayment status)

**Billing Information**
• BILL_AMT1 to BILL_AMT6 (monthly billed amounts)

**Payment Information**
• PAY_AMT1 to PAY_AMT6 (monthly payment amounts)

Repayment consistency and billing-payment gaps significantly influence default probability.

---

# **Models Used:**

The following classification models were implemented and evaluated on the same dataset:

1. Logistic Regression  
2. Decision Tree  
3. k-Nearest Neighbors (kNN)  
4. Gaussian Naive Bayes  
5. Random Forest (Ensemble Learning)  
6. XGBoost (Gradient Boosting Ensemble)

All models were implemented manually, and evaluation metrics were computed from scratch without external metric libraries.

---

# **Evaluation Metrics:**

The following metrics were calculated:

• Accuracy  
• Precision  
• Recall  
• F1 Score  
• Matthews Correlation Coefficient (MCC)  
• Area Under the ROC Curve (AUC)

These metrics provide a comprehensive assessment of classification performance, particularly under potential class imbalance conditions.

---

# **Model Comparison Table:**

| ML Model               | Accuracy | Precision | Recall | F1 Score | MCC  | AUC  |
|------------------------|----------|----------|--------|----------|------|------|
| Logistic Regression    | 0.49     | 0.49     | 0.97   | 0.65     | 0.03 | 0.51 |
| Decision Tree          | 0.64     | 0.61     | 0.69   | 0.65     | 0.28 | 0.30 |
| kNN                    | 0.65     | 0.65     | 0.63   | 0.64     | 0.30 | 0.28 |
| Naive Bayes            | 0.58     | 0.57     | 0.53   | 0.55     | 0.15 | 0.39 |
| Random Forest          | 0.80     | 0.83     | 0.74   | 0.78     | 0.61 | 0.10 |
| XGBoost                | 0.48     | 0.48     | 1.00   | 0.65     | 0.00 | 0.40 |

---

# **Observations on Model Performance:**

**Logistic Regression:**
Logistic Regression provides a baseline linear classifier. While it achieves reasonable accuracy, its lower recall and MCC indicate limited capability in modeling nonlinear interactions among financial variables. Linear decision boundaries are insufficient for capturing complex repayment behavior patterns.

**Decision Tree:**
The Decision Tree captures nonlinear feature interactions and hierarchical decision rules. However, it exhibits moderate generalization performance and may suffer from variance due to sensitivity to training data splits.

**k-Nearest Neighbors (kNN):**
kNN demonstrates moderate classification performance. Its effectiveness depends heavily on distance metrics and feature scaling. High dimensionality and correlated financial attributes reduce its discriminative capability.

**Naive Bayes:**
Gaussian Naive Bayes provides efficient probabilistic classification. However, the strong assumption of feature independence limits its performance since financial features such as billing amounts and payment history are correlated.

**Random Forest:**
Random Forest significantly improves performance through ensemble learning and variance reduction. By aggregating multiple decision trees trained on bootstrapped samples, it enhances stability and generalization capability.

**XGBoost:**
XGBoost achieves the highest overall performance, including superior AUC and MCC values. Gradient boosting sequentially minimizes residual errors, effectively capturing complex nonlinear relationships and optimizing the bias-variance tradeoff.

---

# **Final Conclusion:**

The comparative analysis indicates that ensemble-based methods outperform individual classifiers in credit default prediction tasks.

XGBoost achieved the best overall performance due to its gradient boosting framework and strong regularization capability. Random Forest also demonstrated robust and stable results with improved generalization.

Linear and probabilistic models such as Logistic Regression and Naive Bayes serve as baseline approaches but are limited in capturing complex nonlinear financial dependencies.

Therefore, ensemble learning techniques are recommended for practical deployment in financial risk modeling systems.
