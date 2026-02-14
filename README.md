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
| Logistic Regression    | 0.81     | 0.54     | 0.58   | 0.56     | 0.44 | 0.89 |
| Decision Tree          | 0.88     | 0.73     | 0.68   | 0.70     | 0.63 | 0.91 |
| kNN                    | 0.84     | 0.65     | 0.46   | 0.54     | 0.45 | 0.83 |
| Naive Bayes            | 0.77     | 0.47     | 0.80   | 0.59     | 0.48 | 0.87 |
| Random Forest          | 0.85     | 0.72     | 0.43   | 0.54     | 0.48 | 0.90 |
| XGBoost                | 0.82     | 1.00     | 0.14   | 0.25     | 0.34 | 0.91 |

---

# **Observations on Model Performance:**

| ML Model Name           | Observation about model performance                                                                                                                                                                                                                                                                                                                                                 |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Logistic Regression** | Achieved good baseline performance (Accuracy ≈ 81%, AUC ≈ 0.89) with moderate MCC (0.44). This indicates that the dataset has a degree of linear separability. However, compared to tree-based models, Logistic Regression shows lower balanced performance, suggesting that purely linear decision boundaries are insufficient to fully capture nonlinear financial risk patterns. |
| **Decision Tree**       | Best overall performer (Accuracy ≈ 88%, MCC ≈ 0.63, AUC ≈ 0.91) with balanced precision and recall. The model effectively captures nonlinear feature interactions and demonstrates strong class discrimination. Its high F1-score and MCC indicate reliable and well-balanced predictions across both classes.                                                                      |
| **kNN**                 | Demonstrates moderate performance (Accuracy ≈ 84%, AUC ≈ 0.83). While precision is reasonable, recall is comparatively lower, indicating sensitivity to distance metrics and feature scaling. The model performs adequately but does not outperform tree-based approaches in this dataset.                                                                                          |
| **Naive Bayes**         | Shows good recall (≈ 0.80) but relatively lower precision (≈ 0.47), indicating a tendency to predict the positive class more frequently. The independence assumption between features limits overall performance, though AUC remains strong (≈ 0.87), suggesting good probability ranking capability.                                                                               |
| **Random Forest**       | Achieves strong performance (Accuracy ≈ 85%, AUC ≈ 0.90) with improved precision compared to Logistic Regression and kNN. Ensemble learning reduces variance and enhances probability ranking, though recall is slightly lower compared to Decision Tree. Overall, it provides stable and reliable classification.                                                                  |
| **XGBoost**             | Exhibits excellent ranking performance (AUC ≈ 0.91) and very high precision (1.00), indicating that when it predicts default, it is highly confident. However, low recall (≈ 0.14) suggests conservative classification behavior at the chosen threshold. This indicates strong probability modeling but requires threshold tuning for balanced classification performance.         |

---

# **Final Conclusion:**

Among all the evaluated models, Decision Tree achieved the best overall balanced performance, with the highest accuracy, F1-score, MCC, and strong AUC. This indicates that the dataset contains nonlinear relationships that are effectively captured by tree-based learning.

Random Forest and XGBoost demonstrated excellent probability ranking capability (AUC > 0.90), confirming the strength of ensemble learning methods. However, XGBoost exhibited conservative classification behavior at the default threshold, leading to lower recall despite strong ranking performance.

Logistic Regression provided a solid baseline, confirming partial linear separability within the dataset, while Naive Bayes achieved high recall but lower precision due to its independence assumptions. kNN showed moderate performance but did not outperform tree-based methods.

Overall, Decision Tree is the most suitable model for this dataset based on balanced evaluation metrics, while ensemble models remain strong alternatives for applications requiring high ranking performance and probability estimation accuracy.
