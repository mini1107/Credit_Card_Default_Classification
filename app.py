import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from model.logistic import LogisticRegression
from model.knn import KNN
from model.naive_bayes import GaussianNB
from model.decision_tree import DecisionTree
from model.random_forest import RandomForest
from model.xgboost import XGBoost

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)

st.set_page_config(page_title="Credit Default Prediction", layout="centered")
st.title("Credit Card Default Classification")

# ==============================
# LOAD DATASET
# ==============================
try:
    data = pd.read_csv("credit_default.csv")
    st.success("Dataset loaded successfully!")
except:
    st.error("Dataset not found.")
    st.stop()

target_column = "default.payment.next.month"

if target_column not in data.columns:
    st.error(f"Dataset must contain '{target_column}' column.")
    st.stop()

X = data.drop(target_column, axis=1).values
y = data[target_column].values

# ==============================
# TRAIN TEST SPLIT (80-20)
# ==============================
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# ==============================
# STANDARDIZATION
# ==============================
mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-10

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# ==============================
# MODEL SELECTION
# ==============================
model_choice = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "kNN",
     "Naive Bayes", "Random Forest", "XGBoost"]
)

if model_choice == "Logistic Regression":
    model = LogisticRegression()
elif model_choice == "Decision Tree":
    model = DecisionTree()
elif model_choice == "kNN":
    model = KNN()
elif model_choice == "Naive Bayes":
    model = GaussianNB()
elif model_choice == "Random Forest":
    model = RandomForest()
else:
    model = XGBoost()

# ==============================
# TRAIN MODEL
# ==============================
with st.spinner("Training model..."):
    model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# ==============================
# METRICS
# ==============================
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

st.subheader("📊 Model Evaluation Metrics")

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "MCC", "AUC"],
    "Score": [acc, prec, rec, f1, mcc, auc]
})

st.dataframe(
    metrics_df.style
    .format({"Score": "{:.4f}"})
    .background_gradient(cmap="Blues"),
    use_container_width=True
)

# ==============================
# CONFUSION MATRIX
# ==============================
st.subheader("📌 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax)
st.pyplot(fig)

# ==============================
# ROC CURVE
# ==============================
st.subheader("📈 ROC Curve")

fpr, tpr, _ = roc_curve(y_test, y_prob)

fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
ax2.plot([0, 1], [0, 1], linestyle="--")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC Curve")
ax2.legend()

st.pyplot(fig2)
