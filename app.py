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
from model.metrics import *

st.set_page_config(page_title="Credit Default Prediction", layout="centered")

st.title("Credit Card Default Classification")

# ==============================
# LOAD DEFAULT DATASET
# ==============================
try:
    data = pd.read_csv("credit_default.csv")
    st.success("Default dataset loaded successfully!")
except:
    st.error("Default dataset not found. Please upload CSV.")
    data = None

# Optional upload (overrides default)
uploaded_file = st.file_uploader("Upload Test CSV (Optional)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("Uploaded dataset loaded successfully!")

# ==============================
# MODEL SELECTION
# ==============================
model_choice = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "kNN",
     "Naive Bayes", "Random Forest", "XGBoost"]
)

# ==============================
# RUN MODEL IF DATA AVAILABLE
# ==============================
if data is not None:

    target_column = "default.payment.next.month"

    if target_column not in data.columns:
        st.error(f"Dataset must contain '{target_column}' column.")
        st.stop()

    X = data.drop(target_column, axis=1).values
    y = data[target_column].values

    # Select model
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

    with st.spinner("Training model..."):
        model.fit(X, y)
        y_pred = model.predict(X)

    y_prob = model.predict_proba(X)
    auc = roc_auc_score(y, y_prob)


    # ==============================
    # EVALUATION METRICS TABLE
    # ==============================
    st.subheader("📊 Model Evaluation Metrics")

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)

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

    tp, tn, fp, fn = confusion_matrix(y, y_pred)

    cm = np.array([[tn, fp],
                   [fn, tp]])

    fig, ax = plt.subplots(figsize=(5, 4))
    cax = ax.matshow(cm)

    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f"{val}", ha='center', va='center')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["Actual 0", "Actual 1"])
    plt.colorbar(cax)

    st.pyplot(fig)

    # ==============================
    # ROC CURVE
    # ==============================
    st.subheader("📈 ROC Curve")

    thresholds = np.linspace(0, 1, 100)
    tpr_list = []
    fpr_list = []

    for thresh in thresholds:
        y_temp = (y_prob >= thresh).astype(int)
        tp, tn, fp, fn = confusion_matrix(y, y_temp)

        tpr = tp / (tp + fn + 1e-10)
        fpr = fp / (fp + tn + 1e-10)

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    fig2, ax2 = plt.subplots()
    ax2.plot(fpr_list, tpr_list)
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title(f"ROC Curve (AUC = {auc:.4f})")

    st.pyplot(fig2)
