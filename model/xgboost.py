import numpy as np
from model.decision_tree import DecisionTree

class XGBoost:
    def __init__(self, n_estimators=10, lr=0.1):
        self.n_estimators = n_estimators
        self.lr = lr

    def fit(self, X, y):
        self.models = []
        y_pred = np.zeros(len(y))

        for _ in range(self.n_estimators):
            residuals = y - y_pred
            tree = DecisionTree(max_depth=3)
            tree.fit(X, residuals)
            update = tree.predict_proba(X)
            y_pred += self.lr * update
            self.models.append(tree)

    def predict_proba(self, X):
        y_pred = np.zeros(X.shape[0])
        for tree in self.models:
            y_pred += self.lr * tree.predict_proba(X)

        # Apply sigmoid
        return 1 / (1 + np.exp(-y_pred))

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)
