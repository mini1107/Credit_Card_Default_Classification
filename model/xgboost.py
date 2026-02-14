import numpy as np
from model.decision_tree import DecisionTree

class XGBoost:
    def __init__(self, n_estimators=10, lr=0.3, max_depth=3):
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):

        self.models = []

        base_prob = np.mean(y)
        self.base_pred = np.log(base_prob / (1 - base_prob + 1e-10))

        y_pred = np.full(len(y), self.base_pred)

        for _ in range(self.n_estimators):

            prob = self.sigmoid(y_pred)
            residuals = y - prob

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=15,
                max_features=int(np.sqrt(X.shape[1]))
            )

            tree.fit(X, residuals)

            update = tree.predict_proba(X)
            y_pred += self.lr * update

            self.models.append(tree)

    def predict_proba(self, X):

        y_pred = np.full(X.shape[0], self.base_pred)

        for tree in self.models:
            y_pred += self.lr * tree.predict_proba(X)

        return self.sigmoid(y_pred)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
