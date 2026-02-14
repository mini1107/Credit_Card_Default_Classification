import numpy as np
from model.decision_tree import DecisionTree

class RandomForest:
    def __init__(self, n_trees=10):
        self.n_trees = n_trees

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            indices = np.random.choice(len(X), len(X), replace=True)
            tree = DecisionTree(max_depth=5)
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)

    def predict_proba(self, X):
        tree_probs = np.array([tree.predict_proba(X) for tree in self.trees])
        return np.mean(tree_probs, axis=0)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)
