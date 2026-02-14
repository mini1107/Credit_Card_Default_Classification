import numpy as np
from model.decision_tree import DecisionTree

class RandomForest:
    def __init__(self, n_trees=5, max_features=None):
        self.n_trees = n_trees
        self.max_features = max_features

    def fit(self, X, y):
        self.trees = []
        n_samples, n_features = X.shape

        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))

        for _ in range(self.n_trees):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            tree = DecisionTree(
                max_depth=3,
                min_samples_split=20,
                max_features=self.max_features
            )
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)

    def predict_proba(self, X):
        tree_probs = np.array([tree.predict_proba(X) for tree in self.trees])
        return np.mean(tree_probs, axis=0)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
