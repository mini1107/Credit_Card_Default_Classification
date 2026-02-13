import numpy as np

class DecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth

    def gini(self, y):
        classes = np.unique(y)
        impurity = 1
        for c in classes:
            p = np.sum(y == c) / len(y)
            impurity -= p ** 2
        return impurity

    def split(self, X, y, depth):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return np.bincount(y).argmax()

        best_feature = None
        best_threshold = None
        best_gain = -1

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                left = y[X[:, feature] <= t]
                right = y[X[:, feature] > t]

                if len(left) == 0 or len(right) == 0:
                    continue

                gain = self.gini(y) - (
                    len(left)/len(y)*self.gini(left)
                    + len(right)/len(y)*self.gini(right)
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = t

        if best_gain == -1:
            return np.bincount(y).argmax()

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = X[:, best_feature] > best_threshold

        return {
            "feature": best_feature,
            "threshold": best_threshold,
            "left": self.split(X[left_mask], y[left_mask], depth+1),
            "right": self.split(X[right_mask], y[right_mask], depth+1)
        }

    def fit(self, X, y):
        self.tree = self.split(X, y, 0)

    def predict_sample(self, x, node):
        if not isinstance(node, dict):
            return node
        if x[node["feature"]] <= node["threshold"]:
            return self.predict_sample(x, node["left"])
        return self.predict_sample(x, node["right"])

    def predict(self, X):
        return np.array([self.predict_sample(x, self.tree) for x in X])
