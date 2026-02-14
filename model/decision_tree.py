import numpy as np

class DecisionTree:
    def __init__(self, max_depth=3, min_samples_split=20, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features

    def gini(self, y):
        classes = np.unique(y)
        impurity = 1
        for c in classes:
            p = np.sum(y == c) / len(y)
            impurity -= p ** 2
        return impurity

    def split(self, X, y, depth):

        # Stopping conditions
        if (depth >= self.max_depth or
            len(np.unique(y)) == 1 or
            len(y) < self.min_samples_split):
            return np.mean(y)

        n_features = X.shape[1]

        # Random feature sampling (important for RF)
        if self.max_features:
            features = np.random.choice(n_features, self.max_features, replace=False)
        else:
            features = range(n_features)

        best_feature = None
        best_threshold = None
        best_gain = -1

        for feature in features:

            values = np.unique(X[:, feature])

            # Limit threshold candidates
            if len(values) > 15:
                thresholds = np.linspace(values.min(), values.max(), 15)
            else:
                thresholds = values

            for t in thresholds:
                left_mask = X[:, feature] <= t
                right_mask = X[:, feature] > t

                left = y[left_mask]
                right = y[right_mask]

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
            return np.mean(y)

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

    def predict_proba(self, X):
        return np.array([self.predict_sample(x, self.tree) for x in X])

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
