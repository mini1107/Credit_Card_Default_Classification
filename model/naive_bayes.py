import numpy as np

class GaussianNB:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.priors = {}

        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = X_c.mean(axis=0)
            self.var[c] = X_c.var(axis=0)
            self.priors[c] = X_c.shape[0] / X.shape[0]

    def gaussian(self, x, mean, var):
        return np.exp(-((x - mean) ** 2) / (2 * var + 1e-10)) / np.sqrt(2 * np.pi * var + 1e-10)

    def predict_proba(self, X):
        probs = []
        for x in X:
            posteriors = []
            for c in self.classes:
                prior = np.log(self.priors[c])
                likelihood = np.sum(np.log(self.gaussian(x, self.mean[c], self.var[c])))
                posteriors.append(prior + likelihood)

            exp_vals = np.exp(posteriors)
            probs.append(exp_vals[1] / np.sum(exp_vals))

        return np.array(probs)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
