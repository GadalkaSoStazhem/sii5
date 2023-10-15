import numpy as np
from preps import LabelEncoder
from collections import Counter
class My_KNN:
    def __init__(self, k = 5):
        self.k = k
        self.le = LabelEncoder()

    def fit(self, X, y):
        self.X_train = X
        self.le.fit(y)
        self.y_train = self.le.transform(y)

    def predict(self, X):
        y_pred = [self.le.inverse_transform([self._predict(x)])[0] for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        dists = [np.linalg.norm(np.array(x) - np.array(x_train)) for x_train in self.X_train]
        k_inds = np.argsort(dists)[:self.k]
        k_labels = [self.y_train[i] for i in k_inds]
        most_freq = Counter(k_labels).most_common(1)
        return most_freq[0][0]