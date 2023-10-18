import numpy as np
import pandas as pd
import math

class My_KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def evklid_distance(self, x1, x2):
        dist = 0
        for i in range(len(x1)):
            dist += (x1[i] - x2[i]) ** 2
        return np.sqrt(dist)

    def predict(self, X_test):
        y_pred = [self._predict(x) for x in X_test]
        return np.array(y_pred)

    def _predict(self, x):
        #евклидово расстояние между тест и трейн
        dist = [self.evklid_distance(x, x_train)for x_train in self.X_train]
        #сортинг по расстояниям
        k_indices = np.argsort(dist)[:self.k]
        #классы сседей
        neigh_labels = [self.y_train[i] for i in k_indices]
        #самый частый класс
        res = np.bincount(neigh_labels).argmax()
        return res
