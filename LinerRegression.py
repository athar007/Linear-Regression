import numpy as np


class LinerRegression:
    
    def __init__(self, lr = 0.001, n_its = 1000):
        self.lr = lr
        self.weight = None
        self.bias = None
        self.n_itrs = n_its

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_itrs):
            y_pred = np.dot(X, self.weight) + self.bias
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            self.weight = self.weight - (self.lr * dw)
            self.bias = self.bias - (self.lr * db)


    def predict(self, X):
        y_pred = np.dot(X, self.weight) + self.bias
        return y_pred
    
