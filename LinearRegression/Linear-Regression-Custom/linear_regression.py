# ─────────────────────────────────────────────
# LINEAR REGRESSION — FROM SCRATCH
# Yahi asli ML model hai
# Koi sklearn nahi — sab khud likhenge
# ─────────────────────────────────────────────

import numpy as np

class LinearRegressionCustom:

    def __init__(self, learning_rate=0.01, epochs=1000):
        # learning_rate = har step kitna bada ho
        # epochs = kitni baar weights update karein
        self.lr     = learning_rate
        self.epochs = epochs
        self.weights = None   # w — baad mein banenge
        self.bias    = None   # b — baad mein banega

    # ─────────────────────────────────────────
    # FIT — MODEL TRAIN KARO
    # ─────────────────────────────────────────
    def fit(self, X, y):
        n_samples  = X.shape[0]   # rows  (students)
        n_features = X.shape[1]   # columns (features)

        # weights aur bias zero se shuru karo
        self.weights = np.zeros(n_features)
        self.bias    = 0.0

        # gradient descent loop
        for epoch in range(self.epochs):

            # STEP 1: predict karo current weights se
            y_predicted = X.dot(self.weights) + self.bias

            # STEP 2: error nikalo
            error = y_predicted - y

            # STEP 3: gradients nikalo
            dw = (1/n_samples) * X.T.dot(error)
            db = (1/n_samples) * np.sum(error)

            # STEP 4: weights update karo
            self.weights = self.weights - self.lr * dw
            self.bias    = self.bias    - self.lr * db

            # har 100 epoch pe loss print karo
            if epoch % 100 == 0:
                loss = np.mean(error ** 2)
                print(f"  Epoch {epoch:4d} → Loss: {loss:.4f}")

    # ─────────────────────────────────────────
    # PREDICT — NAYE DATA PE ANSWER DO
    # ─────────────────────────────────────────
    def predict(self, X):
        return X.dot(self.weights) + self.bias


print("linear_regression.py ready!")