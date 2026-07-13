# ─────────────────────────────────────────────
# PERCEPTRON — FROM SCRATCH
# Binary classification
# High Risk ya Not High Risk
# ─────────────────────────────────────────────

import numpy as np

class PerceptronCustom:

    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr      = learning_rate
        self.epochs  = epochs
        self.weights = None
        self.bias    = None
        self.errors  = []  # har epoch mein kitni galtiyan

    # ─────────────────────────────────────────
    # FIT — TRAIN KARO
    # ─────────────────────────────────────────
    def fit(self, X, y):
        n_samples  = X.shape[0]
        n_features = X.shape[1]

        self.weights = np.zeros(n_features)
        self.bias    = 0.0
        self.errors  = []

        for epoch in range(self.epochs):
            epoch_errors = 0

            for i in range(n_samples):

                # STEP 1: predict karo
                z          = X[i].dot(self.weights) + self.bias
                y_predicted = 1 if z >= 0 else 0

                # STEP 2: galat hai toh update karo
                if y_predicted != y[i]:
                    # y[i]=1 aur predicted=0 → weights badhao
                    # y[i]=0 aur predicted=1 → weights ghatao
                    update = self.lr * (y[i] - y_predicted)
                    self.weights = self.weights + update * X[i]
                    self.bias    = self.bias    + update
                    epoch_errors += 1

            self.errors.append(epoch_errors)

            if epoch % 100 == 0:
                print(f"  Epoch {epoch:4d} → Errors: {epoch_errors}")

            # agar koi error nahi → converge ho gaya!
            if epoch_errors == 0:
                print(f"\n  Converged at epoch {epoch}! ✅")
                break

        if self.errors[-1] > 0:
            print(f"\n  Note: Model did not fully converge.")
            print(f"  Final epoch errors: {self.errors[-1]}")

    # ─────────────────────────────────────────
    # PREDICT — CLASS NIKALO
    # ─────────────────────────────────────────
    def predict(self, X):
        z = X.dot(self.weights) + self.bias
        return (z >= 0).astype(int)

    # ─────────────────────────────────────────
    # RAW SCORE — decision boundary se distance
    # ─────────────────────────────────────────
    def raw_score(self, X):
        return X.dot(self.weights) + self.bias


print("perceptron.py ready!")