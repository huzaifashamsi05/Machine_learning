# ─────────────────────────────────────────────
# LOGISTIC REGRESSION — FROM SCRATCH
# Koi sklearn nahi — sab khud likhenge
# ─────────────────────────────────────────────

import numpy as np

class LogisticRegressionCustom:

    def __init__(self, learning_rate=0.01, epochs=2000):
        # learning_rate = har step kitna bada ho
        # epochs = kitni baar weights update karein
        self.lr      = learning_rate
        self.epochs  = epochs
        self.weights = None
        self.bias    = None
        self.losses  = []  # loss curve ke liye

    # ─────────────────────────────────────────
    # SIGMOID — z ko probability mein badlo
    # Formula: 1 / (1 + e^(-z))
    # ─────────────────────────────────────────
    def sigmoid(self, z):
        # clip karo — overflow se bachao
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    # ─────────────────────────────────────────
    # FIT — MODEL TRAIN KARO
    # ─────────────────────────────────────────
    def fit(self, X, y):
        n_samples  = X.shape[0]
        n_features = X.shape[1]

        # weights aur bias zero se shuru
        self.weights = np.zeros(n_features)
        self.bias    = 0.0
        self.losses  = []

        # gradient descent loop
        for epoch in range(self.epochs):

            # STEP 1: z nikalo
            z = X.dot(self.weights) + self.bias

            # STEP 2: probability nikalo
            y_predicted = self.sigmoid(z)

            # STEP 3: gradients nikalo
            error = y_predicted - y
            dw = (1/n_samples) * X.T.dot(error)
            db = (1/n_samples) * np.sum(error)

            # STEP 4: weights update karo
            self.weights = self.weights - self.lr * dw
            self.bias    = self.bias    - self.lr * db

            # STEP 5: loss calculate karo
            epsilon = 1e-15
            y_clipped = np.clip(y_predicted, epsilon, 1 - epsilon)
            loss = -np.mean(
                y * np.log(y_clipped) +
                (1 - y) * np.log(1 - y_clipped)
            )
            self.losses.append(loss)

            # har 200 epoch pe print karo
            if epoch % 200 == 0:
                print(f"  Epoch {epoch:4d} → Loss: {loss:.4f}")

    # ─────────────────────────────────────────
    # PREDICT_PROBA — PROBABILITY NIKALO
    # ─────────────────────────────────────────
    def predict_proba(self, X):
        z = X.dot(self.weights) + self.bias
        return self.sigmoid(z)

    # ─────────────────────────────────────────
    # PREDICT — CLASS NIKALO (0 ya 1)
    # ─────────────────────────────────────────
    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        # threshold se upar → AT RISK (1)
        # threshold se neeche → NOT AT RISK (0)
        return (proba >= threshold).astype(int)


print("logistic_regression.py ready!")