# ─────────────────────────────────────────────
# POISSON REGRESSION — FROM SCRATCH
# Count data predict karna
# Output hamesha positive!
# ─────────────────────────────────────────────

import numpy as np

class PoissonRegressionCustom:

    def __init__(self, learning_rate=0.001, epochs=2000):
        self.lr      = learning_rate
        self.epochs  = epochs
        self.weights = None
        self.bias    = None
        self.losses  = []

    # ─────────────────────────────────────────
    # FIT — TRAIN KARO
    # ─────────────────────────────────────────
    def fit(self, X, y):
        n_samples  = X.shape[0]
        n_features = X.shape[1]

        self.weights = np.zeros(n_features)
        self.bias    = 0.0
        self.losses  = []

        for epoch in range(self.epochs):

            # STEP 1: z nikalo
            z = X.dot(self.weights) + self.bias

            # STEP 2: clip karo overflow se bachao
            z = np.clip(z, -10, 10)

            # STEP 3: mu nikalo — exp() se
            # mu = predicted count
            # hamesha positive! ✅
            mu = np.exp(z)

            # STEP 4: loss nikalo
            # Poisson negative log likelihood
            epsilon = 1e-10
            loss = np.mean(mu - y * np.log(mu + epsilon))
            self.losses.append(loss)

            # STEP 5: gradients nikalo
            error = mu - y
            dw = (1/n_samples) * X.T.dot(error)
            db = np.mean(error)

            # STEP 6: weights update karo
            self.weights = self.weights - self.lr * dw
            self.bias    = self.bias    - self.lr * db

            if epoch % 200 == 0:
                print(f"  Epoch {epoch:4d} → Loss: {loss:.4f}")

    # ─────────────────────────────────────────
    # PREDICT — MU AUR COUNT NIKALO
    # ─────────────────────────────────────────
    def predict(self, X):
        z  = X.dot(self.weights) + self.bias
        z  = np.clip(z, -10, 10)
        mu = np.exp(z)
        # round karo — count integer hota hai
        return mu, np.round(mu).astype(int)


print("poisson_regression.py ready!")