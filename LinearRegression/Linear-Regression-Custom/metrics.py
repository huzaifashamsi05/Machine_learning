# ─────────────────────────────────────────────
# METRICS
# 4 cheezein calculate karenge:
# 1. MAE   — average error
# 2. MSE   — squared error
# 3. RMSE  — MSE ka square root
# 4. R²    — model kitna acha hai (0 se 1)
# ─────────────────────────────────────────────

import numpy as np

def mae(y_actual, y_predicted):
    return np.mean(np.abs(y_actual - y_predicted))

def mse(y_actual, y_predicted):
    return np.mean((y_actual - y_predicted) ** 2)

def rmse(y_actual, y_predicted):
    return np.sqrt(mse(y_actual, y_predicted))

def r2_score(y_actual, y_predicted):
    total_error    = np.sum((y_actual - np.mean(y_actual)) ** 2)
    residual_error = np.sum((y_actual - y_predicted) ** 2)
    return 1 - (residual_error / total_error)

print("metrics.py ready!")