# ─────────────────────────────────────────────
# METRICS
# 6 cheezein calculate karenge:
# 1. Confusion Matrix
# 2. Accuracy
# 3. Precision
# 4. Recall
# 5. F1 Score
# 6. Log Loss
# ─────────────────────────────────────────────

import numpy as np

# ─────────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────────

def confusion_matrix(y_actual, y_predicted):
    TP = int(np.sum((y_predicted == 1) & (y_actual == 1)))
    TN = int(np.sum((y_predicted == 0) & (y_actual == 0)))
    FP = int(np.sum((y_predicted == 1) & (y_actual == 0)))
    FN = int(np.sum((y_predicted == 0) & (y_actual == 1)))
    return TP, TN, FP, FN

# ─────────────────────────────────────────────
# ACCURACY
# ─────────────────────────────────────────────

def accuracy(y_actual, y_predicted):
    # kitne % sahi predict kiye
    return np.mean(y_actual == y_predicted)

# ─────────────────────────────────────────────
# PRECISION
# ─────────────────────────────────────────────

def precision(TP, FP):
    # jab AT RISK bola — kitni baar sahi tha?
    if TP + FP == 0:
        return 0.0
    return TP / (TP + FP)

# ─────────────────────────────────────────────
# RECALL
# ─────────────────────────────────────────────

def recall(TP, FN):
    # actual AT RISK mein se kitne pakde?
    if TP + FN == 0:
        return 0.0
    return TP / (TP + FN)

# ─────────────────────────────────────────────
# F1 SCORE
# ─────────────────────────────────────────────

def f1_score(prec, rec):
    # precision aur recall ka balance
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)

# ─────────────────────────────────────────────
# LOG LOSS 
# ─────────────────────────────────────────────

def log_loss(y_actual, y_proba):
    # epsilon → log(0) se bachao
    epsilon = 1e-15
    y_proba = np.clip(y_proba, epsilon, 1 - epsilon)
    return -np.mean(
        y_actual * np.log(y_proba) +
        (1 - y_actual) * np.log(1 - y_proba)
    )


print("metrics.py ready!")  