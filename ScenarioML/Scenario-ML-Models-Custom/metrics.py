# ─────────────────────────────────────────────
# METRICS — Teeno models ke liye
# Part 1: MAE, RMSE, Poisson Deviance
# Part 2: Confusion Matrix, Accuracy, P, R, F1
# Part 3: Per-class P, R, F1, Macro F1
# ─────────────────────────────────────────────

import numpy as np

# ─────────────────────────────────────────────
# PART 1 — POISSON METRICS
# ─────────────────────────────────────────────

def mae(y_actual, y_predicted):
    return np.mean(np.abs(y_actual - y_predicted))

def rmse(y_actual, y_predicted):
    return np.sqrt(np.mean((y_actual - y_predicted) ** 2))

def poisson_deviance(y_actual, y_predicted):
    epsilon = 1e-10
    y_predicted = np.clip(y_predicted, epsilon, None)
    return 2 * np.mean(
        y_actual * np.log((y_actual + epsilon) / y_predicted)
        - (y_actual - y_predicted)
    )

# ─────────────────────────────────────────────
# PART 2 — PERCEPTRON METRICS
# ─────────────────────────────────────────────

def confusion_matrix(y_actual, y_predicted):
    TP = int(np.sum((y_predicted == 1) & (y_actual == 1)))
    TN = int(np.sum((y_predicted == 0) & (y_actual == 0)))
    FP = int(np.sum((y_predicted == 1) & (y_actual == 0)))
    FN = int(np.sum((y_predicted == 0) & (y_actual == 1)))
    return TP, TN, FP, FN

def accuracy(y_actual, y_predicted):
    return np.mean(y_actual == y_predicted)

def precision(TP, FP):
    if TP + FP == 0:
        return 0.0
    return TP / (TP + FP)

def recall(TP, FN):
    if TP + FN == 0:
        return 0.0
    return TP / (TP + FN)

def f1_score(prec, rec):
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)

# ─────────────────────────────────────────────
# PART 3 — NAIVE BAYES METRICS
# ─────────────────────────────────────────────

def multiclass_metrics(y_actual, y_predicted, classes):
    results = {}
    for cls in classes:
        TP = int(np.sum((y_predicted == cls) & (y_actual == cls)))
        FP = int(np.sum((y_predicted == cls) & (y_actual != cls)))
        FN = int(np.sum((y_predicted != cls) & (y_actual == cls)))

        prec = TP / (TP + FP) if TP + FP > 0 else 0.0
        rec  = TP / (TP + FN) if TP + FN > 0 else 0.0
        f1   = (2 * prec * rec / (prec + rec)
                if prec + rec > 0 else 0.0)

        results[cls] = {
            "precision": round(prec, 4),
            "recall"   : round(rec,  4),
            "f1"       : round(f1,   4)
        }

    # macro F1
    macro_f1 = np.mean([results[c]["f1"] for c in classes])
    return results, round(macro_f1, 4)


print("metrics.py ready!")