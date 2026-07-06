# ─────────────────────────────────────────────
# MAIN — SAB KO JODO
# ─────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

from preprocessing        import encode_categorical, scale_features, train_val_test_split
from logistic_regression  import LogisticRegressionCustom
from metrics              import confusion_matrix, accuracy, precision, recall, f1_score, log_loss

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

script_folder = os.path.dirname(os.path.abspath(__file__))
csv_path      = os.path.join(script_folder, "data", "student_risk_sample.csv")
output_folder = os.path.join(script_folder, "outputs")
os.makedirs(output_folder, exist_ok=True)

# ─────────────────────────────────────────────
# STEP 1 — CSV LOAD KARO
# ─────────────────────────────────────────────

df = pd.read_csv(csv_path)

print("=" * 54)
print("  LOGISTIC REGRESSION — Student At Risk Predictor")
print("=" * 54)
print(f"\n  Dataset loaded successfully!")
print(f"  Rows    : {df.shape[0]}")
print(f"  Columns : {df.shape[1]}")

# ─────────────────────────────────────────────
# STEP 2 — PREPROCESSING
# ─────────────────────────────────────────────

# Yes/No → 1/0
categorical_cols = ["internet_access", "part_time_job"]
df = encode_categorical(df, categorical_cols)

# student_id alag rakh lo — feature nahi hai
student_ids = df["student_id"].values

# features aur target alag karo
feature_cols = [
    "attendance_percentage",
    "study_hours_per_week",
    "previous_exam_score",
    "assignments_completed",
    "late_submissions",
    "class_participation_score",
    "sleep_hours_per_day",
    "internet_access",
    "part_time_job"
]

X = df[feature_cols].values
y = df["at_risk"].values

print(f"\n  Input features:")
for f in feature_cols:
    print(f"    → {f}")
print(f"\n  Target column : at_risk")
print(f"  At Risk (1)   : {int(y.sum())}")
print(f"  Not At Risk(0): {int((y==0).sum())}")

# ─────────────────────────────────────────────
# STEP 3 — SPLIT
# ─────────────────────────────────────────────

X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

# student ids bhi split karo
n         = len(y)
indices   = np.arange(n)
np.random.seed(42)
np.random.shuffle(indices)
test_count  = int(n * 0.15)
val_count   = int(n * 0.15)
train_count = n - test_count - val_count
test_ids  = student_ids[indices[train_count + val_count:]]

print(f"\n  Train size      : {len(y_train)}")
print(f"  Validation size : {len(y_val)}")
print(f"  Test size       : {len(y_test)}")

# ─────────────────────────────────────────────
# STEP 4 — SCALE
# ─────────────────────────────────────────────

X_train_s, X_val_s, X_test_s, mean, std = scale_features(X_train, X_val, X_test)

# ─────────────────────────────────────────────
# STEP 5 — TRAIN
# ─────────────────────────────────────────────

print("\n" + "=" * 54)
print("  TRAINING")
print("=" * 54)
print("\n  Training shuru...\n")

model = LogisticRegressionCustom(learning_rate=0.01, epochs=2000)
model.fit(X_train_s, y_train)

print(f"\n  Learned Weights:")
for name, w in zip(feature_cols, model.weights):
    print(f"    {name:<35}: {w:+.4f}")
print(f"\n  Bias : {model.bias:+.4f}")

# ─────────────────────────────────────────────
# STEP 6 — THRESHOLD COMPARISON
# ─────────────────────────────────────────────

print("\n" + "=" * 54)
print("  THRESHOLD COMPARISON (validation data)")
print("=" * 54)
print(f"\n  {'Threshold':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print(f"  {'-'*54}")

thresholds   = [0.3, 0.5, 0.7]
best_f1      = -1
best_threshold = 0.5
threshold_rows = []

for t in thresholds:
    y_val_pred = model.predict(X_val_s, threshold=t)
    y_val_proba = model.predict_proba(X_val_s)

    TP, TN, FP, FN = confusion_matrix(y_val, y_val_pred)
    acc  = accuracy(y_val, y_val_pred)
    prec = precision(TP, FP)
    rec  = recall(TP, FN)
    f1   = f1_score(prec, rec)

    print(f"  {t:>10.1f} {acc:>10.2f} {prec:>10.2f} {rec:>10.2f} {f1:>10.2f}")

    threshold_rows.append({
        "threshold" : t,
        "accuracy"  : round(acc,  4),
        "precision" : round(prec, 4),
        "recall"    : round(rec,  4),
        "f1_score"  : round(f1,   4),
    })

    if f1 > best_f1:
        best_f1        = f1
        best_threshold = t

print(f"\n  Best threshold selected: {best_threshold}")

# ─────────────────────────────────────────────
# STEP 7 — EVALUATE ON TEST
# ─────────────────────────────────────────────

y_pred  = model.predict(X_test_s, threshold=best_threshold)
y_proba = model.predict_proba(X_test_s)

TP, TN, FP, FN = confusion_matrix(y_test, y_pred)
acc  = accuracy(y_test, y_pred)
prec = precision(TP, FP)
rec  = recall(TP, FN)
f1   = f1_score(prec, rec)
ll   = log_loss(y_test, y_proba)

print("\n" + "=" * 54)
print("  FINAL TEST RESULTS")
print("=" * 54)
print(f"\n  Confusion Matrix:")
print(f"                    Predicted NO   Predicted YES")
print(f"  Actual NO    →    {TN:^14} {FP:^13}")
print(f"  Actual YES   →    {FN:^14} {TP:^13}")
print(f"\n  TP : {TP}  TN : {TN}  FP : {FP}  FN : {FN}")
print(f"\n  Accuracy  : {acc:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print(f"  Log Loss  : {ll:.4f}")
print(f"  Threshold : {best_threshold}")

# ─────────────────────────────────────────────
# STEP 8 — OUTPUT FILES SAVE KARO
# ─────────────────────────────────────────────

# predictions.csv
pred_df = pd.DataFrame({
    "student_id"           : test_ids,
    "actual_label"         : y_test,
    "predicted_probability": np.round(y_proba, 4),
    "predicted_label"      : y_pred,
    "threshold"            : best_threshold,
})
pred_path = os.path.join(output_folder, "predictions.csv")
pred_df.to_csv(pred_path, index=False)
print(f"\n  Predictions saved   : {pred_path}")

# metrics.json
metrics_dict = {
    "accuracy"        : round(acc,  4),
    "precision"       : round(prec, 4),
    "recall"          : round(rec,  4),
    "f1_score"        : round(f1,   4),
    "log_loss"        : round(ll,   4),
    "threshold"       : best_threshold,
    "confusion_matrix": {
        "TP": TP, "TN": TN,
        "FP": FP, "FN": FN
    }
}
metrics_path = os.path.join(output_folder, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics_dict, f, indent=4)
print(f"  Metrics saved       : {metrics_path}")

# threshold_comparison.csv
thresh_df   = pd.DataFrame(threshold_rows)
thresh_path = os.path.join(output_folder, "threshold_comparison.csv")
thresh_df.to_csv(thresh_path, index=False)
print(f"  Threshold comparison: {thresh_path}")

# loss_curve.png
plt.figure(figsize=(8, 5))
plt.plot(model.losses, color="blue")
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
loss_path = os.path.join(output_folder, "loss_curve.png")
plt.savefig(loss_path)
plt.close()
print(f"  Loss curve saved    : {loss_path}")

print("\n" + "=" * 54)
print("  Done!")
print("=" * 54)