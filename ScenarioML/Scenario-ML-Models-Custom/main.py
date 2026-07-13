# ─────────────────────────────────────────────
# MAIN — SCENARIO ML MODELS CUSTOM
# 3 models: Poisson, Perceptron, Naive Bayes
# ─────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

from preprocessing       import (encode_categorical, handle_missing,
                                  scale_features, train_val_test_split,
                                  tokenize)
from poisson_regression  import PoissonRegressionCustom
from perceptron          import PerceptronCustom
from naive_bayes         import NaiveBayesCustom
from metrics             import (mae, rmse, poisson_deviance,
                                  confusion_matrix, accuracy,
                                  precision, recall, f1_score,
                                  multiclass_metrics)

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

script_folder = os.path.dirname(os.path.abspath(__file__))
csv_path      = os.path.join(script_folder, "data",
                             "student_support_sample.csv")
output_folder = os.path.join(script_folder, "outputs")
os.makedirs(output_folder, exist_ok=True)

# ─────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────

df = pd.read_csv(csv_path)

print("=" * 60)
print("  SCENARIO ML MODELS — Student Support System")
print("=" * 60)
print(f"\n  Dataset loaded successfully!")
print(f"  Rows    : {df.shape[0]}")
print(f"  Columns : {df.shape[1]}")
print(f"\n  Targets detected:")
print(f"  Part 1 count target  : support_requests_next_14d")
print(f"  Part 2 binary target : high_risk_intervention")
print(f"  Part 3 multi-class   : dominant_issue_type")

# ─────────────────────────────────────────────
# STEP 2 — PREPROCESSING
# ─────────────────────────────────────────────

# Yes/No encode
cat_cols = ["scholarship_status", "internet_access"]
df = encode_categorical(df, cat_cols)

# numeric cols
numeric_cols = [
    "week_no", "attendance_percentage",
    "missed_classes_14d", "lms_logins_7d",
    "study_hours_per_week", "previous_exam_score",
    "assignments_completed_pct", "late_submissions_30d",
    "fee_pending_amount", "counselor_visits_30d",
    "scholarship_status", "internet_access"
]

df = handle_missing(df, numeric_cols)

# student ids save karo
student_ids = df["student_id"].values

# advisor notes save karo
advisor_notes = df["advisor_note"].values

# ─────────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────────

feature_cols = numeric_cols

X       = df[feature_cols].values
y_count = df["support_requests_next_14d"].values
y_risk  = df["high_risk_intervention"].values
y_issue = df["dominant_issue_type"].values

# ─────────────────────────────────────────────
# STEP 3 — SPLIT
# ─────────────────────────────────────────────

(X_train, X_val, X_test,
 y_count_train, y_count_val, y_count_test,
 test_idx) = train_val_test_split(X, y_count)

(_, _, _,
 y_risk_train, y_risk_val, y_risk_test,
 _) = train_val_test_split(X, y_risk)

(_, _, _,
 y_issue_train, y_issue_val, y_issue_test,
 _) = train_val_test_split(X, y_issue)

# same indices use karo jo split mein use hue
np.random.seed(42)
all_indices = np.arange(len(advisor_notes))
np.random.shuffle(all_indices)

train_count = len(y_count_train)
val_count   = len(y_count_val)

train_idx_notes = all_indices[:train_count]
test_idx_notes  = all_indices[train_count + val_count:]

notes_train = advisor_notes[train_idx_notes]
notes_test  = advisor_notes[test_idx_notes]
test_ids    = student_ids[test_idx]

print(f"\n  Train size      : {len(y_count_train)}")
print(f"  Validation size : {len(y_count_val)}")
print(f"  Test size       : {len(y_count_test)}")

# ─────────────────────────────────────────────
# STEP 4 — SCALE
# ─────────────────────────────────────────────

(X_train_s, X_val_s, X_test_s,
 mean, std) = scale_features(X_train, X_val, X_test)

# ═════════════════════════════════════════════
# PART 1 — POISSON REGRESSION
# ═════════════════════════════════════════════

print("\n" + "=" * 60)
print("  PART 1 — POISSON REGRESSION")
print("=" * 60)
print("\n  Training shuru...\n")

poisson = PoissonRegressionCustom(
    learning_rate=0.001, epochs=2000)
poisson.fit(X_train_s, y_count_train)

mu_test, count_pred = poisson.predict(X_test_s)

# baseline — hamesha mean predict karo
baseline_pred = np.full_like(
    y_count_test,
    fill_value=y_count_train.mean(),
    dtype=float)

mae_poisson  = mae(y_count_test, count_pred)
rmse_poisson = rmse(y_count_test, count_pred)
dev_poisson  = poisson_deviance(y_count_test, mu_test)
mae_baseline = mae(y_count_test, baseline_pred)

print(f"\n  Final Test MAE         : {mae_poisson:.4f}")
print(f"  Final Test RMSE        : {rmse_poisson:.4f}")
print(f"  Mean Poisson Deviance  : {dev_poisson:.4f}")
print(f"  Baseline MAE           : {mae_baseline:.4f}")
beats = "YES ✅" if mae_poisson < mae_baseline else "NO ❌"
print(f"  Poisson beats baseline : {beats}")

# save predictions
pd.DataFrame({
    "student_id"     : test_ids,
    "actual_count"   : y_count_test,
    "predicted_rate" : np.round(mu_test, 4),
    "predicted_count": count_pred,
}).to_csv(os.path.join(output_folder,
          "predictions_poisson.csv"), index=False)
print(f"\n  Saved: outputs/predictions_poisson.csv")

# ═════════════════════════════════════════════
# PART 2 — PERCEPTRON
# ═════════════════════════════════════════════

print("\n" + "=" * 60)
print("  PART 2 — PERCEPTRON")
print("=" * 60)
print("\n  Training shuru...\n")

perc = PerceptronCustom(learning_rate=0.01, epochs=1000)
perc.fit(X_train_s, y_risk_train)

y_risk_pred = perc.predict(X_test_s)
raw_scores  = perc.raw_score(X_test_s)

TP, TN, FP, FN = confusion_matrix(y_risk_test, y_risk_pred)
acc  = accuracy(y_risk_test, y_risk_pred)
prec = precision(TP, FP)
rec  = recall(TP, FN)
f1   = f1_score(prec, rec)

print(f"\n  Confusion Matrix:")
print(f"  TP:{TP}  TN:{TN}  FP:{FP}  FN:{FN}")
print(f"\n  Accuracy  : {acc:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1 Score  : {f1:.4f}")

pd.DataFrame({
    "student_id"     : test_ids,
    "actual_label"   : y_risk_test,
    "predicted_label": y_risk_pred,
    "raw_score"      : np.round(raw_scores, 4),
}).to_csv(os.path.join(output_folder,
          "predictions_perceptron.csv"), index=False)
print(f"\n  Saved: outputs/predictions_perceptron.csv")

# ═════════════════════════════════════════════
# PART 3 — NAIVE BAYES
# ═════════════════════════════════════════════

print("\n" + "=" * 60)
print("  PART 3 — NAIVE BAYES")
print("=" * 60)
print("\n  Training shuru...\n")

# tokenize karo
train_tokens = [tokenize(note) for note in notes_train]
test_tokens  = [tokenize(note) for note in notes_test]

nb = NaiveBayesCustom(alpha=1.0)
nb.fit(train_tokens, y_issue_train)

y_issue_pred, top_scores = nb.predict(test_tokens)

classes = ["Academic", "Attendance", "Financial",
           "Personal", "Stable"]
results, macro_f1 = multiclass_metrics(
    y_issue_test, y_issue_pred, classes)

print(f"  Macro F1 Score : {macro_f1:.4f}")
for cls in classes:
    print(f"  {cls:<12} → "
          f"P:{results[cls]['precision']:.2f} "
          f"R:{results[cls]['recall']:.2f} "
          f"F1:{results[cls]['f1']:.2f}")

print("\n  Top words per class:")
for cls in classes:
    words = [w for w, _ in nb.top_words[cls]]
    print(f"  {cls:<12}: {', '.join(words)}")

pd.DataFrame({
    "student_id"          : test_ids,
    "actual_issue_type"   : y_issue_test,
    "predicted_issue_type": y_issue_pred,
    "top_predicted_score" : np.round(top_scores, 4),
}).to_csv(os.path.join(output_folder,
          "predictions_naive_bayes.csv"), index=False)
print(f"\n  Saved: outputs/predictions_naive_bayes.csv")

# ═════════════════════════════════════════════
# LOSS CURVE
# ═════════════════════════════════════════════

plt.figure(figsize=(8, 5))
plt.plot(poisson.losses, color="blue",
         label="Poisson Loss")
plt.title("Training Loss Curve — Poisson Regression")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
loss_path = os.path.join(output_folder, "loss_curves.png")
plt.savefig(loss_path)
plt.close()
print(f"\n  Saved: outputs/loss_curves.png")

# ═════════════════════════════════════════════
# METRICS JSON
# ═════════════════════════════════════════════

metrics_dict = {
    "poisson": {
        "mae"             : round(float(mae_poisson),  4),
        "rmse"            : round(float(rmse_poisson), 4),
        "poisson_deviance": round(float(dev_poisson),  4),
        "baseline_mae"    : round(float(mae_baseline), 4),
        "beats_baseline"  : bool(mae_poisson < mae_baseline)
    },
    "perceptron": {
        "accuracy" : round(float(acc),  4),
        "precision": round(float(prec), 4),
        "recall"   : round(float(rec),  4),
        "f1_score" : round(float(f1),   4),
        "TP": TP, "TN": TN, "FP": FP, "FN": FN
    },
    "naive_bayes": {
        "macro_f1"   : round(float(macro_f1), 4),
        "per_class"  : results
    }
}

with open(os.path.join(output_folder,
          "metrics.json"), "w") as f:
    json.dump(metrics_dict, f, indent=4)

model_comparison = {
    "part1_best_model": "PoissonRegressionCustom",
    "part1_reason"    : "Count target, non-negative, beats baseline",
    "part2_best_model": "PerceptronCustom",
    "part2_reason"    : "Binary decision, fast, interpretable",
    "part3_best_model": "NaiveBayesCustom",
    "part3_reason"    : "Text classification, multi-class, log probs"
}

with open(os.path.join(output_folder,
          "model_comparison.json"), "w") as f:
    json.dump(model_comparison, f, indent=4)

print(f"\n  Saved: outputs/metrics.json")
print(f"  Saved: outputs/model_comparison.json")

# ═════════════════════════════════════════════
# MODEL SELECTION SUMMARY
# ═════════════════════════════════════════════

print("\n" + "=" * 60)
print("  MODEL SELECTION SUMMARY")
print("=" * 60)
print(f"""
  1. Best model for support request count:
     → Poisson Regression
     Reason: Count target, never negative,
     MAE={mae_poisson:.2f} beats baseline {mae_baseline:.2f}

  2. Best model for high-risk alerting:
     → Perceptron
     Reason: Binary decision, F1={f1:.2f},
     Recall={rec:.2f} (missing risk is costly!)

  3. Best model for dominant issue type:
     → Naive Bayes
     Reason: Text classification, Macro F1={macro_f1:.2f},
     handles unseen words with Laplace smoothing
""")

print("=" * 60)
print("  Done!")
print("=" * 60)