import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ─────────────────────────────────────────────────────────────
# DATASET
# Predicting if a student is ADMITTED (1) or REJECTED (0)
# Features: [test_score (out of 100), interview_score (out of 10)]
# This data is designed to be roughly linearly separable —
# Perceptron needs this to converge properly.
# ─────────────────────────────────────────────────────────────

X = np.array([
    [45, 3], [50, 4], [55, 3.5], [40, 2], [48, 3],
    [85, 8], [90, 9], [88, 8.5], [92, 9.5], [80, 7.5],
    [42, 2.5], [38, 1.5], [52, 4], [47, 3], [44, 2],
    [95, 9], [82, 8], [87, 9], [91, 8.5], [84, 7],
    [50, 3.5], [46, 2.5], [39, 2], [53, 4], [41, 3],
    [89, 8.5], [93, 9.5], [81, 7], [86, 8], [94, 9],
], dtype=float)

# 1 = admitted, 0 = rejected
y = np.array([
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1,
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1,
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1,
])

feature_names = ["Test Score", "Interview Score"]

# ─────────────────────────────────────────────────────────────
# STEP 1 — SPLIT
# ─────────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("=" * 54)
print("  PERCEPTRON — Student Admission Classifier")
print("=" * 54)
print(f"\n  Total students  : {len(y)}")
print(f"  Training set    : {len(y_train)} students")
print(f"  Test set        : {len(y_test)} students")
print(f"  Admitted        : {int(y.sum())} students")
print(f"  Rejected        : {int((y==0).sum())} students")

# ─────────────────────────────────────────────────────────────
# STEP 2 — NORMALIZE
# ─────────────────────────────────────────────────────────────

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────
# STEP 3 — TRAIN
# Perceptron update rule: θ := θ + α(y - ŷ)x  (only on mistakes)
# ─────────────────────────────────────────────────────────────

model = Perceptron(max_iter=1000, eta0=1.0, random_state=42)
model.fit(X_train_scaled, y_train)

print("\n  Model trained successfully.")
print(f"  Iterations until convergence : {model.n_iter_}")

print("\n  Learned Weights (θ):")
print(f"  Bias (θ₀)         : {model.intercept_[0]:+.4f}")
for name, coef in zip(feature_names, model.coef_[0]):
    print(f"  {name:<18}: {coef:+.4f}")

# ─────────────────────────────────────────────────────────────
# STEP 4 — EVALUATE
# ─────────────────────────────────────────────────────────────

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 54)
print("  EVALUATION ON TEST SET")
print("=" * 54)
print(f"\n  Accuracy : {accuracy * 100:.1f}%")

print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Rejected (0)", "Admitted (1)"]))

cm = confusion_matrix(y_test, y_pred)
print("  Confusion Matrix:")
print(f"  {'':22} Predicted REJ   Predicted ADM")
print(f"  {'Actual REJECTED':22} {cm[0][0]:^15} {cm[0][1]:^13}")
print(f"  {'Actual ADMITTED':22} {cm[1][0]:^15} {cm[1][1]:^13}")

print("\n  Predicted vs Actual:")
print(f"  {'#':<5} {'Test':>6} {'Interview':>10} {'Predicted':>11} {'Actual':>9}")
print(f"  {'-'*46}")
for i in range(len(y_test)):
    pred_label = "ADMIT" if y_pred[i] == 1 else "REJECT"
    actual_label = "ADMIT" if y_test[i] == 1 else "REJECT"
    match = "✓" if y_pred[i] == y_test[i] else "✗"
    print(f"  {i+1:<5} {X_test[i][0]:>6.0f} {X_test[i][1]:>10.1f} {pred_label:>11} {actual_label:>9}  {match}")

# ─────────────────────────────────────────────────────────────
# STEP 5 — PREDICT NEW STUDENTS
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 54)
print("  PREDICT NEW STUDENTS")
print("=" * 54)

new_students = [
    [60, 5],    # borderline case
    [95, 9.5],  # strong candidate
    [35, 1.5],  # weak candidate
]
descriptions = [
    "Test: 60 | Interview: 5.0  (borderline)",
    "Test: 95 | Interview: 9.5  (strong)",
    "Test: 35 | Interview: 1.5  (weak)",
]

new_scaled = scaler.transform(new_students)
predictions = model.predict(new_scaled)

for desc, pred in zip(descriptions, predictions):
    label = "ADMITTED ✅" if pred == 1 else "REJECTED ❌"
    print(f"\n  {desc}")
    print(f"  Decision: {label}")

print("\n  Note: Perceptron gives a HARD decision only — no probability/confidence score.")
print("=" * 54)
print("  Done.")
print("=" * 54)