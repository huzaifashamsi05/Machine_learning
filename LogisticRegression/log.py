import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# ─────────────────────────────────────────────────────────────
# DATASET
# Predicting whether a patient has diabetes (1) or not (0)
#
# Features:
#   - age              (years)
#   - bmi              (body mass index)
#   - blood_glucose    (mg/dL)
#   - blood_pressure   (mmHg)
#   - family_history   (1 = yes, 0 = no)
# ─────────────────────────────────────────────────────────────

X = np.array([
    # age   bmi   glucose   bp    family
    [25,   22.5,   85,     70,     0],
    [45,   28.3,  140,     88,     1],
    [34,   24.1,   92,     72,     0],
    [52,   31.5,  175,     95,     1],
    [29,   20.8,   80,     65,     0],
    [61,   35.2,  210,    102,     1],
    [38,   27.4,  130,     84,     1],
    [23,   19.5,   78,     62,     0],
    [55,   33.0,  195,     98,     1],
    [41,   26.8,  115,     80,     0],
    [48,   29.7,  160,     90,     1],
    [30,   23.2,   88,     68,     0],
    [63,   36.5,  220,    105,     1],
    [36,   25.0,  100,     74,     0],
    [50,   30.2,  170,     93,     1],
    [27,   21.3,   82,     66,     0],
    [44,   28.9,  145,     86,     1],
    [58,   34.1,  200,    100,     1],
    [32,   24.8,   95,     71,     0],
    [46,   29.1,  155,     89,     1],
    [22,   18.9,   76,     60,     0],
    [57,   32.8,  190,     97,     1],
    [39,   26.2,  120,     78,     0],
    [53,   31.9,  180,     96,     1],
    [31,   23.7,   90,     69,     0],
    [60,   35.8,  215,    103,     1],
    [28,   22.0,   83,     67,     0],
    [47,   29.5,  158,     91,     1],
    [35,   25.6,  105,     76,     0],
    [56,   33.5,  198,     99,     1],
], dtype=float)

# labels: 1 = has diabetes, 0 = no diabetes
y = np.array([
    0, 1, 0, 1, 0,
    1, 1, 0, 1, 0,
    1, 0, 1, 0, 1,
    0, 1, 1, 0, 1,
    0, 1, 0, 1, 0,
    1, 0, 1, 0, 1,
])

feature_names = ["Age", "BMI", "Blood Glucose", "Blood Pressure", "Family History"]

# ─────────────────────────────────────────────────────────────
# STEP 1 — SPLIT  (60% train, 40% test)
# ─────────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.4,
    random_state=42
)

print("=" * 54)
print("  LOGISTIC REGRESSION — Diabetes Disease Predictor")
print("=" * 54)
print(f"\n  Total patients  : {len(y)}")
print(f"  Training set    : {len(y_train)} patients")
print(f"  Test set        : {len(y_test)} patients")
print(f"  Diabetic (1)    : {int(y.sum())} patients")
print(f"  Non-diabetic (0): {int((y == 0).sum())} patients")

# ─────────────────────────────────────────────────────────────
# STEP 2 — NORMALIZE
# logistic regression is sensitive to feature scale
# standardize: mean=0, std=1 (fit on train only)
# ─────────────────────────────────────────────────────────────

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)   # same scaler, not refit

# ─────────────────────────────────────────────────────────────
# STEP 3 — TRAIN
# solver='lbfgs' is the default and works well for small datasets
# max_iter=1000 gives it enough iterations to converge
# ─────────────────────────────────────────────────────────────

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

print("\n  Model trained successfully.")

# show what the model learned
print("\n  Learned Feature Weights (θ):")
print(f"  {'Feature':<18} {'Weight':>10}   Meaning")
print(f"  {'-'*52}")
for name, coef in zip(feature_names, model.coef_[0]):
    direction = "↑ increases risk" if coef > 0 else "↓ decreases risk"
    print(f"  {name:<18} {coef:>+10.4f}   {direction}")
print(f"\n  Bias (θ₀)         : {model.intercept_[0]:+.4f}")

# ─────────────────────────────────────────────────────────────
# STEP 4 — EVALUATE
# ─────────────────────────────────────────────────────────────

y_pred       = model.predict(X_test_scaled)          # class (0 or 1)
y_pred_proba = model.predict_proba(X_test_scaled)    # probability for each class

accuracy = accuracy_score(y_test, y_pred)
auc      = roc_auc_score(y_test, y_pred_proba[:, 1])

print("\n" + "=" * 54)
print("  EVALUATION ON TEST SET")
print("=" * 54)
print(f"\n  Accuracy  : {accuracy * 100:.1f}%")
print(f"  ROC-AUC   : {auc:.4f}  (1.0 = perfect, 0.5 = random guess)")

# classification report
print("\n  Detailed Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["No Diabetes (0)", "Diabetes (1)"]
))

# confusion matrix — explained clearly
cm = confusion_matrix(y_test, y_pred)
print("  Confusion Matrix:")
print(f"  {'':25} Predicted NO    Predicted YES")
print(f"  {'Actual NO  (healthy)':25} {cm[0][0]:^15} {cm[0][1]:^13}")
print(f"  {'Actual YES (diabetic)':25} {cm[1][0]:^15} {cm[1][1]:^13}")
print(f"\n  True Negatives  (correct NO)  : {cm[0][0]}")
print(f"  True Positives  (correct YES) : {cm[1][1]}")
print(f"  False Positives (wrong alarm) : {cm[0][1]}")
print(f"  False Negatives (missed case) : {cm[1][0]}  ← most dangerous in medicine")

# prediction table
print("\n  Predicted vs Actual (all test patients):")
print(f"  {'#':<5} {'Age':>5} {'BMI':>6} {'Glucose':>9} {'Prob(Diabetes)':>16} {'Predicted':>11} {'Actual':>8} {'':>5}")
print(f"  {'-' * 68}")
for i in range(len(y_test)):
    prob      = y_pred_proba[i][1]           # probability of diabetes
    predicted = "DIABETIC" if y_pred[i] == 1 else "healthy"
    actual    = "DIABETIC" if y_test[i] == 1 else "healthy"
    match     = "✓" if y_pred[i] == y_test[i] else "✗ WRONG"
    age       = int(X_test[i][0])
    bmi       = X_test[i][1]
    glucose   = int(X_test[i][2])
    print(f"  {i+1:<5} {age:>5} {bmi:>6.1f} {glucose:>9} {prob:>15.1%} {predicted:>11} {actual:>8}   {match}")

# ─────────────────────────────────────────────────────────────
# STEP 5 — PREDICT NEW PATIENTS
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 54)
print("  PREDICT NEW PATIENTS")
print("=" * 54)

new_patients = [
    # age  bmi   glucose  bp   family_history
    [40,  27.0,  120,    82,    0],   # middle-aged, normal glucose
    [58,  34.5,  200,   100,    1],   # older, high glucose, family history
    [24,  19.2,   79,    63,    0],   # young, healthy numbers
]
descriptions = [
    "Age 40 | BMI 27 | Glucose 120 | BP 82 | No family history",
    "Age 58 | BMI 34.5 | Glucose 200 | BP 100 | Family history YES",
    "Age 24 | BMI 19.2 | Glucose 79 | BP 63 | No family history",
]

new_scaled = scaler.transform(new_patients)
new_preds  = model.predict(new_scaled)
new_probas = model.predict_proba(new_scaled)

for i, (desc, pred, proba) in enumerate(zip(descriptions, new_preds, new_probas)):
    prob_diabetes = proba[1]
    label = "DIABETIC" if pred == 1 else "NOT DIABETIC"

    if prob_diabetes >= 0.75:
        risk = "HIGH RISK"
    elif prob_diabetes >= 0.40:
        risk = "MODERATE RISK"
    else:
        risk = "LOW RISK"

    print(f"\n  Patient {i+1}: {desc}")
    print(f"  P(Diabetes) = {prob_diabetes:.1%}  →  {label}  [{risk}]")

print("\n" + "=" * 54)
print("  Note: 0.5 is the decision threshold.")
print("  P ≥ 0.5 → model predicts DIABETIC")
print("  P < 0.5 → model predicts NOT DIABETIC")
print("=" * 54)