import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_poisson_deviance, mean_absolute_error

# ─────────────────────────────────────────────────────────────
# DATASET
# Predicting number of orders per day for an online store
# Features: [is_weekend, ad_spend_thousand_pkr, discount_percent, day_of_month]
# Label:    order_count (a COUNT — 0, 1, 2, 3… never negative)
# ─────────────────────────────────────────────────────────────

X = np.array([
    [0, 10, 5,  3],
    [1, 15, 10, 6],
    [0, 8,  0,  9],
    [1, 25, 15, 12],
    [0, 12, 5,  15],
    [1, 30, 20, 18],
    [0, 5,  0,  21],
    [1, 20, 10, 24],
    [0, 18, 5,  27],
    [1, 35, 25, 2],
    [0, 9,  0,  5],
    [1, 22, 15, 8],
    [0, 14, 5,  11],
    [1, 28, 20, 14],
    [0, 6,  0,  17],
    [1, 32, 25, 20],
    [0, 11, 5,  23],
    [1, 26, 15, 26],
    [0, 7,  0,  1],
    [1, 24, 10, 4],
    [0, 13, 5,  7],
    [1, 33, 20, 10],
    [0, 10, 0,  13],
    [1, 29, 15, 16],
    [0, 16, 5,  19],
], dtype=float)

# order counts — non-negative integers (Poisson-distributed)
y = np.array([
    12, 22, 8,  35, 15,
    40, 5,  28, 20, 45,
    9,  30, 17, 38, 6,
    42, 14, 33, 7,  31,
    16, 44, 11, 36, 19,
], dtype=float)

feature_names = ["Is Weekend", "Ad Spend (K PKR)", "Discount %", "Day of Month"]

# ─────────────────────────────────────────────────────────────
# STEP 1 — SPLIT
# ─────────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("=" * 54)
print("  POISSON REGRESSION — Daily Order Count Predictor")
print("=" * 54)
print(f"\n  Total days     : {len(y)}")
print(f"  Training set   : {len(y_train)} days")
print(f"  Test set       : {len(y_test)} days")
print(f"  Order range    : {int(y.min())} to {int(y.max())} orders/day")

# ─────────────────────────────────────────────────────────────
# STEP 2 — NORMALIZE
# ─────────────────────────────────────────────────────────────

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────
# STEP 3 — TRAIN
# alpha = regularization strength (prevents overfitting)
# ─────────────────────────────────────────────────────────────

model = PoissonRegressor(alpha=0.01, max_iter=1000)           # names = ["ALICE", "BOB", "CHARLIE", "DAVID"]
model.fit(X_train_scaled, y_train)

print("\n  Model trained successfully.")
print("\n  Learned Feature Weights:")
print(f"  {'Feature':<20} {'Weight':>10}   Effect")
print(f"  {'-'*54}")
for name, coef in zip(feature_names, model.coef_):
    effect = "↑ more orders" if coef > 0 else "↓ fewer orders"
    print(f"  {name:<20} {coef:>+10.4f}   {effect}")
print(f"\n  Bias (intercept)     : {model.intercept_:+.4f}")

# ─────────────────────────────────────────────────────────────
# STEP 4 — EVALUATE
# ─────────────────────────────────────────────────────────────

y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
deviance = mean_poisson_deviance(y_test, y_pred)

print("\n" + "=" * 54)
print("  EVALUATION ON TEST SET")
print("=" * 54)
print(f"\n  MAE                : {mae:.2f} orders  (avg error)")
print(f"  Poisson Deviance   : {deviance:.4f}  (lower = better fit)")

print("\n  Predicted vs Actual Orders:")
print(f"  {'Day':<6} {'Predicted':>12} {'Actual':>10} {'Error':>10}")
print(f"  {'-'*42}")
for i, (pred, actual) in enumerate(zip(y_pred, y_test)):
    error = pred - actual
    print(f"  {i+1:<6} {pred:>10.1f}   {actual:>8.0f}   {error:>+8.1f}")

# ─────────────────────────────────────────────────────────────
# STEP 5 — PREDICT NEW DAYS
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 54)
print("  PREDICT NEW DAYS")
print("=" * 54)

new_days = [
    [1, 30, 20, 15],   # weekend, high ad spend, high discount, mid-month
    [0, 5,  0,  10],   # weekday, low ad spend, no discount
    [1, 40, 30, 25],   # weekend, very high ad spend & discount, late month
]
descriptions = [
    "Weekend | Ad spend 30K | Discount 20% | Day 15",
    "Weekday | Ad spend 5K  | Discount 0%  | Day 10",
    "Weekend | Ad spend 40K | Discount 30% | Day 25",
]

new_scaled = scaler.transform(new_days)
predictions = model.predict(new_scaled)

for desc, pred in zip(descriptions, predictions):
    print(f"\n  {desc}")
    print(f"  Predicted orders : {pred:.1f}  (≈ {round(pred)} orders)")

print("\n" + "=" * 54)
print("  Done.")
print("=" * 54)