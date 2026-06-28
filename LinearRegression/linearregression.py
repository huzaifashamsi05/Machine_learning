import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# DATASET
# Lahore house prices
# Features: [size_marla, bedrooms, age_years]
# Label:    price in Crore PKR
# ─────────────────────────────────────────────────────────────

X = np.array([
    [5,   2,  10],
    [8,   3,   5],
    [10,  3,   8],
    [12,  4,   3],
    [16,  4,   6],
    [5,   2,  20],
    [7,   3,  12],
    [9,   3,   2],
    [14,  4,   7],
    [20,  5,   4],
    [6,   2,   8],
    [11,  3,   5],
    [18,  5,   3],
    [8,   2,  15],
    [13,  4,   1],
    [32,  6,   5],
    [7,   3,   9],
    [10,  3,  14],
    [15,  4,   2],
    [22,  5,   6],
], dtype=float)

y = np.array([
    1.2, 2.0, 2.4, 3.0, 4.2,
    0.9, 1.7, 2.2, 3.5, 5.5,
    1.4, 2.7, 4.8, 1.6, 3.3,
    9.0, 1.8, 2.3, 3.8, 6.0,
], dtype=float)

feature_names = ["Size (Marla)", "Bedrooms", "Age (Years)"]

# ─────────────────────────────────────────────────────────────
# STEP 1 — SPLIT  (60% train, 40% test)
# ─────────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,    random_state=42)

print("=" * 52)
print("  LINEAR REGRESSION — Lahore House Prices")
print("=" * 52)
print(f"\n  Total houses  : {len(y)}")
print(f"  Training set  : {len(y_train)} houses")
print(f"  Test set      : {len(y_test)} houses")

# ─────────────────────────────────────────────────────────────
# STEP 2 — NORMALIZE  (Z-score: mean=0, std=1)
# fit only on train, apply same to test — no data leakage
# ─────────────────────────────────────────────────────────────

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform
X_test_scaled  = scaler.transform(X_test)        # transform only

# ─────────────────────────────────────────────────────────────
# STEP 3 — TRAIN
# ─────────────────────────────────────────────────────────────

model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("\n  Model trained successfully.")
print("\n  Learned Parameters:")
print(f"    Bias (θ₀)       : {model.intercept_:.4f}")
for name, coef in zip(feature_names, model.coef_):
    print(f"    {name:<16}: {coef:+.4f}")

# ─────────────────────────────────────────────────────────────
# STEP 4 — EVALUATE
# ─────────────────────────────────────────────────────────────

y_pred = model.predict(X_test_scaled)

r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n" + "=" * 52)
print("  EVALUATION ON TEST SET")
print("=" * 52)
print(f"\n  R² Score  : {r2:.4f}  (1.0 = perfect)")
print(f"  MAE       : {mae:.4f} Crore PKR  (avg error)")
print(f"  RMSE      : {rmse:.4f} Crore PKR  (big mistake penalty)")

print("\n  Predicted vs Actual:")
print(f"  {'#':<5} {'Predicted':>12} {'Actual':>12} {'Error':>12}")
print(f"  {'-' * 44}")
for i, (pred, actual) in enumerate(zip(y_pred, y_test)):
    error = pred - actual
    flag  = "  ← big miss" if abs(error) > 0.8 else ""
    print(f"  {i+1:<5} {pred:>10.2f} Cr {actual:>10.2f} Cr {error:>+10.2f}{flag}")

# ─────────────────────────────────────────────────────────────
# STEP 5 — PREDICT NEW HOUSES
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 52)
print("  PREDICT NEW HOUSES")
print("=" * 52)

new_houses = [
    [10, 3, 5],    # 10 Marla, 3 bed, 5 yrs old
    [16, 4, 2],    # 1 Kanal,  4 bed, 2 yrs old
    [32, 6, 10],   # 2 Kanal,  6 bed, 10 yrs old
]
descriptions = [
    "10 Marla | 3 bed | 5 yrs",
    " 1 Kanal | 4 bed | 2 yrs",
    " 2 Kanal | 6 bed | 10 yrs",
]

new_houses_scaled = scaler.transform(new_houses)   # same scaler
prices = model.predict(new_houses_scaled)

for desc, price in zip(descriptions, prices):
    print(f"\n  {desc}")
    print(f"  Predicted : {price:.2f} Crore PKR  (PKR {price*1e7:>12,.0f})")

print("\n" + "=" * 52)
print("  Done.")
print("=" * 52)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Graph 1 — Size vs Price
axes[0].scatter(X[:, 0], y, color="blue")
axes[0].set_xlabel("Size (Marla)")
axes[0].set_ylabel("Price (Crore PKR)")
axes[0].set_title("Size vs Price")

# Graph 2 — Bedrooms vs Price
axes[1].scatter(X[:, 1], y, color="green")
axes[1].set_xlabel("Bedrooms")
axes[1].set_ylabel("Price (Crore PKR)")
axes[1].set_title("Bedrooms vs Price")

# Graph 3 — Age vs Price
axes[2].scatter(X[:, 2], y, color="red")
axes[2].set_xlabel("Age (Years)")
axes[2].set_ylabel("Price (Crore PKR)")
axes[2].set_title("Age vs Price")

plt.tight_layout()
plt.show()