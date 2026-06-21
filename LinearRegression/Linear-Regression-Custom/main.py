# ─────────────────────────────────────────────
# MAIN — SAB KO JODO
# Yahan sab files milke kaam karengi
# ─────────────────────────────────────────────

import numpy as np
import pandas as pd
import os

from preprocessing      import scale_features, train_test_split
from linear_regression  import LinearRegressionCustom
from metrics            import mae, mse, rmse, r2_score

# ─────────────────────────────────────────────
# STEP 1 — CSV LOAD KARO
# ─────────────────────────────────────────────

script_folder = os.path.dirname(os.path.abspath(__file__))
csv_path      = os.path.join(script_folder, "data", "student_scores.csv")

df = pd.read_csv(csv_path)

print("=" * 50)
print("  LINEAR REGRESSION — Student Scores")
print("=" * 50)
print(f"\n  Rows loaded    : {df.shape[0]}")
print(f"  Columns loaded : {df.shape[1]}")

# ─────────────────────────────────────────────
# STEP 2 — FEATURES AUR TARGET ALAG KARO
# ─────────────────────────────────────────────

# sab columns features hain — sirf last column chhod ke
X = df.drop(columns=["final_exam_score"]).values
y = df["final_exam_score"].values

print(f"\n  Features : {list(df.columns[:-1])}")
print(f"  Target   : final_exam_score")

# ─────────────────────────────────────────────
# STEP 3 — TRAIN/TEST SPLIT
# ─────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(X, y)

print(f"\n  Training rows : {len(y_train)}")
print(f"  Testing rows  : {len(y_test)}")

# ─────────────────────────────────────────────
# STEP 4 — SCALE KARO
# ─────────────────────────────────────────────

X_train_s, X_test_s, mean, std = scale_features(X_train, X_test)

# ─────────────────────────────────────────────
# STEP 5 — MODEL TRAIN KARO
# ─────────────────────────────────────────────

print("\n  Training shuru...\n")

model = LinearRegressionCustom(learning_rate=0.01, epochs=1000)
model.fit(X_train_s, y_train)

print(f"\n  Learned Weights : {model.weights}")
print(f"  Learned Bias    : {model.bias:.4f}")

# ─────────────────────────────────────────────
# STEP 6 — EVALUATE KARO
# ─────────────────────────────────────────────

y_pred = model.predict(X_test_s)

print("\n" + "=" * 50)
print("  RESULTS")
print("=" * 50)
print(f"\n  MAE  : {mae(y_test,  y_pred):.4f}")
print(f"  MSE  : {mse(y_test,  y_pred):.4f}")
print(f"  RMSE : {rmse(y_test, y_pred):.4f}")
print(f"  R²   : {r2_score(y_test, y_pred):.4f}")

# ─────────────────────────────────────────────
# STEP 7 — PREDICTIONS CSV SAVE KARO
# ─────────────────────────────────────────────

output_folder = os.path.join(script_folder, "outputs")
os.makedirs(output_folder, exist_ok=True)

results_df = pd.DataFrame({
    "actual"    : y_test,
    "predicted" : np.round(y_pred, 2)
})

output_path = os.path.join(output_folder, "predictions.csv")
results_df.to_csv(output_path, index=False)

print(f"\n  Predictions saved: {output_path}")
print("\n" + "=" * 50)
print("  Done!")
print("=" * 50)