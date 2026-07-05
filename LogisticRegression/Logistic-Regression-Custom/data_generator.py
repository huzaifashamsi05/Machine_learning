# ─────────────────────────────────────────────
# DATA GENERATOR
# 300 students ka fake data banayega
# at_risk predict karne ke liye
# ─────────────────────────────────────────────

import numpy as np
import pandas as pd
import os

np.random.seed(42)
n = 300

# ─────────────────────────────────────────────
# STUDENT IDs
# ─────────────────────────────────────────────

# S001, S002, S003 ... S300
student_ids = [f"S{str(i).zfill(3)}" for i in range(1, n+1)]

# ─────────────────────────────────────────────
# FEATURES BANAO
# ─────────────────────────────────────────────

attendance        = np.random.uniform(0, 100, n)
study_hours       = np.random.uniform(0, 50, n)
previous_score    = np.random.uniform(0, 100, n)
assignments       = np.random.uniform(0, 20, n)
late_submissions  = np.random.uniform(0, 20, n)
participation     = np.random.uniform(0, 10, n)
sleep_hours       = np.random.uniform(0, 12, n)

# Yes/No features — 0 ya 1
internet_access   = np.random.choice([0, 1], n)
part_time_job     = np.random.choice([0, 1], n)

# ─────────────────────────────────────────────
# TARGET BANAO — at_risk
# ─────────────────────────────────────────────

# risk score calculate karo
# zyada attendance  → kam risk
# zyada study       → kam risk
# zyada late sub    → zyada risk
# part time job     → zyada risk

risk_score = (
    -0.08 * attendance        +  # sabse zyada asar
    -0.07 * study_hours       +  # bahut zyada asar
    -0.03 * previous_score    +  # kam asar
    -0.05 * assignments       +  # medium asar
    +0.05 * late_submissions  +  # medium asar
    -0.04 * participation     +  # medium asar
    -0.02 * sleep_hours       +  # kam asar
    -0.01 * internet_access   +  # kam asar
    +0.08 * part_time_job     +  # medium asar ← changed
    np.random.uniform(-1, 1, n)
)

risk_score = (risk_score - risk_score.mean()) / risk_score.std() * 1.5

# sigmoid se probability banao
prob = 1 / (1 + np.exp(-risk_score))

# threshold 0.5 se class banao
at_risk = (prob >= 0.5).astype(int)

# ─────────────────────────────────────────────
# CSV BANAO
# ─────────────────────────────────────────────

df = pd.DataFrame({
    "student_id"               : student_ids,
    "attendance_percentage"    : np.round(attendance, 1),
    "study_hours_per_week"     : np.round(study_hours, 1),
    "previous_exam_score"      : np.round(previous_score, 1),
    "assignments_completed"    : np.round(assignments, 1),
    "late_submissions"         : np.round(late_submissions, 1),
    "class_participation_score": np.round(participation, 1),
    "sleep_hours_per_day"      : np.round(sleep_hours, 1),
    "internet_access"          : internet_access,
    "part_time_job"            : part_time_job,
    "at_risk"                  : at_risk,
})

# save karo
script_folder = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_folder, "data", "student_risk_sample.csv")
df.to_csv(csv_path, index=False)

print(f"CSV ban gayi! Total rows: {len(df)}")
print(f"At Risk students    : {at_risk.sum()}")
print(f"Not At Risk students: {(at_risk==0).sum()}")
print(f"\nPehli 3 rows:")
print(df.head(3))