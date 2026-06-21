# Data generator 
# ya file 300 students ka fake data banai banaygi or cvc mai save kare gi

import numpy as np
import pandas as pd
import os

# random numbers hamesha same rahe ga, take har bar same data bane
np.random.seed(42)

# kitne students ka data banaya hai 
n = 300

#==================================================

# features banano  --> inputs 

# har feature ki range assingment mai dewi hai 

# roz kitne ghante perha 0 sa 50 
study_hours = np.random.uniform(0, 50, n)

# class attendence   0% sa 100% tak
attendance = np.random.uniform(0, 100, n)

# previous test score  - 0 sa 100 tak 
previous_score = np.random.uniform(0, 100, n)

# kitni assignments ki — 0 se 20 tak
assignments = np.random.uniform(0, 20, n)

# practice questions — 0 se 500 tak
practice_questions = np.random.uniform(0, 500, n)

# neend — 0 se 12 ghante tak
sleep_hours = np.random.uniform(0, 12, n)

# ==================================================

# TARGET BANAO (jo predict karna hai)

# final exam score — in sab features se milke banta hai
# har feature ka thora asar hota hai score pe

final_score = (
    study_hours      * 0.5  +   # zyada padho → score zyada
    attendance       * 0.2  +   # attendance ka bhi asar
    previous_score   * 0.3  +   # pichla score bhi matter karta hai
    assignments      * 0.4  +   # assignments bhi count hoti hain
    practice_questions * 0.02 + # practice ka bhi thoda asar
    sleep_hours      * 0.3  +   # neend bhi zaroori hai
    np.random.uniform(-5, 5, n) # thoda random variation
)

# score 0 se 100 ke beech rakho
# agar koi value 100 se zyada ya 0 se kam ho — clip karo
final_score = np.clip(final_score, 0, 100)

# =============================================================

# CSV FILE BANAO
# sab features ek DataFrame mein dalo
df = pd.DataFrame({
    "study_hours_per_week"    : np.round(study_hours, 1),
    "attendance_percentage"   : np.round(attendance, 1),
    "previous_test_score"     : np.round(previous_score, 1),
    "assignments_completed"   : np.round(assignments, 1),
    "practice_questions_solved": np.round(practice_questions, 0),
    "sleep_hours_per_day"     : np.round(sleep_hours, 1),
    "final_exam_score"        : np.round(final_score, 1),
})

# yeh file jahan hai, wahan se data folder dhundo
script_folder = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_folder, "data", "student_scores.csv")

df.to_csv(csv_path, index=False)

print(f"CSV ban gayi! Total rows: {len(df)}")
print(f"\nPehli 3 rows:")
print(df.head(3))