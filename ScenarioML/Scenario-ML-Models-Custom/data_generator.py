# ─────────────────────────────────────────────
# DATA GENERATOR
# 600 students ka fake data banayega
# 3 targets ke liye:
# 1. support_requests_next_14d (count)
# 2. high_risk_intervention (0/1)
# 3. dominant_issue_type (5 classes)
# ─────────────────────────────────────────────

import numpy as np
import pandas as pd
import os

np.random.seed(42)
n = 600

# ─────────────────────────────────────────────
# STUDENT IDs
# ─────────────────────────────────────────────

student_ids = [f"S{str(i).zfill(4)}" for i in range(1, n+1)]

# ─────────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────────

week_no                  = np.random.randint(1, 17, n)
attendance_pct           = np.random.uniform(0, 100, n)
missed_classes_14d       = np.random.randint(0, 31, n)
lms_logins_7d            = np.random.randint(0, 101, n)
study_hours              = np.random.uniform(0, 60, n)
previous_exam_score      = np.random.uniform(0, 100, n)
assignments_completed_pct= np.random.uniform(0, 100, n)
late_submissions_30d     = np.random.randint(0, 21, n)
fee_pending              = np.random.choice(
                            [0, 5000, 10000, 25000, 50000], n)
counselor_visits         = np.random.randint(0, 11, n)
scholarship_status       = np.random.choice([0, 1], n)
internet_access          = np.random.choice([0, 1], n)

# ─────────────────────────────────────────────
# ADVISOR NOTES — text data
# ─────────────────────────────────────────────

academic_notes = [
    "student struggling with exam preparation",
    "failed midterm needs academic support",
    "poor quiz scores needs tutoring help",
    "difficulty understanding course material",
    "requested help with assignment deadlines",
]

attendance_notes = [
    "missed multiple classes this week",
    "absent frequently due to transport issues",
    "attendance below required percentage",
    "missed labs and practical sessions",
    "irregular attendance pattern observed",
]

financial_notes = [
    "fee pending unable to access resources",
    "financial difficulty affecting studies",
    "scholarship issue needs resolution",
    "cannot afford course materials",
    "fee issue causing stress and absence",
]

personal_notes = [
    "family issues affecting performance",
    "health problems causing absences",
    "personal crisis needs counseling support",
    "mental health concerns raised by student",
    "dealing with personal emergency situation",
]

stable_notes = [
    "student performing well no issues",
    "good progress all assignments submitted",
    "active participation no concerns raised",
    "on track with studies no support needed",
    "excellent attendance and performance",
]

# ─────────────────────────────────────────────
# TARGET 3 — dominant_issue_type
# ─────────────────────────────────────────────

issue_types  = ["Academic", "Attendance", "Financial",
                "Personal", "Stable"]
issue_notes  = [academic_notes, attendance_notes,
                financial_notes, personal_notes, stable_notes]

dominant_issue = []
advisor_notes  = []

for i in range(n):
    # issue decide karo features se
    if fee_pending[i] > 10000:
        issue = "Financial"
    elif attendance_pct[i] < 40:
        issue = "Attendance"
    elif previous_exam_score[i] < 40:
        issue = "Academic"
    elif counselor_visits[i] >= 3:
        issue = "Personal"
    else:
        issue = "Stable"

    dominant_issue.append(issue)
    idx   = issue_types.index(issue)
    note  = np.random.choice(issue_notes[idx])
    advisor_notes.append(note)

dominant_issue = np.array(dominant_issue)

# ─────────────────────────────────────────────
# TARGET 2 — high_risk_intervention
# ─────────────────────────────────────────────

risk_score = (
    -0.04 * attendance_pct        +
    +0.05 * missed_classes_14d    +
    -0.02 * lms_logins_7d         +
    -0.03 * study_hours           +
    -0.03 * previous_exam_score   +
    +0.04 * late_submissions_30d  +
    +0.00001 * fee_pending        +
    +0.10 * counselor_visits      +
    np.random.uniform(-1, 1, n)
)

risk_score = (risk_score - risk_score.mean()) / risk_score.std() * 1.5
high_risk  = (1 / (1 + np.exp(-risk_score)) >= 0.5).astype(int)

# ─────────────────────────────────────────────
# TARGET 1 — support_requests_next_14d
# ─────────────────────────────────────────────

lambda_rate = np.exp(
    0.03 * missed_classes_14d   +
    0.02 * late_submissions_30d +
    0.05 * counselor_visits     +
    0.5  * high_risk            +
    np.random.uniform(-0.5, 0.5, n)
)
lambda_rate = np.clip(lambda_rate, 0.1, 10)
support_requests = np.random.poisson(lambda_rate)

# ─────────────────────────────────────────────
# CSV BANAO
# ─────────────────────────────────────────────

df = pd.DataFrame({
    "student_id"                : student_ids,
    "week_no"                   : week_no,
    "attendance_percentage"     : np.round(attendance_pct, 1),
    "missed_classes_14d"        : missed_classes_14d,
    "lms_logins_7d"             : lms_logins_7d,
    "study_hours_per_week"      : np.round(study_hours, 1),
    "previous_exam_score"       : np.round(previous_exam_score, 1),
    "assignments_completed_pct" : np.round(assignments_completed_pct, 1),
    "late_submissions_30d"      : late_submissions_30d,
    "fee_pending_amount"        : fee_pending,
    "counselor_visits_30d"      : counselor_visits,
    "scholarship_status"        : scholarship_status,
    "internet_access"           : internet_access,
    "advisor_note"              : advisor_notes,
    "support_requests_next_14d" : support_requests,
    "high_risk_intervention"    : high_risk,
    "dominant_issue_type"       : dominant_issue,
})

script_folder = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_folder, "data",
                        "student_support_sample.csv")
df.to_csv(csv_path, index=False)

print(f"CSV ban gayi! Total rows: {len(df)}")
print(f"\nTarget 1 — support_requests:")
print(f"  Mean : {support_requests.mean():.2f}")
print(f"  Max  : {support_requests.max()}")
print(f"\nTarget 2 — high_risk:")
print(f"  At Risk     : {high_risk.sum()}")
print(f"  Not At Risk : {(high_risk==0).sum()}")
print(f"\nTarget 3 — dominant_issue:")
for issue in issue_types:
    count = (dominant_issue == issue).sum()
    print(f"  {issue:<12}: {count}")