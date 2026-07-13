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
# TARGET 3 — dominant_issue_type
# ─────────────────────────────────────────────

issue_types = ["Academic", "Attendance", "Financial",
               "Personal", "Stable"]

academic_notes = [
    "struggling with calculus exam preparation",
    "quiz grades dropping needs tutoring urgently",
    "coursework submission poor understanding subject",
    "homework incomplete lecture comprehension weak",
    "midterm failed needs extra academic coaching",
]

attendance_notes = [
    "absent skipping morning sessions repeatedly",
    "transport late irregular punctuality problem",
    "missed practicals labs consistently absent",
    "truancy pattern observed skipping school",
    "below minimum threshold attendance warning issued",
]

financial_notes = [
    "fee overdue dues outstanding payment delayed",
    "scholarship revoked pending semester charges",
    "defaulter blocked portal unpaid outstanding",
    "cannot afford textbooks payment overdue",
    "financial hardship dues unpaid charges pending",
]

personal_notes = [
    "grief bereavement family trauma counseling",
    "anxiety depression mental health therapy",
    "domestic crisis hospitalized chronic illness",
    "emotional distress family emergency referred",
    "psychological support personal tragedy intervention",
]

stable_notes = [
    "excellent consistent progress no worries",
    "performing great assignments submitted timely",
    "no intervention needed doing wonderfully",
    "active engaged thriving academically stable",
    "on track zero concerns outstanding performer",
]

all_notes = [academic_notes, attendance_notes,
             financial_notes, personal_notes, stable_notes]

dominant_issue = []
advisor_notes  = []

np.random.seed(99)
for i in range(n):
    # balanced random assignment
    rand = np.random.random()
    if rand < 0.20:
        issue = "Academic"
    elif rand < 0.40:
        issue = "Attendance"
    elif rand < 0.60:
        issue = "Financial"
    elif rand < 0.80:
        issue = "Personal"
    else:
        issue = "Stable"

    dominant_issue.append(issue)
    idx  = issue_types.index(issue)
    note = np.random.choice(all_notes[idx])
    advisor_notes.append(note)

dominant_issue = np.array(dominant_issue)
advisor_notes  = np.array(advisor_notes)

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