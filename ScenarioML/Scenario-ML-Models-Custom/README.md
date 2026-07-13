# Scenario ML Models Custom

Student Support Early Warning and Triage System
using three custom ML models from scratch.

## Setup
pip install numpy pandas matplotlib

## Run
python main.py

## Dataset Format

CSV must have these columns:
student_id, week_no, attendance_percentage,
missed_classes_14d, lms_logins_7d,
study_hours_per_week, previous_exam_score,
assignments_completed_pct, late_submissions_30d,
fee_pending_amount, counselor_visits_30d,
scholarship_status, internet_access,
advisor_note, support_requests_next_14d,
high_risk_intervention, dominant_issue_type

## Models Used

| Part | Target | Model |
|------|--------|-------|
| 1 | support_requests_next_14d | Poisson Regression |
| 2 | high_risk_intervention | Perceptron |
| 3 | dominant_issue_type | Naive Bayes |

## Example Output
Poisson MAE    : 1.51
Perceptron F1  : 0.86
Naive Bayes F1 : 1.00