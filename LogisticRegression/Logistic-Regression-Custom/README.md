# Logistic Regression Custom

Predict whether a student is At Risk of failing based on academic behavior.

## Setup

Install required libraries:
pip install numpy pandas matplotlib

## Run
python main.py

## CSV Format

Your CSV must have these columns:

| Column | Range |
|---|---|
| student_id | text or integer |
| attendance_percentage | 0 to 100 |
| study_hours_per_week | 0 to 50 |
| previous_exam_score | 0 to 100 |
| assignments_completed | 0 to 20 |
| late_submissions | 0 to 20 |
| class_participation_score | 0 to 10 |
| sleep_hours_per_day | 0 to 12 |
| internet_access | Yes/No or 0/1 |
| part_time_job | Yes/No or 0/1 |
| at_risk | 0 or 1 |

## Example Output
Accuracy  : 0.8667
Precision : 0.9048
Recall    : 0.8261
F1 Score  : 0.8636
Log Loss  : 0.2576

## Threshold Selection

Tested thresholds: 0.3, 0.5, 0.7

Best threshold selected: 0.5

Reason: Threshold 0.5 gave highest F1 score
on validation data — best balance between
precision and recall.