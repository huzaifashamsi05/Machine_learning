# Linear Regression Custom

Predict a student's final exam score using study behavior features.

## Setup

Install required libraries:

pip install numpy pandas matplotlib

## Run
python main.py

## CSV Format

Your CSV must have these columns:

| Column | Range |
|---|---|
| study_hours_per_week | 0 to 50 |
| attendance_percentage | 0 to 100 |
| previous_test_score | 0 to 100 |
| assignments_completed | 0 to 20 |
| practice_questions_solved | 0 to 500 |
| sleep_hours_per_day | 0 to 12 |
| final_exam_score | 0 to 100 |

## Example Output
MAE  : 3.2

MSE  : 15.4

RMSE : 3.9

R²   : 0.89