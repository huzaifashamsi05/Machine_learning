# Decision Summary

## Model Selection for Student Support System

### Part 1 — Poisson Regression
**Target:** support_requests_next_14d (count)
**Model:** PoissonRegressionCustom
**Why:** Count data is never negative. Linear
Regression can predict negative counts which
makes no sense. Poisson uses log link (exp)
ensuring non-negative output always.
**Results:** MAE=1.51, beats baseline MAE=2.14

### Part 2 — Perceptron
**Target:** high_risk_intervention (binary)
**Model:** PerceptronCustom
**Why:** Binary decision needed. Perceptron
gives hard YES/NO decision. Recall=0.88 means
88% of truly at-risk students were caught.
Missing a high-risk student is costly.
**Results:** F1=0.86, Recall=0.88, Accuracy=0.84

### Part 3 — Naive Bayes
**Target:** dominant_issue_type (5 classes)
**Model:** NaiveBayesCustom
**Why:** Advisor notes are text data. Naive
Bayes works with word counts and probabilities.
Laplace smoothing handles unseen words.
Log probabilities prevent underflow.
**Results:** Macro F1=1.00, all 5 classes perfect

## Conclusion
One model cannot solve all three problems.
Each target needs a different approach based
on the nature of the prediction task.