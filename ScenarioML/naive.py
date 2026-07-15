import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ─────────────────────────────────────────────────────────────
# DATASET
# Email spam classification using Naive Bayes + Bag-of-Words
# ─────────────────────────────────────────────────────────────

emails = [
    "Win a free iPhone now click here",
    "Meeting scheduled for tomorrow at 10am",
    "Congratulations you won a lottery prize",
    "Please review the attached quarterly report",
    "Claim your free money today limited offer",
    "Can we reschedule our call to next week",
    "You have been selected for a cash prize",
    "Here are the minutes from yesterday's meeting",
    "Urgent act now to win big cash rewards",
    "Invoice attached for last month's services",
    "Free gift card waiting for you click now",
    "Team lunch is scheduled for Friday afternoon",
    "Limited time offer win a free vacation now",
    "Please find attached the project timeline",
    "You are a winner claim your prize immediately",
    "Reminder submit your timesheet by end of day",
    "Get rich quick with this amazing investment offer",
    "Let's discuss the budget in tomorrow's meeting",
    "Free money no strings attached click here now",
    "The client requested an update on the project",
    "Exclusive deal just for you claim now for free",
    "Please confirm your availability for the interview",
    "Cash prize winner notification act immediately",
    "Attached is the presentation for our review",
    "Hot singles in your area click to win now",
    "Payroll has been processed for this month",
    "You won free tickets claim before they expire",
    "The server maintenance is scheduled for tonight",
    "Double your money instantly limited offer today",
    "Please sign and return the attached contract",
]

# 1 = spam, 0 = not spam (ham)
labels = np.array([
    1, 0, 1, 0, 1,
    0, 1, 0, 1, 0,
    1, 0, 1, 0, 1,
    0, 1, 0, 1, 0,
    1, 0, 1, 0, 1,
    0, 1, 0, 1, 0,
])

# ─────────────────────────────────────────────────────────────
# STEP 1 — SPLIT
# ─────────────────────────────────────────────────────────────

X_train_text, X_test_text, y_train, y_test = train_test_split(
    emails, labels, test_size=0.3, random_state=42, stratify=labels
)

print("=" * 54)
print("  NAIVE BAYES — Email Spam Classifier")
print("=" * 54)
print(f"\n  Total emails    : {len(emails)}")
print(f"  Training set    : {len(X_train_text)} emails")
print(f"  Test set        : {len(X_test_text)} emails")

# ─────────────────────────────────────────────────────────────
# STEP 2 — BAG-OF-WORDS
# Converts each email into a vector of word counts
# fit_transform on train, transform only on test (no data leakage)
# ─────────────────────────────────────────────────────────────

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train_text)
X_test  = vectorizer.transform(X_test_text)

print(f"\n  Vocabulary size : {len(vectorizer.vocabulary_)} unique words")
print(f"  Sample words    : {list(vectorizer.vocabulary_.keys())[:10]}")

# ─────────────────────────────────────────────────────────────
# STEP 3 — TRAIN
# MultinomialNB automatically applies Laplace Smoothing
# alpha=1.0 is the default smoothing parameter (+1 to every count)
# ─────────────────────────────────────────────────────────────

model = MultinomialNB(alpha=1.0)
model.fit(X_train, y_train)

print("\n  Model trained successfully.")
print(f"  Laplace smoothing (alpha) : {model.alpha}")

# most spam-indicating words
feature_names_arr = vectorizer.get_feature_names_out()
log_prob_spam = model.feature_log_prob_[1]
top_spam_idx = np.argsort(log_prob_spam)[-8:][::-1]

print("\n  Top words associated with SPAM:")
for idx in top_spam_idx:
    print(f"    {feature_names_arr[idx]:<15} log-prob = {log_prob_spam[idx]:.3f}")

# ─────────────────────────────────────────────────────────────
# STEP 4 — EVALUATE
# ─────────────────────────────────────────────────────────────

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 54)
print("  EVALUATION ON TEST SET")
print("=" * 54)
print(f"\n  Accuracy : {accuracy * 100:.1f}%")

print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Spam (0)", "Spam (1)"]))

cm = confusion_matrix(y_test, y_pred)
print("  Confusion Matrix:")
print(f"  {'':22} Predicted HAM   Predicted SPAM")
print(f"  {'Actual HAM':22} {cm[0][0]:^14} {cm[0][1]:^14}")
print(f"  {'Actual SPAM':22} {cm[1][0]:^14} {cm[1][1]:^14}")

print("\n  Predicted vs Actual:")
for i, email in enumerate(X_test_text):
    prob_spam = y_pred_proba[i][1]
    pred_label = "SPAM" if y_pred[i] == 1 else "HAM"
    actual_label = "SPAM" if y_test[i] == 1 else "HAM"
    match = "✓" if y_pred[i] == y_test[i] else "✗"
    short_email = email[:40] + "..." if len(email) > 40 else email
    print(f"  {match}  P(spam)={prob_spam:.2f}  Pred={pred_label:<5} Actual={actual_label:<5}  \"{short_email}\"")

# ─────────────────────────────────────────────────────────────
# STEP 5 — PREDICT NEW EMAILS
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 54)
print("  PREDICT NEW EMAILS")
print("=" * 54)

new_emails = [
    "Click here now to win a free cash prize instantly",
    "Please attend the team meeting scheduled for Monday",
    "You are pre-approved for a free credit card offer",
]

new_vectorized = vectorizer.transform(new_emails)
new_preds = model.predict(new_vectorized)
new_probas = model.predict_proba(new_vectorized)

for email, pred, proba in zip(new_emails, new_preds, new_probas):
    label = "🚨 SPAM" if pred == 1 else "✅ NOT SPAM"
    print(f"\n  Email: \"{email}\"")
    print(f"  P(spam) = {proba[1]:.1%}  →  {label}")

print("\n" + "=" * 54)
print("  Done.")
print("=" * 54)