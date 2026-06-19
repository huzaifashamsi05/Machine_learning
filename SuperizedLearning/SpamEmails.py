# ============================================================
# Traditional Programming vs Machine Learning
# Topic: What is ML and why do we need it?
# ============================================================

# ─── TRADITIONAL PROGRAMMING ────────────────────────────────
# In traditional programming, WE write the logic manually.
# The computer just follows our instructions step by step.
# Problem: what if spam words change? We have to update code manually.


    

def is_spam_traditional(email):
    
    # we manually decided these words = spam
    # this list came from OUR brain, not from data
    spam_words = ["FREE", "WIN", "PRIZE", "CLAIM", "LOTTERY"]
    
    # check if any spam word exists in the email
    # .upper() converts email to uppercase so "free" and "FREE" both match    
    for word in spam_words: 
        if word in email.upper():   
            return "SPAM"       # found a spam word → mark as spam
    
    return "NOT SPAM"           # no spam word found → safe email


# test emails to check our function
emails = [
    "Win a FREE iPhone now!!",      # has FREE and WIN → should be SPAM
    "Team meeting at 3pm tomorrow", # normal email → should be NOT SPAM
    "Your invoice is attached",     # normal email → should be NOT SPAM
]

print("=== Traditional Programming ===")
for email in emails:
    result = is_spam_traditional(email)   
    print(f"  '{email}' → {result}")

# OUTPUT:
# 'Win a FREE iPhone now!!' → SPAM
# 'Team meeting at 3pm tomorrow' → NOT SPAM
# 'Your invoice is attached' → NOT SPAM


# ─── MACHINE LEARNING ───────────────────────────────────────
# In ML, WE don't write any rules.
# We give the model labeled examples and it figures out the pattern.
# Advantage: it can learn new spam patterns we never thought of.

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# step 1: our training data (labeled examples)
# these are emails we already know the answer for
training_emails = [
    "Win a FREE iPhone now",     # spam
    "Claim your PRIZE today",    # spam
    "You WON a lottery",         # spam
    "Team meeting at 3pm",       # not spam
    "Your invoice is attached",  # not spam
    "Can you review my PR",      # not spam
]

# labels: 1 = spam, 0 = not spam
# this is the "correct answer" for each email above
labels = [1, 1, 1, 0, 0, 0]

# step 2: convert text to numbers
# computers don't understand words, they understand numbers
# CountVectorizer counts how many times each word appears
# e.g. "Win a FREE iPhone" → {win:1, free:1, iphone:1, a:1}
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(training_emails)   # X = feature matrix

# step 3: train the model
# model looks at word counts + labels and learns the pattern
# Naive Bayes calculates: P(spam | word) for every word
model = MultinomialNB()
model.fit(X, labels)    # this is where learning actually happens

# step 4: predict on a brand new email the model has never seen
test_email = [" huzaifa are there"]

# we must transform using the SAME vectorizer used during training
# so word counts are mapped to the same columns
X_test = vectorizer.transform(test_email)

result = model.predict(X_test)[0]   # returns 1 or 0

print("\n=== Machine Learning ===")
print(f"  '{test_email[0]}' → {'SPAM' if result == 1 else 'NOT SPAM'}")
print("  (model figured this out itself — we never wrote any rules)")

# OUTPUT:
# 'FREE money WIN claim now' → SPAM
# (model figured this out itself — we never wrote any rules)