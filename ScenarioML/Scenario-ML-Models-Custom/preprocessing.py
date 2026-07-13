# ─────────────────────────────────────────────
# PREPROCESSING
# 4 kaam:
# 1. Yes/No → 1/0
# 2. Missing values handle
# 3. Scale features
# 4. Train/Val/Test split (70/15/15)
# ─────────────────────────────────────────────

import numpy as np

# ─────────────────────────────────────────────
# PART 1 — YES/NO → 1/0
# ─────────────────────────────────────────────

def encode_categorical(df, columns):
    for col in columns:
        if df[col].dtype == object:
            df[col] = df[col].map({"Yes": 1, "No": 0})
            df[col] = df[col].fillna(0)
    return df

# ─────────────────────────────────────────────
# PART 2 — MISSING VALUES
# ─────────────────────────────────────────────

def handle_missing(df, numeric_cols):
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())
    return df

# ─────────────────────────────────────────────
# PART 3 — FEATURE SCALING
# ─────────────────────────────────────────────

def scale_features(X_train, X_val, X_test):
    mean = np.mean(X_train, axis=0)
    std  = np.std(X_train, axis=0)
    std[std == 0] = 1

    X_train_s = (X_train - mean) / std
    X_val_s   = (X_val   - mean) / std
    X_test_s  = (X_test  - mean) / std

    X_train_s = np.nan_to_num(X_train_s, nan=0.0)
    X_val_s   = np.nan_to_num(X_val_s,   nan=0.0)
    X_test_s  = np.nan_to_num(X_test_s,  nan=0.0)

    return X_train_s, X_val_s, X_test_s, mean, std

# ─────────────────────────────────────────────
# PART 4 — TRAIN/VAL/TEST SPLIT (70/15/15)
# ─────────────────────────────────────────────

def train_val_test_split(X, y, val_size=0.15,
                         test_size=0.15, seed=42):
    n = len(y)
    indices = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(indices)

    test_count  = int(n * test_size)
    val_count   = int(n * val_size)
    train_count = n - test_count - val_count

    train_idx = indices[:train_count]
    val_idx   = indices[train_count:train_count + val_count]
    test_idx  = indices[train_count + val_count:]

    return (X[train_idx], X[val_idx], X[test_idx],
            y[train_idx], y[val_idx], y[test_idx],
            test_idx)

# ─────────────────────────────────────────────
# TOKENIZER — Naive Bayes ke liye
# ─────────────────────────────────────────────

def tokenize(text):
    # lowercase karo
    text = text.lower()
    # punctuation hatao
    for char in ".,!?;:'\"-()[]{}":
        text = text.replace(char, " ")
    # words nikalo
    return text.split()


print("preprocessing.py ready!")
