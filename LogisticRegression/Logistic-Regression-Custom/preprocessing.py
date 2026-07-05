    # ─────────────────────────────────────────────
# PREPROCESSING
# 3 kaam:
# 1. Yes/No → 1/0 convert karo
# 2. Features scale karo
# 3. Train/Validation/Test split (70/15/15)
# ─────────────────────────────────────────────

import numpy as np

# ─────────────────────────────────────────────
# PART 1 — YES/NO → 1/0
# ─────────────────────────────────────────────

def encode_categorical(df, columns):
    # har Yes/No column ko 1/0 mein badlo
    for col in columns:
        df[col] = df[col].map({"Yes": 1, "No": 0})
        # agar pehle se 0/1 hai toh koi change nahi
    return df

# ─────────────────────────────────────────────
# PART 2 — FEATURE SCALING (Z-score)
# Formula: scaled = (value - mean) / std
# ─────────────────────────────────────────────

def scale_features(X_train, X_val, X_test):
    # sirf train se mean aur std nikalo
    mean = np.mean(X_train, axis=0)
    std  = np.std(X_train, axis=0)

    # std 0 na ho — division error aayega
    std[std == 0] = 1

    # scale karo
    X_train_s = (X_train - mean) / std
    X_val_s   = (X_val   - mean) / std
    X_test_s  = (X_test  - mean) / std

    return X_train_s, X_val_s, X_test_s, mean, std

# ─────────────────────────────────────────────
# PART 3 — TRAIN/VALIDATION/TEST SPLIT
# 70% train, 15% validation, 15% test
# ─────────────────────────────────────────────

def train_val_test_split(X, y, val_size=0.15, test_size=0.15):
    n = len(y)

    # rows shuffle karo
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)

    # sizes calculate karo
    test_count  = int(n * test_size)   # 300 * 0.15 = 45
    val_count   = int(n * val_size)    # 300 * 0.15 = 45
    train_count = n - test_count - val_count  # 300 - 45 - 45 = 210

    # indices alag karo
    train_idx = indices[:train_count]
    val_idx   = indices[train_count:train_count + val_count]
    test_idx  = indices[train_count + val_count:]

    # split karo
    X_train = X[train_idx]
    X_val   = X[val_idx]
    X_test  = X[test_idx]
    y_train = y[train_idx]
    y_val   = y[val_idx]
    y_test  = y[test_idx]

    return X_train, X_val, X_test, y_train, y_val, y_test


print("preprocessing.py ready!")