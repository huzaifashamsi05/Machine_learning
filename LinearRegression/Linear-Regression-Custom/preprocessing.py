# Scaling kyun zaroori hai?
# study_hours      →  0  to  50
# attendance       →  0  to  100
# practice_questions → 0  to  500

# Sab alag alag range pe hain — model confuse ho jaata hai۔ Scaling ke baad sab same range pe aa jaate hain۔ 

# Formula — Z-score:

# scaled = (value - mean) / std
# mean = average
# std  = standard deviation (spread)
 
 
# PREPROCESSING
# Do kaam:
# 1. Features scale karo (same range pe lao)
# 2. Data ko train aur test mein todo (80/20)

import numpy as np

# PART 1 — FEATURE SCALING
# Formula: scaled = (value - mean) / std

def scale_features(X_train, X_test):
    # sirf train data se mean aur std nikalo
    # test data ko mat dekho — cheating hogi
    mean = np.mean(X_train, axis=0)
    std  = np.std(X_train, axis=0)

    # dono ko scale karo — same mean aur std se
    X_train_scaled = (X_train - mean) / std
    X_test_scaled  = (X_test  - mean) / std

    return X_train_scaled, X_test_scaled, mean, std

# PART 2 — TRAIN/TEST SPLIT (80% train, 20% test)

def train_test_split(X, y, test_size = 0.2 ):
    # total rows _size=0.2kitni hain
    n = len(y)

    # rows ko shuffle karo — random order mein
    indices = np.arange(n)           # [0,1,2,3,...,299]
    np.random.shuffle(indices)       # random kar do

    # 80% train ke liye, 20% test ke liye
    split = int(n * (1 - test_size)) # 300 * 0.8 = 240

    train_idx = indices[:split]      # pehle 240
    test_idx  = indices[split:]      # baaki 60

    # split karo
    X_train = X[train_idx]
    X_test  = X[test_idx]
    y_train = y[train_idx]
    y_test  = y[test_idx]

    return X_train, X_test, y_train, y_test


print("preprocessing.py ready!")
