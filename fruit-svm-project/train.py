import os
import glob
import pickle
import numpy as np
from features import extract_features
from svm import LinearSVM, KernelSVM, OneVsRestSVM
from kernels import rbf_kernel, rbf_k

def load_data(data_dir="data/raw"):
    classes = os.listdir(data_dir)
    classes = [c for c in classes if os.path.isdir(os.path.join(data_dir, c))]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for i, c in enumerate(classes)}
    
    X = []
    y = []
    
    print("Extracting features from images...")
    for c in classes:
        print(f"Processing class: {c}")
        img_paths = glob.glob(os.path.join(data_dir, c, "*.jpg"))
        for img_path in img_paths:
            try:
                feats = extract_features(img_path)
                X.append(feats)
                y.append(class_to_idx[c])
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                
    X = np.array(X)
    y = np.array(y)
    
    # Standardize features
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0 # prevent division by zero
    X = (X - mean) / std
    
    return X, y, idx_to_class, mean, std

def split_data(X, y):
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    
    train_end = int(0.7 * n_samples)
    val_end = int(0.85 * n_samples)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    return (X[train_idx], y[train_idx]), (X[val_idx], y[val_idx]), (X[test_idx], y[test_idx])

def evaluate(model, X, y):
    preds = model.predict(X)
    return np.mean(preds == y)


if __name__ == "__main__":
    np.random.seed(42)
    
    X, y, idx_to_class, feature_mean, feature_std = load_data()
    print(f"Total dataset size: {X.shape[0]} images, {X.shape[1]} features.")
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X, y)
    print(f"Train/Val/Test split: {X_train.shape[0]}/{X_val.shape[0]}/{X_test.shape[0]}")
    
    print("\nTraining Linear SVM (One-vs-Rest)...")
    linear_svm = OneVsRestSVM(base_classifier_class=LinearSVM, C=1.0, learning_rate=0.01, epochs=100)
    linear_svm.fit(X_train, y_train)
    
    linear_val_acc = evaluate(linear_svm, X_val, y_val)
    print(f"Linear SVM Validation Accuracy: {linear_val_acc:.2%}")
    
    kernel_svm = OneVsRestSVM(base_classifier_class=KernelSVM, kernel_func=rbf_k, C=1.0, epochs=100)
    kernel_svm.fit(X_train, y_train)
    
    kernel_val_acc = evaluate(kernel_svm, X_val, y_val)
    print(f"Kernel SVM Validation Accuracy: {kernel_val_acc:.2%}")
    
    # Save best model
    best_model = linear_svm if linear_val_acc > kernel_val_acc else kernel_svm
    best_acc = evaluate(best_model, X_test, y_test)
    print(f"\nBest Model Test Accuracy: {best_acc:.2%}")
    
    model_data = {
        'model': best_model,
        'idx_to_class': idx_to_class,
        'feature_mean': feature_mean,
        'feature_std': feature_std
    }
    
    with open("model.pkl", "wb") as f:
        pickle.dump(model_data, f)
    print("Model saved to model.pkl")
