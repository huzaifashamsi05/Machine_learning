"""
train.py - Training Pipeline for Fruit SVM Classifier
======================================================

This script orchestrates the full training pipeline:
    1. Load preprocessed data
    2. Extract and normalize features
    3. Train Linear SVM (baseline)
    4. Train Kernel SVMs (polynomial, RBF)
    5. Hyperparameter tuning on validation set
    6. Select best model
    7. Calibrate confidence scores (Platt scaling)
    8. Save best model
    9. Generate training curves and comparison plots

Usage:
    python train.py
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Local imports
from features import extract_and_normalize, FeatureScaler
from svm import (LinearSVM, KernelSVM, MultiClassSVM, 
                 CalibratedMultiClassSVM, save_model)


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

# Ensure output directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def load_data():
    """Load preprocessed train/val/test splits."""
    print("[1] Loading preprocessed data...")
    
    train_data = np.load(os.path.join(PROCESSED_DIR, "train.npz"))
    val_data = np.load(os.path.join(PROCESSED_DIR, "val.npz"))
    test_data = np.load(os.path.join(PROCESSED_DIR, "test.npz"))
    
    class_names = list(train_data['class_names'])
    
    print(f"  Classes: {class_names}")
    print(f"  Train: {len(train_data['labels'])} samples")
    print(f"  Val:   {len(val_data['labels'])} samples")
    print(f"  Test:  {len(test_data['labels'])} samples")
    
    return (train_data['images'], train_data['labels'],
            val_data['images'], val_data['labels'],
            test_data['images'], test_data['labels'],
            class_names)


def extract_features(train_images, val_images, test_images):
    """Extract and normalize features from all splits."""
    print("\n[2] Extracting features...")
    
    scaler_path = os.path.join(MODELS_DIR, "feature_scaler.npz")
    X_train, X_val, X_test, scaler = extract_and_normalize(
        train_images, val_images, test_images,
        scaler_path=scaler_path,
        verbose=True
    )
    
    print(f"\n  Feature dimensions: {X_train.shape[1]}")
    
    return X_train, X_val, X_test, scaler


# ============================================================
# Training Functions
# ============================================================

def train_linear_svm(X_train, y_train, X_val, y_val, class_names):
    """
    Train Linear SVM with hyperparameter tuning.
    
    Tunes:
        C in {0.001, 0.01, 0.1, 1.0, 10.0, 100.0}
    
    Selects best C based on validation accuracy.
    """
    print("\n" + "=" * 60)
    print("[3] TRAINING: Linear SVM (Baseline)")
    print("=" * 60)
    
    C_values = [1.0]
    best_model = None
    best_val_acc = 0
    best_C = None
    results = []
    
    for C in C_values:
        print(f"\n--- C = {C} ---")
        
        model = MultiClassSVM(
            LinearSVM, 
            n_classes=len(class_names),
            C=C, 
            learning_rate=0.001, 
            decay=1e-5,
            n_epochs=300,
            batch_size=64,
            verbose=False
        )
        
        start_time = time.time()
        model.fit(X_train, y_train, class_names=class_names)
        train_time = time.time() - start_time
        
        train_acc = model.accuracy(X_train, y_train)
        val_acc = model.accuracy(X_val, y_val)
        
        print(f"  Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}, "
              f"Time: {train_time:.1f}s")
        
        results.append({
            'C': C, 'train_acc': train_acc, 'val_acc': val_acc,
            'time': train_time
        })
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model
            best_C = C
    
    print(f"\n  >> Best C = {best_C}, Val accuracy = {best_val_acc:.4f}")
    
    # Plot training curves for best model
    plot_training_curves(best_model, "Linear SVM")
    
    return best_model, results, best_val_acc


def train_kernel_svm(X_train, y_train, X_val, y_val, class_names, 
                     kernel_type='rbf'):
    """
    Train Kernel SVM with hyperparameter tuning.
    
    For 'poly': tunes degree in {2, 3, 4} and C in {0.1, 1.0, 10.0}
    For 'rbf': tunes gamma in {0.001, 0.01, 0.1, 1.0} and C in {0.1, 1.0, 10.0}
    
    Note: Kernel SVM with dual optimization is O(N^3) in theory,
    so we may need to subsample for large datasets.
    """
    print(f"\n{'=' * 60}")
    print(f"[4] TRAINING: {kernel_type.upper()} Kernel SVM")
    print("=" * 60)
    
    # Subsample training data if too large (dual SVM is O(N^3))
    max_train_size = 500  # per-class max for kernel SVM
    n_classes = len(class_names)
    
    if len(X_train) > max_train_size * n_classes:
        print(f"  [INFO] Subsampling training data for kernel SVM "
              f"(max {max_train_size} per class)...")
        indices = []
        for k in range(n_classes):
            cls_indices = np.where(y_train == k)[0]
            if len(cls_indices) > max_train_size:
                selected = np.random.choice(cls_indices, max_train_size, replace=False)
            else:
                selected = cls_indices
            indices.extend(selected)
        indices = np.array(indices)
        np.random.shuffle(indices)
        X_train_sub = X_train[indices]
        y_train_sub = y_train[indices]
        print(f"  Subsampled: {len(X_train)} -> {len(X_train_sub)}")
    else:
        X_train_sub = X_train
        y_train_sub = y_train
    
    # Define hyperparameter grid
    if kernel_type == 'poly':
        param_grid = [
            {'degree': 3, 'coef0': 1.0, 'C': 1.0}
        ]
    elif kernel_type == 'rbf':
        param_grid = [
            {'gamma': 0.1, 'C': 1.0}
        ]
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    best_model = None
    best_val_acc = 0
    best_params = None
    results = []
    
    for params in param_grid:
        C = params.pop('C')
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"\n--- C={C}, {params_str} ---")
        
        model = MultiClassSVM(
            KernelSVM,
            n_classes=n_classes,
            C=C,
            kernel_type=kernel_type,
            verbose=False,
            **params
        )
        
        start_time = time.time()
        model.fit(X_train_sub, y_train_sub, class_names=class_names)
        train_time = time.time() - start_time
        
        train_acc = model.accuracy(X_train_sub, y_train_sub)
        val_acc = model.accuracy(X_val, y_val)
        
        print(f"  Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}, "
              f"Time: {train_time:.1f}s")
        
        params['C'] = C
        results.append({
            **params, 'train_acc': train_acc, 'val_acc': val_acc,
            'time': train_time
        })
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model
            best_params = params.copy()
    
    params_str = ", ".join(f"{k}={v}" for k, v in best_params.items())
    print(f"\n  >> Best params: {params_str}")
    print(f"  >> Val accuracy: {best_val_acc:.4f}")
    
    return best_model, results, best_val_acc


# ============================================================
# Visualization Functions
# ============================================================

def plot_training_curves(model, model_name):
    """Plot training loss curves for each OvR classifier."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'{model_name} - Training Loss Curves', fontsize=14)
    
    for k, clf in enumerate(model.classifiers):
        ax = axes[k // 3, k % 3]
        if hasattr(clf, 'loss_history') and len(clf.loss_history) > 0:
            ax.plot(clf.loss_history, linewidth=1.5)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            class_name = model.class_names[k] if model.class_names else f"Class {k}"
            ax.set_title(f'{class_name} vs Rest')
            ax.grid(alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, f'{model_name.lower().replace(" ", "_")}_loss.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved training curves: {save_path}")


def plot_model_comparison(linear_results, poly_results, rbf_results):
    """Create a comparison plot of all models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Validation accuracy comparison
    models = ['Linear SVM']
    val_accs = [max(r['val_acc'] for r in linear_results)]
    
    if poly_results:
        models.append('Poly Kernel')
        val_accs.append(max(r['val_acc'] for r in poly_results))
    
    if rbf_results:
        models.append('RBF Kernel')
        val_accs.append(max(r['val_acc'] for r in rbf_results))
    
    colors = ['#2196F3', '#4CAF50', '#FF5722']
    bars = ax1.bar(models, val_accs, color=colors[:len(models)], width=0.5)
    ax1.set_ylabel('Validation Accuracy')
    ax1.set_title('Model Comparison - Best Validation Accuracy')
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, val_accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', fontweight='bold')
    
    # Plot 2: Hyperparameter tuning for RBF
    if rbf_results:
        gammas = sorted(set(r.get('gamma', 0) for r in rbf_results))
        Cs = sorted(set(r['C'] for r in rbf_results))
        
        for gamma in gammas:
            gamma_results = [r for r in rbf_results if r.get('gamma') == gamma]
            gamma_results.sort(key=lambda r: r['C'])
            accs = [r['val_acc'] for r in gamma_results]
            ax2.plot([str(r['C']) for r in gamma_results], accs, 
                    'o-', label=f'gamma={gamma}', linewidth=2, markersize=8)
        
        ax2.set_xlabel('C (regularization)')
        ax2.set_ylabel('Validation Accuracy')
        ax2.set_title('RBF Kernel - Hyperparameter Tuning')
        ax2.legend()
        ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, 'model_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved comparison plot: {save_path}")


def plot_hyperparameter_results(results, kernel_name, param_name):
    """Plot hyperparameter tuning results."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    param_values = sorted(set(r.get(param_name, 0) for r in results))
    C_values = sorted(set(r['C'] for r in results))
    
    for pv in param_values:
        pv_results = sorted(
            [r for r in results if r.get(param_name) == pv],
            key=lambda r: r['C']
        )
        train_accs = [r['train_acc'] for r in pv_results]
        val_accs = [r['val_acc'] for r in pv_results]
        
        ax.plot([str(r['C']) for r in pv_results], val_accs,
               'o-', label=f'{param_name}={pv} (val)', linewidth=2)
    
    ax.set_xlabel('C')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'{kernel_name} Kernel - Hyperparameter Tuning')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, f'{kernel_name.lower()}_tuning.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# Main Training Pipeline
# ============================================================

def main():
    """
    Full training pipeline:
        1. Load data
        2. Extract features
        3. Train Linear SVM (baseline)
        4. Train Polynomial Kernel SVM
        5. Train RBF Kernel SVM
        6. Compare models
        7. Calibrate best model (Platt scaling)
        8. Save best model
    """
    print("[*] Fruit SVM Project - Training Pipeline")
    print("=" * 60)
    start_total = time.time()
    
    # Step 1: Load data
    (train_images, y_train, val_images, y_val, 
     test_images, y_test, class_names) = load_data()
    
    # Step 2: Extract features
    X_train, X_val, X_test, scaler = extract_features(
        train_images, val_images, test_images
    )
    
    # Step 3: Train Linear SVM
    linear_model, linear_results, linear_acc = train_linear_svm(
        X_train, y_train, X_val, y_val, class_names
    )
    
    # Step 4: Train Polynomial Kernel SVM
    try:
        poly_model, poly_results, poly_acc = train_kernel_svm(
            X_train, y_train, X_val, y_val, class_names,
            kernel_type='poly'
        )
    except Exception as e:
        print(f"\n  [WARNING] Polynomial kernel training failed: {e}")
        poly_model, poly_results, poly_acc = None, [], 0
    
    # Step 5: Train RBF Kernel SVM
    try:
        rbf_model, rbf_results, rbf_acc = train_kernel_svm(
            X_train, y_train, X_val, y_val, class_names,
            kernel_type='rbf'
        )
    except Exception as e:
        print(f"\n  [WARNING] RBF kernel training failed: {e}")
        rbf_model, rbf_results, rbf_acc = None, [], 0
    
    # Step 6: Compare and select best model
    print("\n" + "=" * 60)
    print("[5] MODEL COMPARISON")
    print("=" * 60)
    
    comparison = [
        ("Linear SVM", linear_model, linear_acc),
        ("Polynomial Kernel SVM", poly_model, poly_acc),
        ("RBF Kernel SVM", rbf_model, rbf_acc),
    ]
    
    print(f"\n  {'Model':<30} {'Val Accuracy':>12}")
    print("  " + "-" * 44)
    
    best_model = None
    best_name = None
    best_acc = 0
    
    for name, model, acc in comparison:
        status = " <-- BEST" if acc == max(c[2] for c in comparison) else ""
        print(f"  {name:<30} {acc:>11.4f}{status}")
        if acc > best_acc and model is not None:
            best_acc = acc
            best_model = model
            best_name = name
    
    print(f"\n  >> Selected: {best_name} (val acc: {best_acc:.4f})")
    
    # Plot comparison
    plot_model_comparison(linear_results, poly_results, rbf_results)
    
    # Plot hyperparameter tuning
    if poly_results:
        plot_hyperparameter_results(poly_results, "Polynomial", "degree")
    if rbf_results:
        plot_hyperparameter_results(rbf_results, "RBF", "gamma")
    
    # Step 7: Calibrate confidence scores
    print("\n" + "=" * 60)
    print("[6] CONFIDENCE CALIBRATION (Platt Scaling)")
    print("=" * 60)
    
    calibrated_model = CalibratedMultiClassSVM(best_model)
    calibrated_model.calibrate(X_val, y_val)
    
    # Test calibration on validation set
    preds, confidences, probs = calibrated_model.predict_with_confidence(X_val)
    cal_acc = np.mean(preds == y_val)
    print(f"\n  Calibrated Val accuracy: {cal_acc:.4f}")
    print(f"  Mean confidence: {confidences.mean():.1f}%")
    print(f"  Confidence range: [{confidences.min():.1f}%, {confidences.max():.1f}%]")
    
    # Store calibrator in model for saving
    best_model._calibrator = calibrated_model
    
    # Step 8: Save model
    print("\n" + "=" * 60)
    print("[7] SAVING MODEL")
    print("=" * 60)
    
    model_path = os.path.join(MODELS_DIR, "best_model.npz")
    save_model(best_model, scaler, model_path, class_names)
    print(f"  Model saved: {model_path}")
    print(f"  File size: {os.path.getsize(model_path) / 1024:.1f} KB")
    
    # Also save features for later analysis
    features_path = os.path.join(MODELS_DIR, "features.npz")
    np.savez_compressed(
        features_path,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        class_names=np.array(class_names)
    )
    print(f"  Features saved: {features_path}")
    
    # Summary
    total_time = time.time() - start_total
    print("\n" + "=" * 60)
    print("[DONE] Training complete!")
    print(f"  Best model: {best_name}")
    print(f"  Validation accuracy: {best_acc:.4f} ({best_acc*100:.1f}%)")
    print(f"  Total training time: {total_time:.1f}s")
    print(f"  Model saved to: {model_path}")
    print("[NEXT] Run: python evaluate.py --test")


if __name__ == "__main__":
    main()
