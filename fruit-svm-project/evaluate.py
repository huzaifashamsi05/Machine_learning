"""
evaluate.py - Evaluation & Single-Image Prediction
====================================================

Two modes:
    1. Full evaluation: Evaluate trained model on test set
       python evaluate.py --test
    
    2. Single image: Predict class and confidence for one image
       python evaluate.py --image path/to/fruit.jpg

Metrics computed:
    - Overall accuracy
    - Per-class precision, recall, F1-score
    - Confusion matrix
    - Confidence calibration plot
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Local imports
from features import extract_features_single, FeatureScaler
from svm import load_model, CalibratedMultiClassSVM


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.npz")
SCALER_PATH = os.path.join(MODELS_DIR, "feature_scaler.npz")
TARGET_SIZE = (64, 64)


# ============================================================
# Single Image Prediction
# ============================================================

def predict_single_image(image_path, model, scaler, class_names, calibrator=None):
    """
    Predict class and confidence for a single fruit image.
    
    Pipeline:
        1. Load image from path
        2. Resize to 64x64
        3. Extract features (color + shape + HOG)
        4. Normalize features using training scaler
        5. Predict with SVM
        6. Get confidence via Platt scaling (if available)
    
    Args:
        image_path: path to the image file
        model: trained MultiClassSVM
        scaler: fitted FeatureScaler
        class_names: list of class name strings
        calibrator: optional CalibratedMultiClassSVM for confidence
    
    Returns:
        predicted_class: string (class name)
        confidence: float (0-100%)
        all_probs: dict mapping class names to probabilities
    """
    # Step 1: Load image
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)
    
    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    
    # Step 2: Resize
    img_resized = img.resize(TARGET_SIZE, Image.BILINEAR)
    img_array = np.array(img_resized, dtype=np.uint8)
    
    # Step 3: Extract features
    features = extract_features_single(img_array)
    features = features.reshape(1, -1)  # (1, D)
    
    # Step 4: Normalize
    features_norm = scaler.transform(features)
    
    # Step 5 & 6: Predict with confidence
    if calibrator is not None and calibrator.is_calibrated:
        preds, confidences, probs = calibrator.predict_with_confidence(features_norm)
        predicted_idx = preds[0]
        confidence = confidences[0]
        all_probs = {class_names[i]: probs[0, i] * 100 for i in range(len(class_names))}
    else:
        predicted_idx = model.predict(features_norm)[0]
        scores = model.decision_function(features_norm)[0]
        # Use softmax on raw scores as fallback
        exp_scores = np.exp(scores - scores.max())
        probs = exp_scores / exp_scores.sum()
        confidence = probs[predicted_idx] * 100
        all_probs = {class_names[i]: probs[i] * 100 for i in range(len(class_names))}
    
    predicted_class = class_names[predicted_idx]
    
    return predicted_class, confidence, all_probs


def display_prediction(image_path, predicted_class, confidence, all_probs):
    """Display prediction results in a nice format."""
    print("\n" + "=" * 50)
    print("  FRUIT CLASSIFICATION RESULT")
    print("=" * 50)
    print(f"  Image: {os.path.basename(image_path)}")
    print(f"  Predicted: {predicted_class}")
    print(f"  Confidence: {confidence:.1f}%")
    print("-" * 50)
    print("  All class probabilities:")
    
    # Sort by probability (descending)
    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
    for cls_name, prob in sorted_probs:
        bar = "#" * int(prob / 2)  # Simple bar chart
        marker = " <<" if cls_name == predicted_class else ""
        print(f"    {cls_name:<20} {prob:>6.1f}% {bar}{marker}")
    
    print("=" * 50)


# ============================================================
# Full Evaluation on Test Set
# ============================================================

def compute_confusion_matrix(y_true, y_pred, n_classes):
    """
    Compute confusion matrix from scratch.
    
    Matrix layout:
        rows = true class
        cols = predicted class
        C[i,j] = number of samples with true class i predicted as class j
    
    The diagonal represents correct predictions.
    Off-diagonal entries show misclassification patterns.
    """
    cm = np.zeros((n_classes, n_classes), dtype=np.int32)
    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1
    return cm


def compute_metrics(y_true, y_pred, n_classes, class_names):
    """
    Compute precision, recall, F1-score for each class.
    
    For class k:
        Precision = TP / (TP + FP)
            "Of all samples predicted as k, how many are truly k?"
        
        Recall = TP / (TP + FN)
            "Of all truly-k samples, how many did we predict correctly?"
        
        F1 = 2 * Precision * Recall / (Precision + Recall)
            Harmonic mean of precision and recall.
            Ranges from 0 (worst) to 1 (best).
    """
    cm = compute_confusion_matrix(y_true, y_pred, n_classes)
    
    metrics = {}
    for k in range(n_classes):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall) 
              if (precision + recall) > 0 else 0)
        
        metrics[class_names[k]] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': int(cm[k, :].sum())
        }
    
    return metrics, cm


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot a nice confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Normalize
    cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
    
    # Plot
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    ax.set_title('Confusion Matrix (Normalized)', fontsize=14)
    
    # Labels
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)
    ax.set_ylabel('True Class')
    ax.set_xlabel('Predicted Class')
    
    # Add text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text_color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{cm[i,j]}\n({cm_norm[i,j]:.0%})',
                   ha='center', va='center', color=text_color, fontsize=9)
    
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved confusion matrix: {save_path}")


def plot_calibration(y_true, y_pred, confidences, n_bins, save_path):
    """
    Plot calibration curve (reliability diagram).
    
    A well-calibrated model has:
        "If I predict 80% confidence, I should be correct 80% of the time"
    
    Plot: predicted confidence (x-axis) vs actual accuracy (y-axis)
    Perfect calibration = diagonal line.
    
    Below diagonal = overconfident
    Above diagonal = underconfident
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bin predictions by confidence
    correct = (y_pred == y_true).astype(np.float64)
    
    bin_edges = np.linspace(0, 100, n_bins + 1)
    bin_centers = []
    bin_accs = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i+1])
        if mask.sum() > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)
            bin_accs.append(correct[mask].mean() * 100)
            bin_counts.append(mask.sum())
    
    # Plot 1: Calibration curve
    ax1.plot([0, 100], [0, 100], 'k--', label='Perfect calibration', alpha=0.5)
    ax1.plot(bin_centers, bin_accs, 'o-', color='#2196F3', 
            linewidth=2, markersize=8, label='Model')
    ax1.set_xlabel('Mean Predicted Confidence (%)')
    ax1.set_ylabel('Actual Accuracy (%)')
    ax1.set_title('Calibration Curve (Reliability Diagram)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    
    # Plot 2: Confidence histogram
    ax2.hist(confidences, bins=20, color='#4CAF50', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Predicted Confidence (%)')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('Confidence Distribution')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved calibration plot: {save_path}")


def full_evaluation(model, scaler, class_names, calibrator=None):
    """
    Run full evaluation on the test set.
    
    Reports:
        - Overall accuracy
        - Per-class precision, recall, F1
        - Confusion matrix
        - Calibration plot
    """
    print("\n" + "=" * 60)
    print("  FULL TEST SET EVALUATION")
    print("=" * 60)
    
    # Load test data
    test_data = np.load(os.path.join(PROCESSED_DIR, "test.npz"))
    test_images = test_data['images']
    y_test = test_data['labels']
    
    print(f"\n  Test samples: {len(y_test)}")
    
    # Extract features
    print("  Extracting features...")
    from features import extract_features_batch
    X_test = extract_features_batch(test_images, verbose=False)
    X_test = scaler.transform(X_test)
    
    # Predict
    print("  Running predictions...")
    n_classes = len(class_names)
    
    if calibrator is not None and calibrator.is_calibrated:
        y_pred, confidences, probs = calibrator.predict_with_confidence(X_test)
    else:
        y_pred = model.predict(X_test)
        scores = model.decision_function(X_test)
        exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        confidences = np.max(probs, axis=1) * 100
    
    # Overall accuracy
    overall_acc = np.mean(y_pred == y_test)
    print(f"\n  Overall Test Accuracy: {overall_acc:.4f} ({overall_acc*100:.1f}%)")
    
    # Per-class metrics
    metrics, cm = compute_metrics(y_test, y_pred, n_classes, class_names)
    
    print(f"\n  {'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("  " + "-" * 62)
    
    for cls_name, m in metrics.items():
        print(f"  {cls_name:<20} {m['precision']:>10.3f} {m['recall']:>10.3f} "
              f"{m['f1']:>10.3f} {m['support']:>10}")
    
    print("  " + "-" * 62)
    
    # Macro averages
    avg_precision = np.mean([m['precision'] for m in metrics.values()])
    avg_recall = np.mean([m['recall'] for m in metrics.values()])
    avg_f1 = np.mean([m['f1'] for m in metrics.values()])
    total_support = sum(m['support'] for m in metrics.values())
    
    print(f"  {'Macro Avg':<20} {avg_precision:>10.3f} {avg_recall:>10.3f} "
          f"{avg_f1:>10.3f} {total_support:>10}")
    
    # Target check
    target = 0.85
    if overall_acc >= target:
        print(f"\n  [PASS] Accuracy {overall_acc*100:.1f}% >= target {target*100}%!")
    else:
        print(f"\n  [INFO] Accuracy {overall_acc*100:.1f}% (target: {target*100}%)")
    
    # Plots
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Confusion matrix
    plot_confusion_matrix(
        cm, class_names, 
        os.path.join(FIGURES_DIR, 'confusion_matrix.png')
    )
    
    # Calibration plot
    plot_calibration(
        y_test, y_pred, confidences, n_bins=10,
        save_path=os.path.join(FIGURES_DIR, 'calibration.png')
    )
    
    print(f"\n  Figures saved to: {FIGURES_DIR}")
    
    return overall_acc, metrics


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Fruit SVM Classifier - Evaluation & Prediction'
    )
    parser.add_argument(
        '--image', type=str, 
        help='Path to a fruit image to classify'
    )
    parser.add_argument(
        '--test', action='store_true',
        help='Run full evaluation on test set'
    )
    
    args = parser.parse_args()
    
    if not args.image and not args.test:
        print("Usage:")
        print("  python evaluate.py --image path/to/fruit.jpg")
        print("  python evaluate.py --test")
        sys.exit(1)
    
    # Load model
    print("[*] Loading trained model...")
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("  Run train.py first!")
        sys.exit(1)
    
    model, scaler, class_names = load_model(MODEL_PATH)
    calibrator = getattr(model, '_calibrator', None)
    
    print(f"  Model loaded: {len(model.classifiers)} classifiers")
    print(f"  Classes: {class_names}")
    
    if args.image:
        # Single image prediction
        predicted_class, confidence, all_probs = predict_single_image(
            args.image, model, scaler, class_names, calibrator
        )
        display_prediction(args.image, predicted_class, confidence, all_probs)
    
    if args.test:
        # Full test set evaluation
        accuracy, metrics = full_evaluation(
            model, scaler, class_names, calibrator
        )


if __name__ == "__main__":
    main()
