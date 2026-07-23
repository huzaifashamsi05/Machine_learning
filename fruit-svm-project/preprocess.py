"""
preprocess.py - Image Preprocessing & Train/Val/Test Split
==========================================================

This script handles:
1. Loading raw images from data/raw/
2. Resizing all images to 64x64 pixels
3. Stratified splitting into 70/15/15 (train/val/test)
4. Saving as NumPy .npz archives
5. Generating a sample visualization grid

Mathematical Notes:
    - Resizing uses bilinear interpolation (PIL.Image.BILINEAR)
      which computes each new pixel as a weighted average of
      the 4 nearest pixels in the original image:
      
      f(x,y) = f(0,0)(1-x)(1-y) + f(1,0)x(1-y) + f(0,1)(1-x)y + f(1,1)xy
    
    - Stratified split ensures class proportions are maintained:
      For class k with N_k images:
        train_k = floor(0.70 * N_k)
        val_k   = floor(0.15 * N_k)
        test_k  = N_k - train_k - val_k  (remaining ~15%)

Why 64x64?
    - Small enough for fast feature extraction and SVM training
    - Large enough to preserve shape/texture details
    - Power of 2 = efficient for HOG grid subdivision (4x4, 8x8 cells)
"""

import os
import sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

TARGET_SIZE = (64, 64)  # Width x Height for PIL

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15  # = 1 - TRAIN - VAL

# Reproducibility
RANDOM_SEED = 42


def load_images_from_directory(raw_dir):
    """
    Load all images from the raw directory.
    
    Expected structure:
        raw_dir/
        |- Apple Red 1/
        |  |- img1.jpg
        |  |- img2.jpg
        |- Banana/
        |  |- ...
        
    Returns:
        images: list of PIL Image objects
        labels: list of integer labels (0, 1, 2, ...)
        class_names: sorted list of class name strings
    
    Note: We load as PIL images first, then resize and convert
    to numpy arrays to maintain quality during interpolation.
    """
    print("Loading images from:", raw_dir)
    
    # Get sorted class names for consistent label assignment
    class_names = sorted([
        d for d in os.listdir(raw_dir) 
        if os.path.isdir(os.path.join(raw_dir, d))
    ])
    
    if len(class_names) == 0:
        print("[ERROR] No class directories found in", raw_dir)
        print("  Run download_data.py first!")
        sys.exit(1)
    
    print(f"  Found {len(class_names)} classes: {class_names}")
    
    images = []
    labels = []
    
    for label_idx, cls_name in enumerate(class_names):
        cls_dir = os.path.join(raw_dir, cls_name)
        cls_images = []
        
        for img_file in sorted(os.listdir(cls_dir)):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(cls_dir, img_file)
                try:
                    img = Image.open(img_path).convert('RGB')
                    cls_images.append(img)
                    labels.append(label_idx)
                except Exception as e:
                    print(f"  [WARNING] Could not load {img_path}: {e}")
        
        images.extend(cls_images)
        print(f"  [{label_idx}] {cls_name}: {len(cls_images)} images loaded")
    
    return images, labels, class_names


def resize_images(images, target_size=TARGET_SIZE):
    """
    Resize all images to the target size using bilinear interpolation.
    
    Args:
        images: list of PIL Image objects
        target_size: (width, height) tuple
    
    Returns:
        numpy array of shape (N, height, width, 3) with dtype uint8
    
    Bilinear interpolation is chosen because:
        - Better quality than nearest-neighbor (no jagged edges)
        - Faster than bicubic (negligible quality difference at 64x64)
        - Preserves smooth color gradients important for histograms
    """
    print(f"\nResizing {len(images)} images to {target_size}...")
    
    resized = []
    for i, img in enumerate(images):
        # PIL.Image.resize uses (width, height) format
        img_resized = img.resize(target_size, Image.BILINEAR)
        # Convert to numpy array: (height, width, channels)
        img_array = np.array(img_resized, dtype=np.uint8)
        resized.append(img_array)
        
        if (i + 1) % 500 == 0:
            print(f"  Resized {i+1}/{len(images)} images...")
    
    result = np.array(resized, dtype=np.uint8)
    print(f"  Final array shape: {result.shape}")
    print(f"  Memory: {result.nbytes / 1024 / 1024:.1f} MB")
    
    return result


def stratified_split(images, labels, class_names, 
                     train_ratio=TRAIN_RATIO, 
                     val_ratio=VAL_RATIO, 
                     seed=RANDOM_SEED):
    """
    Split data into train/val/test sets with stratified sampling.
    
    Stratified sampling ensures that each split has the same proportion
    of each class as the full dataset. This is critical because:
        1. Prevents class imbalance in any split
        2. Ensures validation/test metrics are representative
        3. Prevents edge cases where a class might be missing from a split
    
    Algorithm:
        For each class k with N_k samples:
            1. Shuffle the N_k indices randomly (with seed)
            2. Take first floor(0.70 * N_k) for training
            3. Take next floor(0.15 * N_k) for validation
            4. Remaining go to test (~15%)
    
    Args:
        images: numpy array (N, H, W, C)
        labels: numpy array (N,)
        class_names: list of class name strings
        train_ratio: fraction for training (default 0.70)
        val_ratio: fraction for validation (default 0.15)
        seed: random seed for reproducibility
    
    Returns:
        dict with keys: 'train', 'val', 'test'
        Each value is a dict with 'images' and 'labels' arrays
    """
    print(f"\nSplitting data: {train_ratio:.0%} train / "
          f"{val_ratio:.0%} val / {1-train_ratio-val_ratio:.0%} test")
    
    labels = np.array(labels)
    rng = np.random.RandomState(seed)
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    n_classes = len(class_names)
    
    print(f"\n  {'Class':<20} {'Total':>6} {'Train':>6} {'Val':>6} {'Test':>6}")
    print("  " + "-" * 50)
    
    for cls_idx in range(n_classes):
        # Get all indices for this class
        cls_indices = np.where(labels == cls_idx)[0]
        n_cls = len(cls_indices)
        
        # Shuffle indices
        rng.shuffle(cls_indices)
        
        # Calculate split sizes
        n_train = int(np.floor(train_ratio * n_cls))
        n_val = int(np.floor(val_ratio * n_cls))
        # n_test = remaining
        
        # Split
        train_idx = cls_indices[:n_train]
        val_idx = cls_indices[n_train:n_train + n_val]
        test_idx = cls_indices[n_train + n_val:]
        
        train_indices.extend(train_idx)
        val_indices.extend(val_idx)
        test_indices.extend(test_idx)
        
        print(f"  {class_names[cls_idx]:<20} {n_cls:>6} "
              f"{len(train_idx):>6} {len(val_idx):>6} {len(test_idx):>6}")
    
    # Convert to numpy arrays
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)
    
    # Shuffle within each split (so training order is random)
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)
    
    splits = {
        'train': {
            'images': images[train_indices],
            'labels': labels[train_indices],
        },
        'val': {
            'images': images[val_indices],
            'labels': labels[val_indices],
        },
        'test': {
            'images': images[test_indices],
            'labels': labels[test_indices],
        },
    }
    
    print(f"\n  Total: {len(train_indices)} train + "
          f"{len(val_indices)} val + {len(test_indices)} test = "
          f"{len(train_indices) + len(val_indices) + len(test_indices)}")
    
    return splits


def save_splits(splits, class_names, output_dir):
    """
    Save the train/val/test splits as .npz files.
    
    File format (each .npz contains):
        - 'images': uint8 array of shape (N, 64, 64, 3)
        - 'labels': int array of shape (N,)
        - 'class_names': string array of class names
    
    .npz is NumPy's compressed archive format:
        - Smaller than uncompressed .npy
        - Faster to load than pickle or HDF5 for NumPy arrays
        - Platform-independent
    """
    print(f"\nSaving splits to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, data in splits.items():
        filepath = os.path.join(output_dir, f"{split_name}.npz")
        np.savez_compressed(
            filepath,
            images=data['images'],
            labels=data['labels'],
            class_names=np.array(class_names)
        )
        file_size = os.path.getsize(filepath) / 1024 / 1024
        print(f"  Saved {split_name}.npz: "
              f"{len(data['labels'])} samples, {file_size:.1f} MB")


def visualize_samples(splits, class_names, output_dir):
    """
    Create a sample grid visualization showing example images from each class.
    
    Grid layout: 6 rows (classes) x 5 columns (sample images)
    This serves as a sanity check that:
        1. Images were loaded correctly
        2. Resizing preserved visual quality
        3. Class labels are correctly assigned
    """
    print("\nGenerating sample visualization...")
    os.makedirs(output_dir, exist_ok=True)
    
    n_classes = len(class_names)
    n_samples = 5  # Images per class to show
    
    fig, axes = plt.subplots(n_classes, n_samples, 
                             figsize=(n_samples * 2, n_classes * 2))
    fig.suptitle('Sample Images per Class (64x64, Training Set)', 
                 fontsize=14, fontweight='bold')
    
    train_images = splits['train']['images']
    train_labels = splits['train']['labels']
    
    for cls_idx in range(n_classes):
        # Get indices for this class
        cls_mask = train_labels == cls_idx
        cls_images = train_images[cls_mask]
        
        for col in range(n_samples):
            ax = axes[cls_idx, col]
            if col < len(cls_images):
                ax.imshow(cls_images[col])
            ax.axis('off')
            
            # Add class name as row label
            if col == 0:
                ax.set_title(class_names[cls_idx], fontsize=10, 
                           fontweight='bold', loc='left')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'sample_grid.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def visualize_class_distribution(splits, class_names, output_dir):
    """
    Create a bar chart showing the number of images per class per split.
    
    This visualization confirms:
        1. Stratification worked (proportions are balanced)
        2. No class was accidentally dropped
        3. All classes meet the minimum image requirement
    """
    print("Generating class distribution plot...")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(len(class_names))
    width = 0.25
    
    for i, (split_name, data) in enumerate(splits.items()):
        counts = [np.sum(data['labels'] == cls_idx) 
                  for cls_idx in range(len(class_names))]
        bars = ax.bar(x + i * width, counts, width, 
                     label=f'{split_name} ({sum(counts)})')
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                   str(count), ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Fruit Class')
    ax.set_ylabel('Number of Images')
    ax.set_title('Class Distribution Across Train/Val/Test Splits')
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, rotation=30, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'class_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    """
    Main preprocessing pipeline:
        1. Load raw images from data/raw/
        2. Resize to 64x64
        3. Stratified split 70/15/15
        4. Save as .npz files
        5. Generate visualizations
    """
    print("[*] Fruit SVM Project - Data Preprocessing")
    print("=" * 60)
    
    # Check if raw data exists
    if not os.path.exists(RAW_DIR) or len(os.listdir(RAW_DIR)) == 0:
        print("[ERROR] No raw data found!")
        print("  Run download_data.py first.")
        sys.exit(1)
    
    # Step 1: Load images
    images, labels, class_names = load_images_from_directory(RAW_DIR)
    
    # Step 2: Resize
    images_resized = resize_images(images, TARGET_SIZE)
    
    # Step 3: Stratified split
    splits = stratified_split(images_resized, labels, class_names)
    
    # Step 4: Save
    save_splits(splits, class_names, PROCESSED_DIR)
    
    # Step 5: Visualize
    visualize_samples(splits, class_names, FIGURES_DIR)
    visualize_class_distribution(splits, class_names, FIGURES_DIR)
    
    print("\n" + "=" * 60)
    print("[DONE] Preprocessing complete!")
    print(f"  Processed data: {PROCESSED_DIR}")
    print(f"  Figures: {FIGURES_DIR}")
    print("[NEXT] Run: python train.py")


if __name__ == "__main__":
    main()
