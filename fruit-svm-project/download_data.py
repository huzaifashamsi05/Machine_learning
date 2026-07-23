"""
download_data.py — Fruits-360 Dataset Download & Organization
=============================================================

This script handles:
1. Downloading the Fruits-360 dataset from Kaggle
2. Extracting the archive
3. Selecting 6 specific fruit classes
4. Copying them to data/raw/ with a clean structure

Selected Classes (chosen for visual diversity):
    - Apple Red 1:  Red, round, smooth         → strong red hue
    - Banana:       Yellow, elongated, smooth   → unique shape
    - Orange:       Orange, round, textured     → distinct color
    - Strawberry:   Red, conical, seeded        → unique shape + texture
    - Lemon:        Yellow, oval, smooth        → yellow hue, oval shape
    - Kiwi:         Brown/green, oval, fuzzy    → unique texture + color

Why these 6?
    - Color diversity:   red, yellow, orange, brown/green
    - Shape diversity:   round, elongated, conical, oval
    - Texture diversity: smooth, fuzzy, seeded, dimpled
    → Maximizes separability with handcrafted features
"""

import os
import sys
import shutil
import zipfile


# ============================================================
# Configuration
# ============================================================

# Kaggle dataset identifier
KAGGLE_DATASET = "moltean/fruits"

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
DOWNLOAD_DIR = os.path.join(DATA_DIR, "_download")  # temporary download location

# The 6 fruit classes we want (must match folder names in Fruits-360)
SELECTED_CLASSES = [
    "Apple Red 1",
    "Banana",
    "Orange",
    "Strawberry",
    "Lemon",
    "Kiwi",
]

# Minimum images required per class
MIN_IMAGES_PER_CLASS = 80


def download_dataset():
    """
    Download Fruits-360 dataset from Kaggle using the Kaggle API.
    
    Prerequisites:
        - pip install kaggle
        - kaggle.json API token in ~/.kaggle/ (Windows: C:\\Users\\<user>\\.kaggle\\)
    
    The dataset is ~1GB compressed and extracts to ~2GB.
    Structure after extraction:
        fruits-360/
        ├── Training/
        │   ├── Apple Braeburn/
        │   ├── Apple Red 1/
        │   ├── Banana/
        │   └── ... (131+ classes)
        └── Test/
            ├── Apple Braeburn/
            ├── Apple Red 1/
            └── ...
    """
    print("=" * 60)
    print("STEP 1: Downloading Fruits-360 from Kaggle")
    print("=" * 60)
    
    # Create download directory
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    # Check if already downloaded
    zip_path = os.path.join(DOWNLOAD_DIR, "fruits.zip")
    if os.path.exists(zip_path):
        print(f"[INFO] Archive already exists: {zip_path}")
        print("[INFO] Skipping download. Delete the file to re-download.")
        return zip_path
    
    try:
        import kaggle
        kaggle.api.authenticate()
        print(f"[INFO] Downloading dataset: {KAGGLE_DATASET}")
        print("[INFO] This may take a few minutes (~1GB)...")
        kaggle.api.dataset_download_files(
            KAGGLE_DATASET,
            path=DOWNLOAD_DIR,
            unzip=False  # We'll handle extraction ourselves
        )
        print(f"[OK] Download complete!")
        
        # Kaggle names the file based on dataset slug
        downloaded_file = os.path.join(DOWNLOAD_DIR, "fruits.zip")
        if not os.path.exists(downloaded_file):
            # Sometimes kaggle names it differently
            for f in os.listdir(DOWNLOAD_DIR):
                if f.endswith(".zip"):
                    downloaded_file = os.path.join(DOWNLOAD_DIR, f)
                    break
        
        return downloaded_file
        
    except ImportError:
        print("[ERROR] kaggle package not installed!")
        print("  Fix: pip install kaggle")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        print("  Make sure kaggle.json is in C:\\Users\\HP\\.kaggle\\")
        sys.exit(1)


def extract_dataset(zip_path):
    """
    Extract the downloaded zip archive.
    
    The Fruits-360 zip contains a top-level folder structure:
        fruits-360_dataset/fruits-360/Training/...
        fruits-360_dataset/fruits-360/Test/...
    
    We extract everything to DOWNLOAD_DIR and then locate the
    Training/ and Test/ folders.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Extracting dataset")
    print("=" * 60)
    
    extract_dir = os.path.join(DOWNLOAD_DIR, "extracted")
    
    # Check if already extracted
    if os.path.exists(extract_dir):
        print(f"[INFO] Already extracted to: {extract_dir}")
        return extract_dir
    
    print(f"[INFO] Extracting {zip_path}...")
    print("[INFO] This may take a minute...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    print(f"[OK] Extraction complete!")
    return extract_dir


def find_dataset_folders(extract_dir):
    """
    Locate the Training/ and Test/ folders within the extracted archive.
    
    The folder structure can vary between dataset versions:
        - fruits-360/Training/
        - fruits-360_dataset/fruits-360/Training/
        - Training/  (directly)
    
    This function searches recursively to find them.
    """
    training_dir = None
    test_dir = None
    
    for root, dirs, files in os.walk(extract_dir):
        for d in dirs:
            if d == "Training" and training_dir is None:
                training_dir = os.path.join(root, d)
            elif d == "Test" and test_dir is None:
                test_dir = os.path.join(root, d)
        
        if training_dir and test_dir:
            break
    
    if not training_dir or not test_dir:
        print("[ERROR] Could not find Training/ and Test/ folders!")
        print(f"  Searched in: {extract_dir}")
        # List what we found
        for root, dirs, _ in os.walk(extract_dir):
            depth = root.replace(extract_dir, "").count(os.sep)
            if depth < 3:
                indent = " " * 2 * depth
                print(f"{indent}{os.path.basename(root)}/")
        sys.exit(1)
    
    print(f"[OK] Found Training: {training_dir}")
    print(f"[OK] Found Test: {test_dir}")
    return training_dir, test_dir


def select_and_copy_classes(training_dir, test_dir):
    """
    Select the 6 target fruit classes and copy images to data/raw/.
    
    Output structure:
        data/raw/
        ├── Apple Red 1/
        │   ├── img1.jpg
        │   ├── img2.jpg
        │   └── ...
        ├── Banana/
        │   └── ...
        └── ... (6 classes total)
    
    We combine Training + Test images into one folder per class,
    because we'll do our own 70/15/15 split in preprocess.py.
    
    Why combine?
        The original split is ~70/30. We want 70/15/15 with
        stratified random sampling, so we pool all images first.
    """
    print("\n" + "=" * 60)
    print("STEP 3: Selecting 6 fruit classes")
    print("=" * 60)
    
    # First, let's see what classes are available
    available_classes = sorted(os.listdir(training_dir))
    print(f"[INFO] Total classes available: {len(available_classes)}")
    
    # Verify our selected classes exist
    for cls_name in SELECTED_CLASSES:
        if cls_name not in available_classes:
            print(f"[ERROR] Class '{cls_name}' not found in dataset!")
            print(f"  Available classes containing similar names:")
            for ac in available_classes:
                if any(word.lower() in ac.lower() for word in cls_name.split()):
                    print(f"    - {ac}")
            sys.exit(1)
    
    # Clean the raw directory
    if os.path.exists(RAW_DIR):
        shutil.rmtree(RAW_DIR)
    os.makedirs(RAW_DIR, exist_ok=True)
    
    total_images = 0
    class_stats = {}
    
    for cls_name in SELECTED_CLASSES:
        cls_output_dir = os.path.join(RAW_DIR, cls_name)
        os.makedirs(cls_output_dir, exist_ok=True)
        
        img_count = 0
        
        # Copy from Training/
        train_cls_dir = os.path.join(training_dir, cls_name)
        if os.path.exists(train_cls_dir):
            for img_file in os.listdir(train_cls_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src = os.path.join(train_cls_dir, img_file)
                    dst = os.path.join(cls_output_dir, f"train_{img_file}")
                    shutil.copy2(src, dst)
                    img_count += 1
        
        # Copy from Test/
        test_cls_dir = os.path.join(test_dir, cls_name)
        if os.path.exists(test_cls_dir):
            for img_file in os.listdir(test_cls_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src = os.path.join(test_cls_dir, img_file)
                    dst = os.path.join(cls_output_dir, f"test_{img_file}")
                    shutil.copy2(src, dst)
                    img_count += 1
        
        class_stats[cls_name] = img_count
        total_images += img_count
        
        # Verify minimum image count
        status = "✓" if img_count >= MIN_IMAGES_PER_CLASS else "✗ BELOW MINIMUM!"
        print(f"  [{status}] {cls_name}: {img_count} images")
    
    print(f"\n[OK] Total images selected: {total_images}")
    print(f"[OK] Output directory: {RAW_DIR}")
    
    # Verify all classes meet minimum
    for cls_name, count in class_stats.items():
        if count < MIN_IMAGES_PER_CLASS:
            print(f"\n[WARNING] {cls_name} has only {count} images "
                  f"(minimum: {MIN_IMAGES_PER_CLASS})")
    
    return class_stats


def print_summary(class_stats):
    """Print a nice summary table of the downloaded data."""
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"{'Class':<20} {'Images':>10}")
    print("-" * 32)
    for cls_name, count in class_stats.items():
        print(f"{cls_name:<20} {count:>10}")
    print("-" * 32)
    print(f"{'TOTAL':<20} {sum(class_stats.values()):>10}")
    print(f"\nData saved to: {RAW_DIR}")
    print("Next step: python preprocess.py")


def main():
    """
    Main execution pipeline:
        1. Download Fruits-360 from Kaggle
        2. Extract the zip archive
        3. Find Training/ and Test/ folders
        4. Select and copy 6 target classes to data/raw/
        5. Print summary statistics
    """
    print("[*] Fruit SVM Project - Data Download")
    print("=" * 60)
    
    # Step 1: Download
    zip_path = download_dataset()
    
    # Step 2: Extract
    extract_dir = extract_dataset(zip_path)
    
    # Step 3: Find dataset folders
    training_dir, test_dir = find_dataset_folders(extract_dir)
    
    # Step 4: Select and copy classes
    class_stats = select_and_copy_classes(training_dir, test_dir)
    
    # Step 5: Summary
    print_summary(class_stats)
    
    print("\n[DONE] Data download and organization complete!")
    print("[NEXT] Run: python preprocess.py")


if __name__ == "__main__":
    main()
