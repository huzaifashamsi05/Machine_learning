import os
import zipfile
import shutil

ZIP_PATH = "data/_download/fruits.zip"
RAW_DIR = "data/raw"

SELECTED_CLASSES = [
    "Apple Red 1",
    "Banana",
    "Orange",
    "Strawberry",
    "Lemon",
    "Kiwi",
]

print("Selective extraction started...")
os.makedirs(RAW_DIR, exist_ok=True)

with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    all_files = z.namelist()
    
    for cls in SELECTED_CLASSES:
        cls_dir = os.path.join(RAW_DIR, cls)
        os.makedirs(cls_dir, exist_ok=True)
        
        # Find files for this class in Training or Test
        # Format usually: fruits-360_dataset/fruits-360/Training/Apple Red 1/0_100.jpg
        # Or: Training/Apple Red 1/0_100.jpg
        
        count = 0
        for f in all_files:
            if f"/{cls}/" in f and f.lower().endswith(('.jpg', '.jpeg', '.png')):
                # We extract it directly to our raw folder, ignoring original paths
                filename = os.path.basename(f)
                prefix = "train_" if "/Training/" in f else "test_" if "/Test/" in f else "img_"
                
                # Extract to memory then write
                source = z.open(f)
                target_path = os.path.join(cls_dir, f"{prefix}{filename}")
                with open(target_path, "wb") as out:
                    shutil.copyfileobj(source, out)
                count += 1
                
        print(f"Extracted {count} images for {cls}")

print("Selective extraction complete!")
