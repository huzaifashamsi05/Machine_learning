import os
import numpy as np
from PIL import Image

RAW_DIR = "data/raw"
CLASSES = {
    "Apple Red 1": [200, 20, 20],   # Red
    "Banana": [220, 220, 20],      # Yellow
    "Orange": [250, 150, 0],       # Orange
    "Strawberry": [220, 50, 50],   # Pinkish Red
    "Lemon": [255, 255, 0],        # Bright Yellow
    "Kiwi": [139, 69, 19]          # Brown
}

IMAGES_PER_CLASS = 85  # Matches the project requirement of 80+ images

def generate_dummy_data():
    print("Generating miniature dummy dataset to save disk space...")
    os.makedirs(RAW_DIR, exist_ok=True)
    
    np.random.seed(42)
    
    for cls, base_color in CLASSES.items():
        cls_dir = os.path.join(RAW_DIR, cls)
        os.makedirs(cls_dir, exist_ok=True)
        
        for i in range(IMAGES_PER_CLASS):
            # Create a 64x64 image with base color + some random noise
            noise = np.random.randint(-30, 30, size=(64, 64, 3))
            img_data = np.clip(np.array(base_color) + noise, 0, 255).astype(np.uint8)
            
            img = Image.fromarray(img_data)
            img.save(os.path.join(cls_dir, f"dummy_{i:03d}.jpg"))
            
        print(f"Generated {IMAGES_PER_CLASS} images for {cls}")
        
    print(f"Complete! Total images: {IMAGES_PER_CLASS * len(CLASSES)}")

if __name__ == "__main__":
    generate_dummy_data()
