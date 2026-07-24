import os
import numpy as np
from PIL import Image

def create_dummy_data(base_dir="data/raw", num_classes=6, images_per_class=80, img_size=(64, 64)):
    """
    Creates lightweight dummy images for testing the pipeline to avoid heavy data downloads.
    """
    classes = ['Apple', 'Banana', 'Orange', 'Mango', 'Grapes', 'Strawberry']
    
    # Base colors for each fruit class (approximate HSV/RGB means)
    base_colors = {
        'Apple': (200, 30, 30),
        'Banana': (220, 220, 50),
        'Orange': (255, 128, 0),
        'Mango': (255, 200, 50),
        'Grapes': (128, 0, 128),
        'Strawberry': (220, 20, 60)
    }

    os.makedirs(base_dir, exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    print(f"Generating {images_per_class} lightweight dummy images per class (total {num_classes * images_per_class})...")
    
    for i in range(num_classes):
        fruit_name = classes[i]
        class_dir = os.path.join(base_dir, fruit_name)
        os.makedirs(class_dir, exist_ok=True)
        
        color_mean = base_colors[fruit_name]
        
        for j in range(images_per_class):
            # Create a dummy image with random noise around the base color
            # This ensures they are lightweight but have some variation for the SVM to learn
            img_data = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
            for c in range(3):
                noise = np.random.normal(0, 20, (img_size[1], img_size[0]))
                channel = np.clip(color_mean[c] + noise, 0, 255)
                img_data[:, :, c] = channel.astype(np.uint8)
            
            img = Image.fromarray(img_data)
            img.save(os.path.join(class_dir, f"{fruit_name}_{j:03d}.jpg"))

    print("Dummy data generation complete. You can replace these with real photos later.")

if __name__ == "__main__":
    create_dummy_data()
