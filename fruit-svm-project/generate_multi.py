import cv2
import numpy as np
import os

def create_multi_fruit_image():
    # Base colors used in setup_data.py
    base_colors = {
        'Apple': (30, 30, 200),     # BGR
        'Banana': (50, 220, 220),
        'Orange': (0, 128, 255),
        'Mango': (50, 200, 255),
        'Grapes': (128, 0, 128),
        'Strawberry': (60, 20, 220)
    }

    # Create a white background
    img = np.ones((400, 400, 3), dtype=np.uint8) * 255

    # Draw some fruits (circles) on the image
    fruits = [
        ('Apple', (100, 100), 40),
        ('Banana', (300, 100), 30),
        ('Orange', (200, 200), 50),
        ('Grapes', (100, 300), 45)
    ]
    
    for name, center, radius in fruits:
        color = base_colors[name]
        # Add a bit of noise
        noise = np.random.normal(0, 20, 3)
        color = tuple(np.clip(color + noise, 0, 255).astype(int))
        color = (int(color[0]), int(color[1]), int(color[2]))
        
        cv2.circle(img, center, radius, color, -1)
        
        # Draw a small contour/stem to give it texture
        cv2.circle(img, (center[0]-10, center[1]-10), 5, (0,0,0), -1)

    # Save
    out_path = "test_multi.jpg"
    cv2.imwrite(out_path, img)
    print(f"Created synthetic multi-fruit image at {out_path}")

if __name__ == "__main__":
    create_multi_fruit_image()
