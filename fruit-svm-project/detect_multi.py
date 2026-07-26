import cv2
import numpy as np
import pickle
import os
from features import extract_features

def detect_and_classify(image_path, model_path="model.pkl"):
    # Load model
    try:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
    except FileNotFoundError:
        print("Model not found. Train the model first.")
        return

    model = model_data['model']
    idx_to_class = model_data['idx_to_class']
    feature_mean = model_data['feature_mean']
    feature_std = model_data['feature_std']

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found.")
        return
        
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Basic Segmentation: thresholding
    # Since our synthetic data has bright base colors, they will contrast with a black or white background.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter small contours (noise)
    min_area = 100
    
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
            
        # Get bounding box
        x, y, w, h = cv2.boundingRect(c)
        
        # Skip boxes that are too small
        if w < 10 or h < 10:
            continue
            
        # Extract region
        roi = img[y:y+h, x:x+w]
        
        # Save temporary ROI to extract features (since extract_features takes a path)
        roi_path = "temp_roi.jpg"
        cv2.imwrite(roi_path, roi)
        
        try:
            feats = extract_features(roi_path)
            feats = (feats - feature_mean) / feature_std
            feats = feats.reshape(1, -1)
            
            # Predict
            pred_idx = model.predict(feats)[0]
            fruit_name = idx_to_class[pred_idx]
            probs = model.predict_proba(feats)[0]
            conf = probs[pred_idx] * 100
            
            # Draw box and label
            cv2.rectangle(original, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"{fruit_name} ({conf:.1f}%)"
            cv2.putText(original, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
        except Exception as e:
            print(f"Error classifying ROI: {e}")
            
    # Save output
    out_path = "multi_fruit_output.jpg"
    cv2.imwrite(out_path, original)
    print(f"Multi-fruit detection complete. Found {len(contours)} contours. Output saved to {out_path}")
    
    # Clean up
    if os.path.exists("temp_roi.jpg"):
        os.remove("temp_roi.jpg")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python detect_multi.py <image_path>")
    else:
        detect_and_classify(sys.argv[1])
