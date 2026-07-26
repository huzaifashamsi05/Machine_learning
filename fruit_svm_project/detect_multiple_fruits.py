#!/usr/bin/env python3
"""
Multi-Fruit Detection Script
Usage: python detect_multiple_fruits.py <image_path>
"""

import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
from scipy.special import expit

from evaluate import load_model, extract_all_features, KernelBinarySVM, KernelSVM_OvR

FRUIT_CLASSES = ['Apple', 'Banana', 'Orange', 'Mango', 'Grapes', 'Strawberry']

def detect_multiple_fruits(image_path, model_path="models/complete_model.pkl"):
    print("Loading model...")
    model, platt_data = load_model(model_path)
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load: {image_path}")
        
    original = img.copy()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Grayscale and thresholding/edges for contour detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Using Otsu's thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    
    # Minimum and maximum area to consider as a fruit
    min_area = 500
    max_area = (img.shape[0] * img.shape[1]) * 0.9
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Crop the bounding box
            roi = img_rgb[y:y+h, x:x+w]
            
            # Preprocess the ROI just like in predict_fruit
            roi_resized = cv2.resize(roi, (64, 64))
            roi_normalized = roi_resized.astype(np.float32) / 255.0
            
            # Extract features
            features = extract_all_features(roi_normalized).reshape(1, -1)
            
            # Predict
            _, scores = model.predict(features)
            
            # Platt Scaling
            platt_probs = np.zeros(6)
            for i, (A, B) in enumerate(platt_data['scalers']):
                platt_probs[i] = expit(A * scores[0,i] + B)
            platt_probs = platt_probs / np.sum(platt_probs)
            
            pred_idx = np.argmax(platt_probs)
            confidence = platt_probs[pred_idx] * 100
            
            # Only keep high confidence predictions
            if confidence > 50.0:
                fruit_name = FRUIT_CLASSES[pred_idx]
                detections.append({
                    'fruit': fruit_name,
                    'confidence': confidence,
                    'bbox': (x, y, w, h)
                })
                
                # Draw on the original image
                cv2.rectangle(original, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label = f"{fruit_name} {confidence:.1f}%"
                cv2.putText(original, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
    annotated_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    return annotated_rgb, detections, None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python detect_multiple_fruits.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output_annotated.png"
    
    annotated, detections, _ = detect_multiple_fruits(image_path)
    
    print(f"\nDetected {len(detections)} fruits:")
    for i, d in enumerate(detections):
        print(f"  {i+1}. {d['fruit']} ({d['confidence']:.1f}%)")
    
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        print(f"\nSaved annotated image to: {output_path}")
