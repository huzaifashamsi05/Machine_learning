#!/usr/bin/env python3
"""
Multi-Fruit Detection Script
Usage: python detect_multiple_fruits.py <image_path>
"""

import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
from scipy.special import expit
from skimage.feature import hog, local_binary_pattern

FRUIT_CLASSES = ['Apple', 'Banana', 'Orange', 'Mango', 'Grapes', 'Strawberry']
IMG_SIZE = 64

# ============================================================
# FEATURE EXTRACTION
# ============================================================
def extract_color_features_hsv(img, bins=16):
    img_uint8 = (img * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [bins], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [bins], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [bins], [0, 256]).flatten()
    h_hist = h_hist / (h_hist.sum() + 1e-8)
    s_hist = s_hist / (s_hist.sum() + 1e-8)
    v_hist = v_hist / (v_hist.sum() + 1e-8)
    h_mean, h_std = hsv[:,:,0].mean(), hsv[:,:,0].std()
    s_mean, s_std = hsv[:,:,1].mean(), hsv[:,:,1].std()
    v_mean, v_std = hsv[:,:,2].mean(), hsv[:,:,2].std()
    return np.concatenate([h_hist, s_hist, v_hist,
        [h_mean/180, h_std/180, s_mean/256, s_std/256, v_mean/256, v_std/256]])

def extract_shape_features(img):
    img_uint8 = (img * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return np.array([0.5]*8)
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)
    circularity = 4*np.pi*area/(perimeter**2) if perimeter>0 else 0
    x,y,w,h = cv2.boundingRect(largest)
    aspect_ratio = float(w)/(h+1e-8)
    hull = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull)
    solidity = area/(hull_area+1e-8)
    if len(largest)>=5:
        ellipse = cv2.fitEllipse(largest)
        major_axis, minor_axis = max(ellipse[1]), min(ellipse[1])
        eccentricity = np.sqrt(1-(minor_axis/major_axis)**2) if major_axis>0 else 0
    else:
        eccentricity = 0
    extent = area/(w*h+1e-8)
    norm_area = area/(img.shape[0]*img.shape[1])
    return np.array([circularity, aspect_ratio, solidity, eccentricity,
                     extent, norm_area, w/img.shape[1], h/img.shape[0]])

def extract_hog_features(img):
    img_uint8 = (img*255).astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    fd = hog(gray, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2),
             visualize=False, feature_vector=True)
    return fd/(np.linalg.norm(fd)+1e-8)

def extract_lbp_features(img, P=8, R=1, bins=16):
    img_uint8 = (img*255).astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, P=P, R=R, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0,bins))
    lbp_hist = lbp_hist.astype(np.float32)
    lbp_hist = lbp_hist/(lbp_hist.sum()+1e-8)
    return np.concatenate([lbp_hist, [lbp.mean()/255.0, lbp.std()/255.0]])

def extract_all_features(img):
    return np.concatenate([
        extract_color_features_hsv(img),
        extract_shape_features(img),
        extract_hog_features(img),
        extract_lbp_features(img)
    ])

# ============================================================
# MODEL
# ============================================================
class KernelBinarySVM:
    def __init__(self, gamma=0.01):
        self.gamma = gamma
        self.alpha_sv = None
        self.support_vectors = None
        self.support_labels = None
        self.b = 0
        
    def _compute_kernel(self, x, z):
        diff = x - z
        return np.exp(-self.gamma * np.dot(diff, diff))
    
    def decision_function(self, X_test):
        n_test = len(X_test)
        n_sv = len(self.support_vectors)
        K_test_sv = np.zeros((n_test, n_sv))
        for i in range(n_test):
            for j in range(n_sv):
                K_test_sv[i, j] = self._compute_kernel(X_test[i], self.support_vectors[j])
        return K_test_sv @ (self.alpha_sv * self.support_labels) + self.b

class KernelSVM_OvR:
    def __init__(self, classifiers, classes):
        self.classifiers = classifiers
        self.classes = classes
        
    def predict(self, X):
        scores = np.zeros((len(X), len(self.classifiers)))
        for i, svm in enumerate(self.classifiers):
            scores[:, i] = svm.decision_function(X)
        predictions = np.argmax(scores, axis=1)
        return predictions, scores

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    svm_data = model_data['svm']
    classifiers = []
    for alpha_sv, sv, sl, b in svm_data['classifiers']:
        svm = KernelBinarySVM(gamma=svm_data['gamma'])
        svm.alpha_sv = alpha_sv
        svm.support_vectors = sv
        svm.support_labels = sl
        svm.b = b
        classifiers.append(svm)
    return KernelSVM_OvR(classifiers, svm_data['classes']), model_data['platt']

def classify_region(img_region, model, platt_data):
    img_resized = cv2.resize(img_region, (IMG_SIZE, IMG_SIZE))
    img_norm = img_resized.astype(np.float32) / 255.0
    features = extract_all_features(img_norm).reshape(1, -1)
    
    _, scores = model.predict(features)
    
    platt_probs = np.zeros(6)
    for i, (A, B) in enumerate(platt_data['scalers']):
        platt_probs[i] = expit(A * scores[0, i] + B)
    
    platt_probs = platt_probs / np.sum(platt_probs)
    pred_idx = np.argmax(platt_probs)
    confidence = platt_probs[pred_idx] * 100
    
    return FRUIT_CLASSES[pred_idx], confidence, platt_probs

# ============================================================
# MULTI-FRUIT DETECTION
# ============================================================
def detect_multiple_fruits(image_path, model, platt_data, output_path=None):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load: {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    print(f"\n📸 Processing: {image_path}")
    print(f"   Size: {w}x{h}")
    
    # Color segmentation
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    color_ranges = [
        {'lower': np.array([0, 50, 50]), 'upper': np.array([10, 255, 255])},
        {'lower': np.array([170, 50, 50]), 'upper': np.array([180, 255, 255])},
        {'lower': np.array([10, 50, 50]), 'upper': np.array([35, 255, 255])},
        {'lower': np.array([35, 40, 40]), 'upper': np.array([85, 255, 255])},
    ]
    
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    for cr in color_ranges:
        mask = cv2.inRange(hsv, cr['lower'], cr['upper'])
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"   Found {len(contours)} contours")
    
    detections = []
    annotated_img = img_rgb.copy()
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < 1000:
            continue
        
        x, y, bw, bh = cv2.boundingRect(contour)
        pad = 10
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)
        
        region = img_rgb[y1:y2, x1:x2]
        fruit_name, confidence, all_probs = classify_region(region, model, platt_data)
        
        if confidence < 30:
            continue
        
        detections.append({
            'fruit': fruit_name,
            'confidence': confidence,
            'bbox': (x1, y1, x2-x1, y2-y1)
        })
        
        color = (0, 255, 0) if confidence > 70 else (255, 165, 0) if confidence > 50 else (255, 0, 0)
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
        
        label = f"{fruit_name}: {confidence:.1f}%"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated_img, (x1, y1-label_size[1]-10), 
                     (x1+label_size[0], y1), color, -1)
        cv2.putText(annotated_img, label, (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    print(f"   ✅ Detected {len(detections)} fruits")
    
    # Save output
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
        print(f"   💾 Saved: {output_path}")
    
    # Show result
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original", fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(combined_mask, cmap='gray')
    axes[1].set_title("Segmentation Mask", fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(annotated_img)
    axes[2].set_title(f"Detected: {len(detections)} fruits", fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return annotated_img, detections

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python detect_multiple_fruits.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = os.path.join(os.path.dirname(__file__), "models", "complete_model.pkl")
    
    print("Loading model...")
    model, platt_data = load_model(model_path)
    
    print(f"Detecting fruits in: {image_path}")
    detect_multiple_fruits(image_path, model, platt_data, output_path="detected_output.jpg")