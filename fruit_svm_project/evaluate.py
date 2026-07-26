#!/usr/bin/env python3
"""
Fruit Classifier - evaluate.py
Usage: python evaluate.py <image_path>
"""

import sys
import numpy as np
import cv2
from PIL import Image
import pickle
from skimage.feature import hog, local_binary_pattern
from scipy.special import expit

# Configuration
IMG_SIZE = 64
FRUIT_CLASSES = ['Apple', 'Banana', 'Orange', 'Mango', 'Grapes', 'Strawberry']

# [Feature extraction functions - same as above]
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
        n_test, n_sv = len(X_test), len(self.support_vectors)
        K = np.zeros((n_test, n_sv))
        for i in range(n_test):
            for j in range(n_sv):
                K[i,j] = self._compute_kernel(X_test[i], self.support_vectors[j])
        return K @ (self.alpha_sv * self.support_labels) + self.b

class KernelSVM_OvR:
    def __init__(self, classifiers, classes):
        self.classifiers = classifiers
        self.classes = classes
    def predict(self, X):
        scores = np.zeros((len(X), len(self.classifiers)))
        for i, svm in enumerate(self.classifiers):
            scores[:,i] = svm.decision_function(X)
        return np.argmax(scores, axis=1), scores

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

def predict_fruit(image_path, model, platt_data):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (64, 64))
    img = img.astype(np.float32)/255.0
    features = extract_all_features(img).reshape(1, -1)
    _, scores = model.predict(features)
    platt_probs = np.zeros(6)
    for i, (A, B) in enumerate(platt_data['scalers']):
        platt_probs[i] = expit(A * scores[0,i] + B)
    platt_probs = platt_probs / np.sum(platt_probs)
    pred_idx = np.argmax(platt_probs)
    return FRUIT_CLASSES[pred_idx], platt_probs[pred_idx]*100

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = "models/complete_model.pkl"
    
    print("Loading model...")
    model, platt_data = load_model(model_path)
    
    print(f"Classifying: {image_path}")
    fruit, confidence = predict_fruit(image_path, model, platt_data)
    
    print(f"\nPredicted Fruit: {fruit}")
    print(f"Confidence: {confidence:.1f}%")
