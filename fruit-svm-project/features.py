import cv2
import numpy as np

def extract_color_features(img_hsv, bins=(8, 8, 8)):
    """Extracts a 3D color histogram from the HSV image."""
    hist = cv2.calcHist([img_hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_shape_features(img_gray):
    """Extracts basic shape features (aspect ratio, extent, equivalent diameter) using contours."""
    # Thresholding to find the main object
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return np.array([0, 0, 0])
    
    # Assume the largest contour is the fruit
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    
    # Bounding box aspect ratio
    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = float(w) / h if h != 0 else 0
    
    # Extent (object area / bounding box area)
    rect_area = w * h
    extent = float(area) / rect_area if rect_area != 0 else 0
    
    # Equivalent Diameter
    eq_diameter = np.sqrt(4 * area / np.pi)
    
    # Normalize eq_diameter by image diagonal for scale invariance
    h_img, w_img = img_gray.shape
    diag = np.sqrt(h_img**2 + w_img**2)
    eq_diameter /= diag
    
    return np.array([aspect_ratio, extent, eq_diameter])

def extract_texture_features(img_gray):
    """Extracts texture features using Sobel edge detection density."""
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    
    # Calculate mean and std deviation of edge magnitudes
    mean_mag = np.mean(magnitude)
    std_mag = np.std(magnitude)
    
    # Edge density: proportion of pixels with magnitude above a threshold
    threshold = 50
    edge_density = np.sum(magnitude > threshold) / magnitude.size
    
    return np.array([mean_mag, std_mag, edge_density])

def extract_features(img_path):
    """Combines color, shape, and texture features into a single feature vector."""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    
    # Resize to standard size for consistent shape/texture scales
    img = cv2.resize(img, (64, 64))
    
    # Color features in HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_feats = extract_color_features(img_hsv)
    
    # Shape and Texture on Grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    shape_feats = extract_shape_features(img_gray)
    texture_feats = extract_texture_features(img_gray)
    
    # Concatenate all features
    features = np.hstack((color_feats, shape_feats, texture_feats))
    return features
