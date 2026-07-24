import sys
import pickle
import numpy as np
from features import extract_features

def main(img_path):
    try:
        with open("model.pkl", "rb") as f:
            model_data = pickle.load(f)
    except FileNotFoundError:
        print("Error: model.pkl not found. Please run train.py first.")
        return

    model = model_data['model']
    idx_to_class = model_data['idx_to_class']
    feature_mean = model_data['feature_mean']
    feature_std = model_data['feature_std']

    try:
        feats = extract_features(img_path)
    except Exception as e:
        print(f"Error extracting features: {e}")
        return

    # Standardize
    feats = (feats - feature_mean) / feature_std
    feats = feats.reshape(1, -1)

    # Predict
    pred_idx = model.predict(feats)[0]
    fruit_name = idx_to_class[pred_idx]
    
    # Confidence
    probs = model.predict_proba(feats)[0]
    confidence = probs[pred_idx] * 100

    print(f"Prediction: {fruit_name}")
    print(f"Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <path_to_image>")
        sys.exit(1)
        
    main(sys.argv[1])
