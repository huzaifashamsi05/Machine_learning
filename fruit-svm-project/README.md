# 🍎 Fruit Image Classification using SVM from Scratch

A complete fruit image classification project using **Support Vector Machines (SVM)** 
and **Kernel Methods** implemented entirely from scratch — no sklearn SVM or pretrained models.

## 📊 Dataset
- **Fruits-360** dataset from Kaggle (6 classes selected)
- Classes: Apple Red 1, Banana, Orange, Strawberry, Lemon, Kiwi
- 80+ images per class, resized to 64×64

## 🛠️ Features Extracted
- **Color Histograms** (RGB/HSV) — 48 features
- **Shape Features** (roundness, area, aspect ratio, solidity) — 4 features  
- **Texture Features** (HOG - Histogram of Oriented Gradients) — 36 features
- **Total**: 88 features per image

## 🧮 Models (All from Scratch)
1. **Linear SVM** — Soft margin with hinge loss gradient descent
2. **Polynomial Kernel SVM** — K(x,z) = (xᵀz + c)^d
3. **RBF Kernel SVM** — K(x,z) = exp(-||x-z||²/2σ²)
4. **One-vs-Rest** multi-class classification
5. **Platt Scaling** for confidence calibration

## 📁 Project Structure
```
fruit-svm-project/
├── data/
│   ├── raw/            # Original Fruits-360 images
│   └── processed/      # Preprocessed .npz files
├── models/             # Saved trained models (.npz)
├── figures/            # Training curves, confusion matrices
├── download_data.py    # Dataset download & organization
├── preprocess.py       # Resize, split, save
├── features.py         # Feature extraction pipeline
├── kernels.py          # Kernel functions (linear, poly, RBF)
├── svm.py              # SVM implementations from scratch
├── train.py            # Training pipeline
├── evaluate.py         # Evaluation & single-image prediction
└── report.md           # Project report
```

## 🚀 Usage
```bash
# Download and preprocess data
python download_data.py
python preprocess.py

# Train models
python train.py

# Evaluate on test set
python evaluate.py --test

# Predict single image
python evaluate.py --image path/to/fruit.jpg
```

## 📦 Dependencies
- numpy
- Pillow (PIL)
- matplotlib
- pandas
- scipy

**No sklearn SVM, no pretrained models.**

## 🎯 Target
- 85%+ validation accuracy
- Confidence scores (0-100%) via Platt scaling
