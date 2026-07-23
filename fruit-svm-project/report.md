# Fruit Image Classification using SVM from Scratch - Report

## 1. Project Overview

This project implements a complete fruit image classification system using 
Support Vector Machines (SVM) built entirely from scratch. No sklearn SVM or 
pretrained models were used — only NumPy, PIL, matplotlib, and scipy.optimize.

**Goal**: Classify fruit images into 6 categories with 85%+ accuracy.

## 2. Dataset

**Source**: Fruits-360 dataset from Kaggle (by Horea Muresan & Mihai Oltean)

**Selected Classes** (6 classes chosen for maximum visual diversity):

| Class        | Color        | Shape      | Texture    | Why Selected                     |
|-------------|-------------|------------|------------|----------------------------------|
| Apple Red 1  | Red          | Round      | Smooth     | Baseline round red fruit         |
| Banana       | Yellow       | Elongated  | Smooth     | Unique elongated shape           |
| Orange       | Orange       | Round      | Dimpled    | Distinct orange color            |
| Strawberry   | Red          | Conical    | Seeded     | Unique shape + texture           |
| Lemon        | Yellow       | Oval       | Smooth     | Yellow like banana but different shape |
| Kiwi         | Brown/Green  | Oval       | Fuzzy      | Unique color + texture           |

**Preprocessing**:
- Resized to 64x64 pixels (bilinear interpolation)
- Split: 70% train / 15% validation / 15% test (stratified)

## 3. Feature Engineering

Total features per image: **196** (48 color + 4 shape + 144 texture)

### 3.1 Color Histograms (HSV) — 48 features
- Converted RGB to HSV color space (manual implementation)
- 16-bin histogram for each of H, S, V channels
- L1 normalized (sum to 1)

**Why HSV?** 
HSV separates chromatic information (Hue) from brightness (Value). 
This makes features robust to lighting changes — a red apple should have 
similar Hue values whether photographed in bright or dim light.

### 3.2 Shape Features — 4 features
- **Area ratio**: fraction of image occupied by the fruit
- **Roundness**: 4*pi*area/perimeter^2 (1.0 = perfect circle)
- **Aspect ratio**: bounding box width/height
- **Solidity**: area / bounding box area

**Why these?**
Bananas are elongated (low roundness, high aspect ratio), oranges are round 
(high roundness, ~1.0 aspect ratio), strawberries are conical.

Binary mask created using Otsu thresholding (implemented from scratch):
- Finds optimal threshold by maximizing inter-class variance
- Adapts to each image's brightness distribution

### 3.3 HOG (Histogram of Oriented Gradients) — 144 features
- Gradient computation using Sobel filters: Gx = I(x+1) - I(x-1)
- Image divided into 4x4 grid of 16x16 pixel cells
- 9-bin orientation histogram per cell (0-180 degrees)
- Soft binning (bilinear interpolation between adjacent bins)
- L2 normalized for contrast invariance

**Why HOG?**
HOG captures the distribution of edge orientations, which encodes texture 
and shape structure. Kiwi's fuzzy texture produces many random-oriented 
edges, while banana's smooth surface has fewer, more uniform edges.

### 3.4 Normalization
Z-score standardization: x_scaled = (x - mean) / std
- Computed on training set only (prevents data leakage)
- Critical for SVM: features with larger magnitudes would dominate 
  the kernel computation without normalization

## 4. SVM Implementation

### 4.1 Linear SVM (Soft Margin)

**Objective** (Primal formulation):
```
min_{w,b} (1/2)||w||^2 + C * (1/N) * sum_i max(0, 1 - y_i(w^T x_i + b))
```

**Optimization**: Mini-batch sub-gradient descent
```
if y_i(w^T x_i + b) < 1:     (misclassified or in margin)
    dw = w - C * y_i * x_i
    db = -C * y_i
else:
    dw = w
    db = 0

w <- w - eta(t) * dw
b <- b - eta(t) * db
eta(t) = eta_0 / (1 + lambda * t)
```

**Why sub-gradient?**
The hinge loss max(0, 1-z) has a "kink" at z=1 where it's not differentiable. 
Sub-gradient descent works on non-differentiable convex functions.

### 4.2 Kernel SVM (Dual Formulation)

**Dual objective**:
```
max_alpha sum_i alpha_i - (1/2) sum_{i,j} alpha_i alpha_j y_i y_j K(x_i, x_j)
s.t. 0 <= alpha_i <= C, sum_i alpha_i y_i = 0
```

**Decision function**:
```
f(x) = sum_i alpha_i y_i K(x_i, x) + b
```

Solved using scipy.optimize.minimize (SLSQP method).

### 4.3 Kernel Functions

| Kernel     | Formula                          | Hyperparameters    |
|-----------|----------------------------------|--------------------|
| Linear    | K(x,z) = x^T z                  | None               |
| Polynomial| K(x,z) = (x^T z + c)^d          | degree d, coef0 c  |
| RBF       | K(x,z) = exp(-gamma*||x-z||^2)  | gamma              |

**RBF kernel** maps to infinite-dimensional space, enabling arbitrarily 
complex decision boundaries. The gamma parameter controls the "width" of 
the Gaussian: smaller gamma = smoother boundary, larger gamma = more complex.

### 4.4 Multi-Class: One-vs-Rest (OvR)

For K=6 classes, train 6 binary classifiers:
- Classifier k: class k (+1) vs all others (-1)
- Prediction: argmax_k f_k(x) (class with highest score)

**Why OvR over One-vs-One?**
- OvR: K classifiers (6 for our case)
- OvO: K(K-1)/2 classifiers (15 for our case)
- OvR is simpler, faster, and decision values are directly comparable

## 5. Confidence Calibration (Platt Scaling)

SVM outputs raw scores f(x), not probabilities. Platt scaling fits a 
sigmoid to convert scores to calibrated probabilities:

```
P(y=1|f) = 1 / (1 + exp(A*f + B))
```

Parameters A and B are learned by minimizing cross-entropy on validation set.

For multi-class, apply softmax normalization across OvR probabilities:
```
P(class k) = exp(p_k) / sum_j exp(p_j)
```

## 6. Hyperparameter Tuning

Tuned on validation set (no test set contamination):

**Linear SVM**: C in {0.001, 0.01, 0.1, 1.0, 10.0, 100.0}

**Polynomial Kernel**: 
- degree in {2, 3, 4}
- C in {0.1, 1.0, 10.0}

**RBF Kernel**: 
- gamma in {0.001, 0.01, 0.1, 1.0}
- C in {0.1, 1.0, 10.0}

## 7. Results

(Results will be filled after training completes)

### Model Comparison
| Model           | Val Accuracy | Test Accuracy |
|----------------|-------------|---------------|
| Linear SVM     |             |               |
| Poly Kernel    |             |               |
| RBF Kernel     |             |               |

### Best Model Configuration
- Model: TBD
- Parameters: TBD
- Test Accuracy: TBD
- Mean Confidence: TBD

## 8. Lessons Learned

1. **Feature engineering is crucial**: With handcrafted features + SVM, 
   the quality of features directly determines performance
2. **Normalization matters**: Without z-score normalization, SVM fails 
   to converge properly
3. **Kernel choice**: RBF kernel typically outperforms linear for 
   non-linearly separable data, but requires careful gamma tuning
4. **Calibration**: Raw SVM scores are not probabilities — Platt scaling 
   is necessary for meaningful confidence values

## 9. Dependencies

- numpy: Array operations, linear algebra
- Pillow (PIL): Image loading and resizing
- matplotlib: Plotting and visualization
- scipy.optimize: Dual SVM optimization (SLSQP solver)

No sklearn SVM, no pretrained models, no deep learning.
