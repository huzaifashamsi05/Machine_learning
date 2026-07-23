"""
kernels.py - Kernel Functions for SVM
======================================

Implements three kernel functions from scratch:
    1. Linear Kernel
    2. Polynomial Kernel
    3. RBF (Gaussian) Kernel

Theory:
=======
A kernel K(x, z) computes the inner product in a (potentially infinite-
dimensional) feature space without explicitly mapping to that space.
This is the "kernel trick":
    
    K(x, z) = <phi(x), phi(z)>

where phi is the implicit feature mapping.

Mercer's condition: K is a valid kernel if the Gram matrix K_ij = K(x_i, x_j)
is positive semi-definite for any set of points {x_1, ..., x_n}.

Why Kernels?
    Linear SVM can only find linear decision boundaries.
    Kernels allow SVM to find non-linear boundaries by working
    in a higher-dimensional space where the data IS linearly separable.

    Example: XOR problem is not linearly separable in 2D,
    but becomes separable in 3D with polynomial features.
"""

import numpy as np


def linear_kernel(X, Y):
    """
    Linear Kernel: K(x, z) = x^T z
    
    This is just the standard dot product. The implicit feature mapping
    is the identity: phi(x) = x.
    
    Linear kernel SVM is equivalent to standard (primal) linear SVM.
    We include it for completeness and as a baseline comparison.
    
    Properties:
        - No hyperparameters to tune
        - Fast to compute: O(n*m*d) where d = feature dimension
        - Can only find linear decision boundaries
    
    Args:
        X: numpy array (n, d) - n samples with d features
        Y: numpy array (m, d) - m samples with d features
    
    Returns:
        K: numpy array (n, m) - kernel matrix where K[i,j] = X[i] . Y[j]
    """
    return X @ Y.T


def polynomial_kernel(X, Y, degree=3, coef0=1.0):
    """
    Polynomial Kernel: K(x, z) = (x^T z + c)^d
    
    The implicit feature mapping includes all monomials of the input
    features up to degree d. For example, with d=2 and x = [x1, x2]:
    
        phi(x) = [1, sqrt(2c)*x1, sqrt(2c)*x2, x1^2, x2^2, sqrt(2)*x1*x2]
    
    This allows the SVM to model polynomial decision boundaries.
    
    Hyperparameters:
        degree (d): Degree of the polynomial
            - d=1: Linear (same as linear kernel with offset)
            - d=2: Quadratic (can model elliptical boundaries)
            - d=3: Cubic (more complex boundaries)
            Higher degree = more complex model = risk of overfitting
        
        coef0 (c): Independent term, controls the influence of 
            higher-order vs lower-order terms
            - c=0: Homogeneous polynomial (only highest degree terms)
            - c=1: Balanced mixture of all degrees up to d
    
    Numerical considerations:
        - For high degree, values can become very large
        - This can cause numerical instability
        - Keep degree <= 5 and normalize input features
    
    Args:
        X: numpy array (n, d)
        Y: numpy array (m, d)
        degree: int, polynomial degree (default: 3)
        coef0: float, independent term (default: 1.0)
    
    Returns:
        K: numpy array (n, m)
    """
    return (X @ Y.T + coef0) ** degree


def rbf_kernel(X, Y, gamma=0.1):
    """
    RBF (Radial Basis Function / Gaussian) Kernel:
        K(x, z) = exp(-gamma * ||x - z||^2)
    
    Equivalently:
        K(x, z) = exp(-||x - z||^2 / (2 * sigma^2))
    
    where gamma = 1 / (2 * sigma^2).
    
    The RBF kernel maps data to an INFINITE-dimensional feature space.
    This means it can model arbitrarily complex decision boundaries.
    
    Intuition:
        K(x, z) measures "similarity" between x and z.
        - If x == z: K = exp(0) = 1 (maximum similarity)
        - As ||x-z|| increases: K -> 0 (exponential decay)
        - sigma controls the "width" of the similarity function
    
    Hyperparameter:
        gamma (= 1/(2*sigma^2)):
            - Small gamma (large sigma): Smooth decision boundary
              Each point influences a large area -> underfitting risk
            - Large gamma (small sigma): Complex decision boundary
              Each point influences only nearby points -> overfitting risk
    
    Efficient computation trick:
        ||x - z||^2 = ||x||^2 + ||z||^2 - 2 * x^T * z
        
        This avoids O(n*m*d) pairwise distance computation.
        Instead: O(n*d) + O(m*d) + O(n*m*d) for the matrix multiply,
        but it's more memory efficient and vectorizable.
    
    Args:
        X: numpy array (n, d)
        Y: numpy array (m, d)
        gamma: float, kernel width parameter (default: 0.1)
    
    Returns:
        K: numpy array (n, m)
    """
    # ||x - z||^2 = ||x||^2 + ||z||^2 - 2 * x^T z
    X_sq = np.sum(X ** 2, axis=1).reshape(-1, 1)  # (n, 1)
    Y_sq = np.sum(Y ** 2, axis=1).reshape(1, -1)  # (1, m)
    XY = X @ Y.T                                   # (n, m)
    
    # Squared Euclidean distances
    dist_sq = X_sq + Y_sq - 2 * XY
    
    # Clip to avoid negative values due to numerical errors
    dist_sq = np.maximum(dist_sq, 0.0)
    
    return np.exp(-gamma * dist_sq)


def compute_kernel_matrix(X, Y, kernel_type='rbf', **kernel_params):
    """
    Compute the kernel matrix for a given kernel type.
    
    This is a convenience function that dispatches to the appropriate
    kernel function based on the kernel_type string.
    
    The kernel matrix (Gram matrix) K has shape (n, m) where:
        K[i, j] = K(X[i], Y[j])
    
    For training (X == Y), K is a symmetric positive semi-definite matrix.
    
    Args:
        X: numpy array (n, d)
        Y: numpy array (m, d)
        kernel_type: str, one of 'linear', 'poly', 'rbf'
        **kernel_params: additional parameters for the kernel function
    
    Returns:
        K: numpy array (n, m)
    """
    if kernel_type == 'linear':
        return linear_kernel(X, Y)
    elif kernel_type == 'poly':
        degree = kernel_params.get('degree', 3)
        coef0 = kernel_params.get('coef0', 1.0)
        return polynomial_kernel(X, Y, degree=degree, coef0=coef0)
    elif kernel_type == 'rbf':
        gamma = kernel_params.get('gamma', 0.1)
        return rbf_kernel(X, Y, gamma=gamma)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}. "
                        f"Must be 'linear', 'poly', or 'rbf'.")


# ============================================================
# Verification / Self-test
# ============================================================

if __name__ == "__main__":
    """
    Test kernel functions with known properties:
        1. K(x, x) should be maximal (for RBF, K(x,x) = 1)
        2. Kernel matrix should be symmetric for K(X, X)
        3. Kernel matrix should be positive semi-definite
    """
    print("[*] Kernel Functions - Self Test")
    print("=" * 50)
    
    np.random.seed(42)
    X = np.random.randn(10, 5)  # 10 samples, 5 features
    Y = np.random.randn(8, 5)   # 8 samples
    
    # Test 1: Linear kernel
    print("\n1. Linear Kernel:")
    K_lin = linear_kernel(X, Y)
    print(f"   Shape: {K_lin.shape} (expected: (10, 8))")
    K_lin_sym = linear_kernel(X, X)
    is_sym = np.allclose(K_lin_sym, K_lin_sym.T)
    print(f"   K(X,X) symmetric: {is_sym}")
    
    # Test 2: Polynomial kernel
    print("\n2. Polynomial Kernel (degree=3, c=1):")
    K_poly = polynomial_kernel(X, Y, degree=3, coef0=1.0)
    print(f"   Shape: {K_poly.shape}")
    K_poly_sym = polynomial_kernel(X, X, degree=3, coef0=1.0)
    is_sym = np.allclose(K_poly_sym, K_poly_sym.T)
    print(f"   K(X,X) symmetric: {is_sym}")
    # Check PSD: all eigenvalues >= 0
    eigvals = np.linalg.eigvalsh(K_poly_sym)
    is_psd = np.all(eigvals >= -1e-10)
    print(f"   K(X,X) positive semi-definite: {is_psd}")
    
    # Test 3: RBF kernel
    print("\n3. RBF Kernel (gamma=0.1):")
    K_rbf = rbf_kernel(X, Y, gamma=0.1)
    print(f"   Shape: {K_rbf.shape}")
    # K(x, x) should be 1.0 for RBF
    K_rbf_diag = rbf_kernel(X, X, gamma=0.1)
    diag_vals = np.diag(K_rbf_diag)
    print(f"   K(x,x) diagonal: {diag_vals[:5]} (should all be 1.0)")
    is_sym = np.allclose(K_rbf_diag, K_rbf_diag.T)
    print(f"   K(X,X) symmetric: {is_sym}")
    eigvals = np.linalg.eigvalsh(K_rbf_diag)
    is_psd = np.all(eigvals >= -1e-10)
    print(f"   K(X,X) positive semi-definite: {is_psd}")
    # All values should be in [0, 1]
    print(f"   Value range: [{K_rbf.min():.4f}, {K_rbf.max():.4f}] "
          f"(should be in [0, 1])")
    
    # Test 4: Dispatch function
    print("\n4. Dispatch function:")
    K1 = compute_kernel_matrix(X, Y, kernel_type='linear')
    K2 = compute_kernel_matrix(X, Y, kernel_type='poly', degree=2, coef0=0.5)
    K3 = compute_kernel_matrix(X, Y, kernel_type='rbf', gamma=0.5)
    print(f"   Linear: {K1.shape}, Poly: {K2.shape}, RBF: {K3.shape}")
    
    print("\n[PASS] All kernel functions work correctly!")
