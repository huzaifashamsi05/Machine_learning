import numpy as np

def polynomial_kernel(x, z, c=1.0, d=2.0):
    """
    Computes the Polynomial kernel between x and z.
    K(x, z) = (x^T z + c)^d
    Supports broadcast operations if x and z are matrices.
    """
    return (np.dot(x, z.T) + c) ** d

def rbf_kernel(x, z, sigma=1.0):
    """
    Computes the RBF (Gaussian) kernel between x and z.
    K(x, z) = exp(-||x - z||^2 / (2 * sigma^2))
    Supports broadcast operations if x and z are matrices.
    """
    # ||x - z||^2 = ||x||^2 + ||z||^2 - 2x^T z
    x = np.atleast_2d(x)
    z = np.atleast_2d(z)
    
    x_norm = np.sum(x ** 2, axis=-1)[:, np.newaxis]
    z_norm = np.sum(z ** 2, axis=-1)[np.newaxis, :]
    
    # Clip negative values that might occur due to floating point inaccuracies
    dist_sq = np.maximum(x_norm + z_norm - 2 * np.dot(x, z.T), 0.0)
    
    res = np.exp(-dist_sq / (2 * sigma**2))
    
    # if both were 1D, return scalar
    if res.shape == (1, 1):
        return res[0, 0]
    return res

def rbf_k(x, z):
    return rbf_kernel(x, z, sigma=5.0)

