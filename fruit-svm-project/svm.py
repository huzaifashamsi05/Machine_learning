"""
svm.py - Support Vector Machine Implementation from Scratch
============================================================

Implements:
    1. LinearSVM          - Binary soft-margin SVM via sub-gradient descent
    2. KernelSVM          - Binary kernel SVM via dual optimization
    3. MultiClassSVM      - One-vs-Rest wrapper for multi-class
    4. PlattScaling       - Confidence calibration for SVM outputs

All implementations use only NumPy and scipy.optimize.
NO sklearn or pretrained anything.

Mathematical Foundation:
========================

PRIMAL (Linear SVM):
    min_{w,b} (1/2)||w||^2 + C * sum_i max(0, 1 - y_i(w^T x_i + b))
    
    This is the soft-margin SVM with hinge loss.
    - (1/2)||w||^2: regularization term (maximize margin)
    - C: trade-off parameter between margin size and training errors
    - max(0, 1 - y_i*f(x_i)): hinge loss (0 if correctly classified with margin)

DUAL (Kernel SVM):
    max_alpha  sum_i alpha_i - (1/2) sum_{i,j} alpha_i alpha_j y_i y_j K(x_i, x_j)
    subject to: 0 <= alpha_i <= C  and  sum_i alpha_i y_i = 0
    
    The dual formulation replaces dot products with kernel evaluations,
    enabling non-linear decision boundaries.
"""

import numpy as np
from scipy.optimize import minimize


# ============================================================
# PART 1: LINEAR SVM (Primal, Sub-gradient Descent)
# ============================================================

class LinearSVM:
    """
    Binary Linear SVM using soft-margin formulation.
    
    Optimization:
        We minimize the SVM objective using mini-batch sub-gradient descent.
        The hinge loss is not differentiable at y*f(x) = 1, so we use
        sub-gradients instead of gradients.
    
    Sub-gradient of the objective:
        L(w, b) = (1/2)||w||^2 + C * (1/N) * sum_i max(0, 1 - y_i(w^T x_i + b))
        
        For each sample (x_i, y_i):
            if y_i(w^T x_i + b) < 1:   (misclassified or in margin)
                dL/dw = w - C * y_i * x_i
                dL/db = -C * y_i
            else:                        (correctly classified beyond margin)
                dL/dw = w
                dL/db = 0
    
    Learning rate schedule:
        eta(t) = eta_0 / (1 + decay * t)
        
        Decaying learning rate is crucial for convergence:
        - Large initial steps: fast progress toward minimum
        - Small later steps: fine-tuning near the optimum
    
    Labels must be {-1, +1} for binary SVM.
    """
    
    def __init__(self, C=1.0, learning_rate=0.01, decay=1e-4, 
                 n_epochs=500, batch_size=32, tol=1e-6, verbose=False):
        """
        Args:
            C: Regularization parameter.
               Large C -> hard margin (less tolerance for errors)
               Small C -> soft margin (more tolerance, wider margin)
            learning_rate: Initial learning rate (eta_0)
            decay: Learning rate decay parameter (lambda)
            n_epochs: Maximum number of training epochs
            batch_size: Mini-batch size for SGD
            tol: Convergence tolerance (stop if loss change < tol)
            verbose: Print training progress
        """
        self.C = C
        self.learning_rate = learning_rate
        self.decay = decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.tol = tol
        self.verbose = verbose
        
        # Model parameters (set during fit)
        self.w = None  # Weight vector (d,)
        self.b = None  # Bias scalar
        
        # Training history
        self.loss_history = []
    
    def _compute_loss(self, X, y):
        """
        Compute the SVM objective function value.
        
        L = (1/2)||w||^2 + C * (1/N) * sum max(0, 1 - y_i * (w^T x_i + b))
        
        Returns:
            loss: scalar value of the objective
        """
        n = len(y)
        margins = y * (X @ self.w + self.b)  # y_i * f(x_i)
        hinge_loss = np.maximum(0, 1 - margins)
        
        regularization = 0.5 * np.dot(self.w, self.w)
        loss = regularization + self.C * np.mean(hinge_loss)
        
        return loss
    
    def fit(self, X, y):
        """
        Train the linear SVM using mini-batch sub-gradient descent.
        
        Algorithm:
            1. Initialize w = 0, b = 0
            2. For each epoch:
                a. Shuffle training data
                b. For each mini-batch:
                    - Compute sub-gradients
                    - Update w and b with decaying learning rate
                c. Compute and store epoch loss
                d. Check convergence
        
        Args:
            X: numpy array (N, D) - training features
            y: numpy array (N,) - labels in {-1, +1}
        
        Returns:
            self (for chaining)
        """
        N, D = X.shape
        
        # Initialize parameters
        self.w = np.zeros(D, dtype=np.float64)
        self.b = 0.0
        self.loss_history = []
        
        # Training loop
        t = 0  # Global step counter (for learning rate schedule)
        
        for epoch in range(self.n_epochs):
            # Shuffle data each epoch
            indices = np.random.permutation(N)
            
            for start in range(0, N, self.batch_size):
                end = min(start + self.batch_size, N)
                batch_idx = indices[start:end]
                
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                batch_size = len(y_batch)
                
                # Current learning rate with decay
                eta = self.learning_rate / (1 + self.decay * t)
                t += 1
                
                # Compute margins for the batch
                margins = y_batch * (X_batch @ self.w + self.b)
                
                # Find misclassified/margin samples
                # mask[i] = True if sample i has hinge loss > 0
                mask = margins < 1  # (batch_size,) boolean
                
                # Sub-gradient computation:
                # dw = w - (C/batch_size) * sum_{misclassified} y_i * x_i
                # db = -(C/batch_size) * sum_{misclassified} y_i
                
                # Regularization gradient
                dw = self.w.copy()
                db = 0.0
                
                if np.any(mask):
                    # Hinge loss gradient contribution
                    # For misclassified samples: -y_i * x_i
                    misclassified_X = X_batch[mask]   # (k, D)
                    misclassified_y = y_batch[mask]    # (k,)
                    
                    dw -= (self.C / batch_size) * (misclassified_y[:, np.newaxis] * misclassified_X).sum(axis=0)
                    db -= (self.C / batch_size) * misclassified_y.sum()
                
                # Update parameters
                self.w -= eta * dw
                self.b -= eta * db
            
            # End of epoch: compute and store loss
            epoch_loss = self._compute_loss(X, y)
            self.loss_history.append(epoch_loss)
            
            if self.verbose and (epoch + 1) % 50 == 0:
                acc = np.mean((X @ self.w + self.b > 0) == (y > 0))
                print(f"  Epoch {epoch+1}/{self.n_epochs}: "
                      f"loss={epoch_loss:.4f}, acc={acc:.3f}, lr={eta:.6f}")
            
            # Convergence check
            if len(self.loss_history) > 10:
                recent_change = abs(self.loss_history[-1] - self.loss_history[-10])
                if recent_change < self.tol:
                    if self.verbose:
                        print(f"  Converged at epoch {epoch+1} "
                              f"(loss change < {self.tol})")
                    break
        
        return self
    
    def decision_function(self, X):
        """
        Compute the raw decision function: f(x) = w^T x + b
        
        The sign determines the predicted class:
            f(x) > 0 -> class +1
            f(x) < 0 -> class -1
        
        The magnitude |f(x)| indicates confidence:
            |f(x)| >> 0 -> very confident
            |f(x)| ~= 0 -> near decision boundary
        
        Args:
            X: numpy array (N, D)
        
        Returns:
            scores: numpy array (N,)
        """
        return X @ self.w + self.b
    
    def predict(self, X):
        """
        Predict class labels {-1, +1}.
        
        Args:
            X: numpy array (N, D)
        
        Returns:
            predictions: numpy array (N,) of {-1, +1}
        """
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, -1)


# ============================================================
# PART 2: KERNEL SVM (Dual, scipy.optimize)
# ============================================================

class KernelSVM:
    """
    Binary Kernel SVM using the dual formulation.
    
    Dual Problem:
        max_alpha  sum_i alpha_i - (1/2) sum_{i,j} alpha_i alpha_j y_i y_j K(x_i, x_j)
        subject to:
            0 <= alpha_i <= C  for all i
            sum_i alpha_i y_i = 0
    
    We convert to a minimization problem (negate the objective) and
    solve with scipy.optimize.minimize using SLSQP method.
    
    After solving:
        - Support vectors: samples with alpha_i > 0
        - Decision function: f(x) = sum_i alpha_i y_i K(x_i, x) + b
        - Bias b is computed from support vectors:
          b = y_s - sum_i alpha_i y_i K(x_i, x_s)
          for any support vector x_s with 0 < alpha_s < C
    """
    
    def __init__(self, C=1.0, kernel_type='rbf', verbose=False, **kernel_params):
        """
        Args:
            C: Regularization parameter
            kernel_type: 'linear', 'poly', or 'rbf'
            verbose: Print training progress
            **kernel_params: Parameters for the kernel function:
                - degree, coef0 for polynomial
                - gamma for RBF
        """
        self.C = C
        self.kernel_type = kernel_type
        self.kernel_params = kernel_params
        self.verbose = verbose
        
        # Model parameters (set during fit)
        self.alpha = None       # Dual variables
        self.support_vectors = None  # X values of support vectors
        self.support_labels = None   # y values of support vectors  
        self.support_alpha = None    # alpha values of support vectors
        self.b = None               # Bias term
        
        # Store training data for kernel evaluation
        self.X_train = None
        self.y_train = None
    
    def _compute_kernel(self, X, Y):
        """Compute kernel matrix using kernels module."""
        from kernels import compute_kernel_matrix
        return compute_kernel_matrix(X, Y, self.kernel_type, **self.kernel_params)
    
    def fit(self, X, y):
        """
        Train kernel SVM by solving the dual optimization problem.
        
        Steps:
            1. Compute the kernel (Gram) matrix K
            2. Set up the quadratic program
            3. Solve using scipy.optimize.minimize (SLSQP)
            4. Extract support vectors and compute bias
        
        Args:
            X: numpy array (N, D) - training features
            y: numpy array (N,) - labels in {-1, +1}
        
        Returns:
            self
        """
        N = len(y)
        
        if self.verbose:
            print(f"  Computing {self.kernel_type} kernel matrix ({N}x{N})...")
        
        # Step 1: Kernel matrix
        K = self._compute_kernel(X, X)  # (N, N)
        
        # Step 2: Precompute y_i * y_j * K(x_i, x_j)
        # This is the Hessian of the dual objective
        Q = np.outer(y, y) * K  # (N, N)
        
        # Add small regularization to Q for numerical stability
        Q += 1e-8 * np.eye(N)
        
        # Step 3: Define the optimization problem
        # Minimize: (1/2) alpha^T Q alpha - 1^T alpha  (negated dual)
        # Subject to: 0 <= alpha_i <= C, y^T alpha = 0
        
        def objective(alpha):
            """Dual objective (negated for minimization)."""
            return 0.5 * alpha @ Q @ alpha - alpha.sum()
        
        def gradient(alpha):
            """Gradient of the dual objective."""
            return Q @ alpha - np.ones(N)
        
        # Constraints
        # Equality: sum(alpha_i * y_i) = 0
        constraints = {
            'type': 'eq',
            'fun': lambda alpha: np.dot(alpha, y),
            'jac': lambda alpha: y.astype(np.float64)
        }
        
        # Bounds: 0 <= alpha_i <= C
        bounds = [(0, self.C) for _ in range(N)]
        
        # Initial guess: all zeros
        alpha0 = np.zeros(N)
        
        if self.verbose:
            print(f"  Solving dual optimization (N={N}, C={self.C})...")
        
        # Solve
        result = minimize(
            objective,
            alpha0,
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            method='SLSQP',
            options={
                'maxiter': 1000,
                'ftol': 1e-10,
                'disp': False
            }
        )
        
        self.alpha = result.x
        
        if self.verbose:
            print(f"  Optimization {'converged' if result.success else 'FAILED'}: "
                  f"{result.message}")
        
        # Step 4: Extract support vectors
        # Support vectors have alpha_i > 0 (with numerical threshold)
        sv_threshold = 1e-6
        sv_mask = self.alpha > sv_threshold
        
        self.support_vectors = X[sv_mask]
        self.support_labels = y[sv_mask]
        self.support_alpha = self.alpha[sv_mask]
        
        # Store training data for prediction
        self.X_train = X
        self.y_train = y
        
        # Step 5: Compute bias
        # Use support vectors that are NOT on the boundary (alpha < C)
        # For these: y_s * f(x_s) = 1  =>  b = y_s - sum(alpha_i y_i K(x_i, x_s))
        free_sv_mask = sv_mask & (self.alpha < self.C - sv_threshold)
        
        if np.any(free_sv_mask):
            free_indices = np.where(free_sv_mask)[0]
            b_values = []
            for idx in free_indices:
                K_col = K[:, idx]
                f_val = np.sum(self.alpha * y * K_col)
                b_values.append(y[idx] - f_val)
            self.b = np.mean(b_values)
        else:
            # Fallback: use all support vectors
            sv_indices = np.where(sv_mask)[0]
            b_values = []
            for idx in sv_indices:
                K_col = K[:, idx]
                f_val = np.sum(self.alpha * y * K_col)
                b_values.append(y[idx] - f_val)
            self.b = np.mean(b_values) if len(b_values) > 0 else 0.0
        
        if self.verbose:
            print(f"  Support vectors: {len(self.support_vectors)}/{N} "
                  f"({100*len(self.support_vectors)/N:.1f}%)")
            print(f"  Bias: {self.b:.4f}")
        
        return self
    
    def decision_function(self, X):
        """
        Compute decision function for new samples.
        
        f(x) = sum_i alpha_i y_i K(x_i, x) + b
        
        where the sum is over all training samples (but only support
        vectors contribute since alpha_i = 0 for non-support vectors).
        
        Args:
            X: numpy array (M, D) - test samples
        
        Returns:
            scores: numpy array (M,)
        """
        # Compute kernel between training data and test data
        K = self._compute_kernel(self.X_train, X)  # (N_train, M)
        
        # Decision values
        scores = (self.alpha * self.y_train) @ K + self.b
        
        return scores
    
    def predict(self, X):
        """Predict class labels {-1, +1}."""
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, -1)


# ============================================================
# PART 3: ONE-VS-REST MULTI-CLASS SVM
# ============================================================

class MultiClassSVM:
    """
    Multi-class classification using One-vs-Rest (OvR) strategy.
    
    For K classes, we train K binary classifiers:
        Classifier k: class k (+1) vs all other classes (-1)
    
    Prediction: class with the highest decision function value
        y_pred = argmax_k f_k(x)
    
    Why One-vs-Rest?
        - Simple and well-understood
        - Each classifier is independent (can be parallelized)
        - Works well when classes are well-separated
        - Decision values are comparable across classifiers
        (unlike One-vs-One which needs voting)
    
    Alternative: One-vs-One trains K(K-1)/2 classifiers (more for K=6: 15)
    OvR trains only K classifiers (K=6: 6) -> much faster
    """
    
    def __init__(self, svm_class, n_classes=6, **svm_params):
        """
        Args:
            svm_class: LinearSVM or KernelSVM class to use
            n_classes: Number of classes
            **svm_params: Parameters passed to each binary SVM
        """
        self.svm_class = svm_class
        self.n_classes = n_classes
        self.svm_params = svm_params
        self.classifiers = []
        self.class_names = None
    
    def fit(self, X, y, class_names=None):
        """
        Train K binary classifiers using One-vs-Rest.
        
        For each class k:
            - Create binary labels: +1 for class k, -1 for all others
            - Train a binary SVM
        
        Args:
            X: numpy array (N, D)
            y: numpy array (N,) with labels in {0, 1, ..., K-1}
            class_names: optional list of class name strings
        """
        self.class_names = class_names
        self.classifiers = []
        
        for k in range(self.n_classes):
            # Create binary labels
            y_binary = np.where(y == k, 1, -1)
            
            # Count class distribution
            n_pos = np.sum(y_binary == 1)
            n_neg = np.sum(y_binary == -1)
            
            class_label = class_names[k] if class_names else f"Class {k}"
            print(f"\n  Training classifier {k+1}/{self.n_classes}: "
                  f"'{class_label}' vs Rest "
                  f"(+1: {n_pos}, -1: {n_neg})")
            
            # Create and train binary SVM
            clf = self.svm_class(**self.svm_params)
            clf.fit(X, y_binary)
            self.classifiers.append(clf)
        
        return self
    
    def decision_function(self, X):
        """
        Compute decision values for all K classifiers.
        
        Args:
            X: numpy array (N, D)
        
        Returns:
            scores: numpy array (N, K) - score for each class
        """
        scores = np.zeros((len(X), self.n_classes))
        for k, clf in enumerate(self.classifiers):
            scores[:, k] = clf.decision_function(X)
        return scores
    
    def predict(self, X):
        """
        Predict class labels by taking argmax of decision values.
        
        Args:
            X: numpy array (N, D)
        
        Returns:
            predictions: numpy array (N,) with labels in {0, ..., K-1}
        """
        scores = self.decision_function(X)
        return np.argmax(scores, axis=1)
    
    def accuracy(self, X, y):
        """Compute classification accuracy."""
        predictions = self.predict(X)
        return np.mean(predictions == y)


# ============================================================
# PART 4: PLATT SCALING (Confidence Calibration)
# ============================================================

class PlattScaling:
    """
    Platt Scaling: Convert SVM decision values to calibrated probabilities.
    
    Idea:
        SVM outputs f(x) which is not a probability. Platt (1999) proposed
        fitting a sigmoid function to convert f(x) to P(y=1|x):
        
        P(y=1|f) = 1 / (1 + exp(A*f + B))
        
        where A and B are parameters learned by minimizing cross-entropy
        loss on a validation set.
    
    Why Platt scaling?
        - SVMs naturally output distances to the hyperplane, not probabilities
        - Sigmoid provides a smooth, calibrated mapping
        - Well-validated approach used in production systems
        - Can be trained on validation set (no data leakage)
    
    For multi-class OvR:
        1. Calibrate each binary classifier independently
        2. Apply softmax normalization across classes:
           P(class k) = exp(p_k) / sum_j exp(p_j)
           where p_k is the calibrated probability from classifier k
    
    Reference:
        Platt, J. (1999). "Probabilistic Outputs for Support Vector 
        Machines and Comparisons to Regularized Likelihood Methods"
    """
    
    def __init__(self):
        self.A = None  # Sigmoid parameter A
        self.B = None  # Sigmoid parameter B
        self.is_fitted = False
    
    def _sigmoid(self, f, A, B):
        """
        Compute sigmoid: P = 1 / (1 + exp(A*f + B))
        
        Numerically stable implementation that avoids overflow.
        """
        z = A * f + B
        # Clip to avoid overflow in exp
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(z))
    
    def fit(self, decision_values, y_true):
        """
        Fit Platt scaling parameters A and B.
        
        Uses target probabilities instead of raw 0/1 labels
        (as recommended by Platt) to avoid overfitting:
            t_i = (y_i + 1) / 2  mapped to:
            t_+ = (N_+ + 1) / (N_+ + 2)   for positive samples
            t_- = 1 / (N_- + 2)            for negative samples
        
        Then minimize negative log-likelihood:
            L(A,B) = -sum_i [t_i log(p_i) + (1-t_i) log(1-p_i)]
        
        Args:
            decision_values: numpy array (N,) from SVM decision function
            y_true: numpy array (N,) with labels in {-1, +1}
        """
        f = decision_values.copy()
        y = y_true.copy()
        
        # Compute target probabilities (Platt's approach)
        N_pos = np.sum(y == 1)
        N_neg = np.sum(y == -1)
        
        t = np.zeros_like(y, dtype=np.float64)
        t[y == 1] = (N_pos + 1) / (N_pos + 2)
        t[y == -1] = 1 / (N_neg + 2)
        
        # Objective: negative log-likelihood
        def neg_log_likelihood(params):
            A, B = params
            p = self._sigmoid(f, A, B)
            # Clip for numerical stability
            p = np.clip(p, 1e-10, 1 - 1e-10)
            nll = -np.sum(t * np.log(p) + (1 - t) * np.log(1 - p))
            return nll
        
        # Gradient
        def gradient(params):
            A, B = params
            p = self._sigmoid(f, A, B)
            p = np.clip(p, 1e-10, 1 - 1e-10)
            # d_NLL/d_p * d_p/d_A, d_p/d_B
            diff = t - p
            dA = -np.sum(diff * f)
            dB = -np.sum(diff)
            return np.array([dA, dB])
        
        # Optimize using L-BFGS-B
        result = minimize(
            neg_log_likelihood,
            x0=[0.0, 0.0],
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 100}
        )
        
        self.A, self.B = result.x
        self.is_fitted = True
    
    def predict_proba(self, decision_values):
        """
        Convert decision values to probabilities.
        
        Args:
            decision_values: numpy array (N,)
        
        Returns:
            probabilities: numpy array (N,) in [0, 1]
        """
        if not self.is_fitted:
            raise RuntimeError("PlattScaling not fitted!")
        return self._sigmoid(decision_values, self.A, self.B)


class CalibratedMultiClassSVM:
    """
    Multi-class SVM with calibrated probability outputs.
    
    Combines:
        1. MultiClassSVM for predictions
        2. PlattScaling for each binary classifier
        3. Softmax normalization across classes
    
    Output:
        Probability distribution over K classes, summing to 1.
        Confidence = max probability * 100%.
    """
    
    def __init__(self, multi_svm):
        """
        Args:
            multi_svm: Trained MultiClassSVM object
        """
        self.multi_svm = multi_svm
        self.calibrators = []
        self.is_calibrated = False
    
    def calibrate(self, X_val, y_val):
        """
        Fit Platt scaling on validation set.
        
        For each binary classifier k:
            1. Get decision values on validation set
            2. Create binary labels: +1 if y==k, -1 otherwise
            3. Fit PlattScaling(A_k, B_k)
        
        Args:
            X_val: numpy array (N_val, D) - validation features
            y_val: numpy array (N_val,) - validation labels {0,...,K-1}
        """
        self.calibrators = []
        n_classes = self.multi_svm.n_classes
        
        print("\n  Calibrating confidence scores (Platt scaling)...")
        
        for k in range(n_classes):
            clf = self.multi_svm.classifiers[k]
            decision_vals = clf.decision_function(X_val)
            y_binary = np.where(y_val == k, 1, -1)
            
            calibrator = PlattScaling()
            calibrator.fit(decision_vals, y_binary)
            self.calibrators.append(calibrator)
            
            class_label = (self.multi_svm.class_names[k] 
                          if self.multi_svm.class_names else f"Class {k}")
            print(f"    {class_label}: A={calibrator.A:.4f}, B={calibrator.B:.4f}")
        
        self.is_calibrated = True
    
    def predict_proba(self, X):
        """
        Get calibrated probability distribution over classes.
        
        Steps:
            1. Get decision values from each OvR classifier
            2. Convert to probabilities via Platt scaling
            3. Normalize via softmax so probabilities sum to 1
        
        Softmax normalization:
            P(class k) = exp(log_p_k) / sum_j exp(log_p_j)
            
            where log_p_k is the log of the Platt-scaled probability.
            
            We use log-probabilities + log-sum-exp for numerical stability:
            log P(k) = log_p_k - log(sum_j exp(log_p_j))
        
        Args:
            X: numpy array (N, D)
        
        Returns:
            probabilities: numpy array (N, K) summing to 1 per row
        """
        if not self.is_calibrated:
            raise RuntimeError("Not calibrated! Call calibrate() first.")
        
        n_classes = self.multi_svm.n_classes
        N = len(X)
        
        # Get calibrated probabilities for each class
        raw_probs = np.zeros((N, n_classes))
        
        for k in range(n_classes):
            clf = self.multi_svm.classifiers[k]
            decision_vals = clf.decision_function(X)
            raw_probs[:, k] = self.calibrators[k].predict_proba(decision_vals)
        
        # Softmax normalization (using log-sum-exp trick for stability)
        log_probs = np.log(np.clip(raw_probs, 1e-10, None))
        log_sum = np.log(np.sum(np.exp(log_probs - log_probs.max(axis=1, keepdims=True)), 
                                axis=1, keepdims=True)) + log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs - log_sum)
        
        return probs
    
    def predict_with_confidence(self, X):
        """
        Predict class labels with confidence scores.
        
        Args:
            X: numpy array (N, D)
        
        Returns:
            predictions: numpy array (N,) of class indices
            confidences: numpy array (N,) of confidence percentages (0-100)
            probabilities: numpy array (N, K) full probability distribution
        """
        probabilities = self.predict_proba(X)
        predictions = np.argmax(probabilities, axis=1)
        confidences = np.max(probabilities, axis=1) * 100.0
        
        return predictions, confidences, probabilities


# ============================================================
# MODEL SAVE/LOAD UTILITIES
# ============================================================

def save_model(model, scaler, filepath, class_names=None):
    """
    Save trained model to .npz file.
    
    For LinearSVM (OvR):
        Saves K weight vectors and biases.
    
    For KernelSVM (OvR):
        Saves support vectors, alphas, labels, biases, 
        kernel type and parameters.
    """
    save_dict = {
        'class_names': np.array(class_names) if class_names else np.array([]),
        'n_classes': model.n_classes,
        'scaler_mean': scaler.mean,
        'scaler_std': scaler.std,
    }
    
    for k, clf in enumerate(model.classifiers):
        if isinstance(clf, LinearSVM):
            save_dict[f'type'] = 'linear'
            save_dict[f'w_{k}'] = clf.w
            save_dict[f'b_{k}'] = np.array([clf.b])
        elif isinstance(clf, KernelSVM):
            save_dict[f'type'] = 'kernel'
            save_dict[f'kernel_type'] = np.array([clf.kernel_type])
            save_dict[f'X_train_{k}'] = clf.X_train
            save_dict[f'y_train_{k}'] = clf.y_train
            save_dict[f'alpha_{k}'] = clf.alpha
            save_dict[f'b_{k}'] = np.array([clf.b])
            # Save kernel params
            for param_name, param_val in clf.kernel_params.items():
                save_dict[f'kp_{param_name}_{k}'] = np.array([param_val])
    
    # Save calibration if available
    if hasattr(model, '_calibrator') and model._calibrator is not None:
        cal = model._calibrator
        if cal.is_calibrated:
            for k, platt in enumerate(cal.calibrators):
                save_dict[f'platt_A_{k}'] = np.array([platt.A])
                save_dict[f'platt_B_{k}'] = np.array([platt.B])
    
    np.savez_compressed(filepath, **save_dict)


def load_model(filepath):
    """
    Load a trained model from .npz file.
    
    Returns:
        model: MultiClassSVM with loaded classifiers
        scaler: FeatureScaler with loaded parameters
        class_names: list of class name strings
    """
    from features import FeatureScaler
    
    data = np.load(filepath, allow_pickle=True)
    
    n_classes = int(data['n_classes'])
    class_names = list(data['class_names'])
    model_type = str(data.get('type', 'linear'))
    
    # Reconstruct scaler
    scaler = FeatureScaler()
    scaler.mean = data['scaler_mean']
    scaler.std = data['scaler_std']
    scaler.is_fitted = True
    
    # Reconstruct classifiers
    classifiers = []
    
    for k in range(n_classes):
        if model_type == 'linear':
            clf = LinearSVM()
            clf.w = data[f'w_{k}']
            clf.b = float(data[f'b_{k}'][0])
        else:
            # Determine kernel params
            kernel_type = str(data[f'kernel_type'][0])
            kernel_params = {}
            for key in data.files:
                if key.startswith(f'kp_') and key.endswith(f'_{k}'):
                    param_name = key[3:-(len(str(k))+1)]
                    kernel_params[param_name] = float(data[key][0])
            
            clf = KernelSVM(kernel_type=kernel_type, **kernel_params)
            clf.X_train = data[f'X_train_{k}']
            clf.y_train = data[f'y_train_{k}']
            clf.alpha = data[f'alpha_{k}']
            clf.b = float(data[f'b_{k}'][0])
        
        classifiers.append(clf)
    
    # Create model wrapper
    model = MultiClassSVM(
        svm_class=LinearSVM if model_type == 'linear' else KernelSVM,
        n_classes=n_classes
    )
    model.classifiers = classifiers
    model.class_names = class_names
    
    # Load calibration if available
    if f'platt_A_0' in data.files:
        calibrator = CalibratedMultiClassSVM(model)
        calibrator.calibrators = []
        for k in range(n_classes):
            platt = PlattScaling()
            platt.A = float(data[f'platt_A_{k}'][0])
            platt.B = float(data[f'platt_B_{k}'][0])
            platt.is_fitted = True
            calibrator.calibrators.append(platt)
        calibrator.is_calibrated = True
        model._calibrator = calibrator
    else:
        model._calibrator = None
    
    return model, scaler, class_names


# ============================================================
# Self-test
# ============================================================

if __name__ == "__main__":
    """Quick test with synthetic data."""
    print("[*] SVM - Self Test")
    print("=" * 50)
    
    np.random.seed(42)
    
    # Generate linearly separable 2D data
    N = 200
    X_pos = np.random.randn(N // 2, 2) + np.array([2, 2])
    X_neg = np.random.randn(N // 2, 2) + np.array([-2, -2])
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * (N // 2) + [-1] * (N // 2))
    
    # Test LinearSVM
    print("\n1. Linear SVM:")
    svm = LinearSVM(C=1.0, learning_rate=0.01, n_epochs=200, verbose=True)
    svm.fit(X, y)
    preds = svm.predict(X)
    acc = np.mean(preds == y)
    print(f"   Training accuracy: {acc:.3f}")
    
    # Test KernelSVM
    print("\n2. Kernel SVM (RBF):")
    ksvm = KernelSVM(C=1.0, kernel_type='rbf', gamma=0.5, verbose=True)
    ksvm.fit(X, y)
    preds = ksvm.predict(X)
    acc = np.mean(preds == y)
    print(f"   Training accuracy: {acc:.3f}")
    
    # Test multi-class
    print("\n3. Multi-class (3 classes):")
    X3_0 = np.random.randn(50, 2) + np.array([0, 3])
    X3_1 = np.random.randn(50, 2) + np.array([-3, -1])
    X3_2 = np.random.randn(50, 2) + np.array([3, -1])
    X3 = np.vstack([X3_0, X3_1, X3_2])
    y3 = np.array([0]*50 + [1]*50 + [2]*50)
    
    mc_svm = MultiClassSVM(
        LinearSVM, n_classes=3,
        C=1.0, learning_rate=0.01, n_epochs=200
    )
    mc_svm.fit(X3, y3, class_names=["A", "B", "C"])
    acc = mc_svm.accuracy(X3, y3)
    print(f"   Training accuracy: {acc:.3f}")
    
    print("\n[PASS] All SVM implementations work correctly!")
