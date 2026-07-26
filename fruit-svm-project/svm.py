import numpy as np

class LinearSVM:
    """Soft-margin linear SVM using gradient descent on the hinge loss."""
    def __init__(self, C=1.0, learning_rate=0.01, epochs=1000):
        self.C = C
        self.lr = learning_rate
        self.epochs = epochs
        self.theta = None
        self.b = 0
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.theta = np.zeros(n_features)
        self.b = 0
        
        for _ in range(self.epochs):
            margins = y * (np.dot(X, self.theta) + self.b)
            misclassified = margins < 1
            
            d_theta = self.theta.copy()
            d_b = 0
            
            if np.any(misclassified):
                X_mis = X[misclassified]
                y_mis = y[misclassified]
                d_theta -= self.C * np.dot(y_mis, X_mis)
                d_b -= self.C * np.sum(y_mis)
            
            self.theta -= self.lr * d_theta
            self.b -= self.lr * d_b
            
    def decision_function(self, X):
        return np.dot(X, self.theta) + self.b
        
    def predict(self, X):
        return np.sign(self.decision_function(X))


class KernelSVM:
    """Kernelized SVM using stochastic/batch gradient descent (Pegasos style)."""
    def __init__(self, kernel_func, C=1.0, epochs=1000):
        self.kernel_func = kernel_func
        self.C = C
        self.epochs = epochs
        self.alpha = None
        self.X_train = None
        self.y_train = None
        self.b = 0
        
    def fit(self, X, y):
        n_samples = X.shape[0]
        self.alpha = np.zeros(n_samples)
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.b = 0
        
        # Precompute kernel matrix for faster training
        K = self.kernel_func(X, X)
        
        lr = 0.01
        for _ in range(self.epochs):
            margins = y * (np.dot(K, self.alpha * y) + self.b)
            misclassified = margins < 1
            
            d_alpha = self.alpha.copy() 
            d_b = 0
            
            if np.any(misclassified):
                d_alpha[misclassified] -= self.C
                d_b -= self.C * np.sum(y[misclassified])
            
            self.alpha -= lr * d_alpha
            self.alpha = np.maximum(self.alpha, 0)
            self.b -= lr * d_b

    def decision_function(self, X):
        K_test = self.kernel_func(X, self.X_train)
        return np.dot(K_test, self.alpha * self.y_train) + self.b
        
    def predict(self, X):
        return np.sign(self.decision_function(X))


class OneVsRestSVM:
    """One-vs-Rest multiclass classifier wrapper for Binary SVMs."""
    def __init__(self, base_classifier_class, **kwargs):
        self.base_classifier_class = base_classifier_class
        self.kwargs = kwargs
        self.classifiers = []
        self.classes_ = None
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.classifiers = []
        
        for c in self.classes_:
            # Binary labels: +1 for class c, -1 for others
            y_binary = np.where(y == c, 1, -1)
            clf = self.base_classifier_class(**self.kwargs)
            clf.fit(X, y_binary)
            self.classifiers.append(clf)
            
    def decision_function(self, X):
        scores = np.zeros((X.shape[0], len(self.classes_)))
        for i, clf in enumerate(self.classifiers):
            scores[:, i] = clf.decision_function(X)
        return scores
        
    def predict(self, X):
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]
        
    def predict_proba(self, X):
        # Convert decision scores to probabilities using Softmax over OvR scores
        scores = self.decision_function(X)
        # Subtract max for numerical stability
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs
