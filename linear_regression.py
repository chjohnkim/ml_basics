import numpy as np
import matplotlib.pyplot as plt

"""
DERIVATION OF NORMAL EQUATION

Goal: Find w that minimizes the loss L(w) = ||y - Xw||²

Step 1: Expand the loss function
-------------------------------
L(w) = ||y - Xw||²
     = (y - Xw)ᵀ(y - Xw)
     = yᵀy - yᵀXw - wᵀXᵀy + wᵀXᵀXw
     = yᵀy - 2wᵀXᵀy + wᵀXᵀXw
     
Note: yᵀXw = wᵀXᵀy (both are scalars, so they're equal)

Step 2: Take derivative with respect to w
-----------------------------------------
∂L/∂w = ∂/∂w [yᵀy - 2wᵀXᵀy + wᵀXᵀXw]
      = 0 - 2Xᵀy + 2XᵀXw

Key matrix calculus rules used:
- ∂/∂w (wᵀa) = a
- ∂/∂w (wᵀAw) = 2Aw (when A is symmetric, and XᵀX is symmetric)

Step 3: Set derivative to zero (optimality condition)
----------------------------------------------------
∂L/∂w = 0
-2Xᵀy + 2XᵀXw = 0
2XᵀXw = 2Xᵀy
XᵀXw = Xᵀy

Step 4: Solve for w
------------------
XᵀXw = Xᵀy
w = (XᵀX)⁻¹Xᵀy  ✓

This is the NORMAL EQUATION (closed-form solution)
"""

class LinearRegression:

    def __init__(self, fit_intercept):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        """
        Fit using normal equation: w = (X^T X)^(-1) X^T y
        
        X: (n_samples, n_features)
        y: (n_samples,)
        """
        
        if self.fit_intercept:
            X = np.column_stack([np.ones(len(X)), X])
        self.weights = np.linalg.inv(X.T@X)@X.T@y

    def predict(self, X):
        if self.fit_intercept:
            X = np.column_stack([np.ones(len(X)), X])
        return X@self.weights

def plot_regression(X, y, model):
    """
    Plot regression results for multi-feature case
    """
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    n_features = X.shape[1]
    
    fig, axes = plt.subplots(1, n_features + 1, figsize=(5*(n_features+1), 4))
    
    # Plot each feature vs target
    for i in range(n_features):
        axes[i].scatter(X[:, i], y, alpha=0.6, label='Actual')
        axes[i].scatter(X[:, i], y_pred, alpha=0.6, label='Predicted')
        axes[i].set_xlabel(f'Feature {i+1}')
        axes[i].set_ylabel('y')
        axes[i].set_title(f'Feature {i+1} vs Target')
        axes[i].legend()
        axes[i].grid(alpha=0.3)
    
    # Residual plot
    axes[n_features].scatter(y_pred, residuals, alpha=0.6)
    axes[n_features].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[n_features].set_xlabel('Predicted')
    axes[n_features].set_ylabel('Residuals')
    axes[n_features].set_title('Residual Plot')
    axes[n_features].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    X = np.random.randn(100, 2)
    true_weights = np.array([2.0, -1.5])
    y = X @ true_weights + 1.0 + np.random.randn(100) * 0.1
    
    # Fit model
    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate MSE
    mse = np.mean((y - y_pred)**2)
    print(f"Learned weights: {model.weights}")
    print(f"MSE: {mse:.4f}")
    plot_regression(X, y, model)
