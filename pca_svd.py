import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None 
    
    def fit(self, X):
        # Step 1: Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        # Step 2: Compute SVD (more stable than eigendecomposition)
        # X_centered = U @ Σ @ V^T
        U, S, Vt = np.linalg.svd(X_centered)
        # Step 3: Principal components are rows of Vt (columns of V)
        self.components = Vt[:self.n_components]
        # Step 4: Singular values and explained variance
        self.explained_variance = (S**2) / (X.shape[0] - 1)
        self.explained_variance = self.explained_variance[:self.n_components]

    def transform(self, X):
        X_centered = X - self.mean
        # Project onto principal components
        return X_centered @ self.components.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        # Reconstruct original data from reduced representation
        return X @ self.components + self.mean

    def explained_variance_ratio(self):
        # Get proportion of variance explained by each component
        total_variance = np.sum(self.explained_variance)
        return self.explained_variance / total_variance

        

def test_pca(X, y, labels=None):
    """
    Visualize PCA results
    """
    # Fit PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    # Get variance explained
    var_ratio = pca.explained_variance_ratio()
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: 2D projection
    ax = axes[0]
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', 
                        alpha=0.7, edgecolors='k', s=50)
    ax.set_xlabel(f'PC1 ({var_ratio[0]*100:.1f}% variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({var_ratio[1]*100:.1f}% variance)', fontsize=12)
    ax.set_title('PCA Projection (2D)', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Class')
    
    # Plot 2: Scree plot (variance explained)
    ax = axes[1]
    n_features = X.shape[1]
    pca_full = PCA(n_components=n_features)
    pca_full.fit(X)
    var_ratio_full = pca_full.explained_variance_ratio()
    cumsum_var = np.cumsum(var_ratio_full)
    
    ax.bar(range(1, n_features + 1), var_ratio_full, alpha=0.7, 
           label='Individual', edgecolor='k')
    ax.plot(range(1, n_features + 1), cumsum_var, 'ro-', linewidth=2, 
            markersize=8, label='Cumulative')
    ax.axhline(y=0.95, color='g', linestyle='--', linewidth=2, label='95% threshold')
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Variance Explained', fontsize=12)
    ax.set_title('Scree Plot', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: Reconstruction error vs components
    ax = axes[2]
    errors = []
    components_range = range(1, n_features + 1)
    
    for k in components_range:
        pca_k = PCA(n_components=k)
        X_reduced = pca_k.fit_transform(X)
        X_reconstructed = pca_k.inverse_transform(X_reduced)
        error = np.mean((X - X_reconstructed) ** 2)
        errors.append(error)
    
    ax.plot(components_range, errors, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Components', fontsize=12)
    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.set_title('Reconstruction Error', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("="*60)
    print("PCA SUMMARY")
    print("="*60)
    print(f"Original dimensions: {X.shape[1]}")
    print(f"Reduced dimensions: 2")
    print(f"\nVariance explained by PC1: {var_ratio[0]*100:.2f}%")
    print(f"Variance explained by PC2: {var_ratio[1]*100:.2f}%")
    print(f"Total variance (2 PCs): {sum(var_ratio)*100:.2f}%")
    print(f"\nComponents needed for 95% variance: {np.argmax(cumsum_var >= 0.95) + 1}")
    print("="*60)


# Example usage
if __name__ == "__main__":
    # Load iris dataset
    iris = load_iris()
    X = iris.data  # 4 features
    y = iris.target  # 3 classes
    
    print("="*60)
    print("PCA ON IRIS DATASET")
    print("="*60)
    print(f"Data shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Visualize PCA
    test_pca(X, y, labels=iris.target_names)
        
    # Test reconstruction
    print("\n" + "="*60)
    print("TESTING RECONSTRUCTION")
    print("="*60)
    
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_reduced)
    
    reconstruction_error = np.mean((X - X_reconstructed) ** 2)
    print(f"Reconstruction error (2 components): {reconstruction_error:.6f}")
    print(f"Relative error: {reconstruction_error / np.var(X):.2%}")
    
    print("\n✓ PCA implementation complete!")
    print("✓ Saved 'pca_analysis.png'")