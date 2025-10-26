import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class KNNClassifier:
    """
    K-Nearest Neighbors classifier with automatic K selection.
    
    Features:
        - Distance-weighted voting (inverse squared distance)
        - Automatic K selection via validation set
        - Euclidean distance metric
    
    Attributes:
        X (np.ndarray): Training features
        y (np.ndarray): Training labels
        k (int): Number of neighbors (auto-selected)
    """

    def __init__(self):
        self.X = None
        self.y = None
        self.k = None

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Fit KNN classifier and find optimal K using validation set.
        
        Args:
            X_train (np.ndarray): Training features, shape (n_samples, n_features)
            y_train (np.ndarray): Training labels, shape (n_samples,)
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation labels
            k_range (range): Range of K values to search (default: 1 to 2√n)
        """
        self.X = X_train
        self.y = y_train
        num_samples, dim = X_train.shape
        best_score = 0
        # Default K search range: 1 to 2√n
        for k in range(1, int(2*np.sqrt(num_samples))):
            self.k = k
            score = self.score(X_val, y_val)    
            if score > best_score:
                best_score = score
                best_k = k
        self.k = best_k
        print(f"\n✓ Optimal K: {best_k} with validation accuracy {best_score:.4f}")

    def score(self, X, y):
        """
        Compute accuracy on given data.        
        Returns:
            float: Accuracy score
        """
        output = self.predict(X)
        score = (output==y).sum() / len(X)
        return score

    def predict(self, X):
        """
        Predict labels for test data.
        
        Args:
            X (np.ndarray): Test features, shape (n_samples, n_features)
        
        Returns:
            np.ndarray: Predicted labels, shape (n_samples,)
        
        Algorithm:
            1. Compute distances to all training points
            2. Find K nearest neighbors
            3. Weighted voting (inverse squared distance)
            4. Return class with highest weighted vote
        """
        predictions = np.zeros(len(X))
        for i, x in enumerate(X):
            # Step 1: Compute Euclidean distances to all training points
            dists = np.linalg.norm(self.X - x, axis=1)
            indices = np.argsort(dists)
            # Step 2: Find K nearest neighbors
            k_labels = self.y[indices[:self.k]]
            k_dists = dists[indices[:self.k]]
            # Step 3: Distance-weighted voting
            # Weight = 1 / distance² (closer neighbors have more influence)
            k_weights = 1 / (k_dists+1e-9)**2
            # Accumulate weighted votes for each class
            weighted_votes = {}
            for cls in np.unique(k_labels):
                cls_score = k_weights[k_labels==cls].sum()
                weighted_votes[cls] = cls_score
            # Step 4: Return class with highest weighted vote
            predictions[i] = max(weighted_votes, key=weighted_votes.get) 
        return predictions
    
class Normalizer:
    def __init__(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        
    def __call__(self, X):
        return (X - self.mean) / self.std
        
def prepare_classification_dataset():
    """
    Generate and prepare synthetic classification dataset.
    
    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    X, y = make_classification(
        n_samples=500,
        n_features=2,  # 2D for visualization
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        n_classes=3,
        random_state=42
    )
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Normalize input features
    normalizer = Normalizer(X_train)
    X_train = normalizer(X_train)
    X_val = normalizer(X_val)
    X_test = normalizer(X_test)

    print(f"\nDataset sizes:")
    print(f"  Train: {X_train.shape[0]}")
    print(f"  Val: {X_val.shape[0]}")
    print(f"  Test: {X_test.shape[0]}")
    return X_train, y_train, X_val, y_val, X_test, y_test

def plot_decision_boundary(X, y, model, title="KNN Decision Boundary"):
    """
    Visualize decision boundary for 2D data.
    
    Args:
        X (np.ndarray): Features, shape (n_samples, 2)
        y (np.ndarray): Labels
        model: Trained KNN model
        title (str): Plot title
    """
    # Create mesh grid
    h = 0.02  # Step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict for each point in mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', 
                edgecolors='black', linewidth=1, s=50)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.colorbar()
    plt.show()

def test_knn_classification():
    """Test KNN classifier on synthetic data."""
    print("="*70)
    print("KNN CLASSIFICATION TEST")
    print("="*70)
    # Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_classification_dataset()
    # Train KNN
    print("\nTraining KNN classifier...")
    knn = KNNClassifier()
    knn.fit(X_train, y_train, X_val, y_val)
    
    # Evaluate on all splits
    print("\nEvaluating model...")
    train_score = knn.score(X_train, y_train)
    val_score = knn.score(X_val, y_val)
    test_score = knn.score(X_test, y_test)
    print(f"\nResults:")
    print(f"  Train Accuracy: {train_score:.4f}")
    print(f"  Val Accuracy:   {val_score:.4f}")
    print(f"  Test Accuracy:  {test_score:.4f}")
    
    # Visualize decision boundary
    plot_decision_boundary(X_train, y_train, knn)

if __name__=="__main__":
    test_knn_classification()