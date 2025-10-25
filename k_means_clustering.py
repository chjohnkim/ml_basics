import numpy as np
import matplotlib.pyplot as plt 


class KMeans:
    """
    K-Means clustering algorithm implementation.
    
    Partitions data into K clusters by iteratively:
    1. Assigning points to nearest centroid
    2. Updating centroids as cluster means    
    """

    def __init__(self, n_clusters, threshold=0.0001, max_iter=10000):
        """
        Initialize K-Means clustering model.
        
        Args:
            n_clusters (int): Number of clusters to form
            threshold (float): Convergence threshold - algorithm stops when
                             max centroid movement is below this value
            max_iter (int): Maximum number of iterations to run
        """
        self.k = n_clusters
        self.threshold = threshold
        self.max_iter = max_iter

    def _kmeans_plusplus_initialization(self, x):
        """
        Initialize centroids using K-means++ algorithm for better convergence.
        
        K-means++ spreads initial centroids far apart by choosing each new
        centroid with probability proportional to squared distance from
        nearest existing centroid.
        
        Args:
            x (np.ndarray): Training data, shape (n_samples, n_features)
        
        Algorithm:
            1. Choose first centroid randomly from data
            2. For each remaining centroid:
                - Compute D(x)² = squared distance to nearest existing centroid
                - Choose next centroid with probability ∝ D(x)²
        """
        num_samples, dim = x.shape
        centroids = []

        # Step 1: Choose first centroid uniformly at random
        centroids.append(x[np.random.randint(num_samples)])
        # Step 2: Choose remaining k-1 centroids
        for i in range(1, self.k):
            # Compute squared distance from each point to nearest centroid
            distances = np.array([
                min(np.linalg.norm(sample-c)**2 for c in centroids) for sample in x
            ])
            # Normalize distances to get probability distribution
            weights = distances / distances.sum()
            # Sample next centroid with probability proportional to distance²
            idx = np.random.choice(num_samples, p=weights)
            centroids.append(x[idx])
        # Store centroids as numpy array for efficient computation
        self.centroids = np.array(centroids)

    def _assign_label(self, x):
        """
        Assignment step: Assign each point to nearest centroid.
        
        Args:
            x (np.ndarray): Data points, shape (n_samples, n_features)
        
        Returns:
            np.ndarray: Cluster labels for each point, shape (n_samples,)
                       Values in range [0, k-1]
        
        Implementation:
            - Computes Euclidean distance from each point to each centroid
            - Assigns point to cluster with minimum distance
        """
        # Initialize distance matrix: rows=samples, cols=centroids
        dists = np.zeros((x.shape[0], self.k))
        # Compute distance from each point to each centroid
        for i, centroid in enumerate(self.centroids):
            dists[:, i] = np.linalg.norm(x - centroid, axis=1)
        # Assign each point to closest centroid (minimum distance)
        labels = np.argmin(dists, axis=1)
        return labels
    
    def _update_centroid(self, x, labels):
        """
        Update step: Recompute centroids as mean of assigned points.
        
        Args:
            x (np.ndarray): Data points, shape (n_samples, n_features)
            labels (np.ndarray): Current cluster assignments, shape (n_samples,)
        
        Returns:
            float: Maximum centroid movement (used to check convergence)
        
        Algorithm:
            For each cluster:
                1. Find all points assigned to that cluster
                2. Compute mean of those points → new centroid
                3. Track maximum movement across all centroids
        """
        max_error = 0
        # Update each centroid
        for i in range(self.k):
            # Get all points assigned to cluster i
            cluster = x[labels==i]
            # Compute new centroid as mean of cluster points
            updated_centroid = np.mean(cluster, axis=0)
            # Calculate how far this centroid moved (for convergence check)
            error = np.linalg.norm(self.centroids[i] - updated_centroid)
            # Update centroid
            self.centroids[i] = updated_centroid
            # Track maximum movement across all centroids
            if error > max_error:
                max_error = error
        return max_error

    def fit(self, x):
        """
        Fit K-Means model to training data.
        
        Args:
            x (np.ndarray): Training data, shape (n_samples, n_features)
        
        Algorithm:
            1. Initialize centroids using K-means++
            2. Repeat until convergence or max_iter:
                a. Assign each point to nearest centroid
                b. Update centroids as mean of assigned points
                c. Check if centroids moved less than threshold
        
        Convergence:
            Stops when maximum centroid movement < threshold, indicating
            centroids have stabilized and further iterations won't help.
        """
        # Step 1: Initialize centroids using K-means++ for better starting point
        self._kmeans_plusplus_initialization(x)
        # Step 2: Iteratively optimize centroids
        for i in range(self.max_iter):
            labels = self._assign_label(x)
            error = self._update_centroid(x, labels)
            # Check convergence: if centroids barely moved, we're done
            if error < self.threshold:
                print(f"K-means converged after {i+1} iterations")
                return
        # If we exit loop, we hit max iterations without converging
        print("Max iterations reached")
            
        
    def predict(self, x):
        """
        Predict cluster labels for new data points.
        
        Args:
            x (np.ndarray): New data points, shape (n_samples, n_features)
        
        Returns:
            np.ndarray: Predicted cluster labels, shape (n_samples,)
        
        Note:
            Must call fit() before predict() to have trained centroids.
            Simply assigns each point to its nearest centroid.
        """
        return self._assign_label(x)

# Example usage and visualization
def test_kmeans():
    """Demonstrate K-Means on synthetic data"""
    np.random.seed(42)
    
    # Generate synthetic data with 3 clusters
    n_samples = 300
    cluster1 = np.random.randn(n_samples, 2) + np.array([2, 2])
    cluster2 = np.random.randn(n_samples, 2) + np.array([-2, -2])
    cluster3 = np.random.randn(n_samples, 2) + np.array([2, -2])
    X = np.vstack([cluster1, cluster2, cluster3])


    # Visualize Data
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
    plt.title('K-Means Data Visualization')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')    
    
    # Fit K-Means
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    labels = kmeans.predict(X)
        
    # Visualize Results
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
                c='red', marker='X', s=200, edgecolors='black', linewidth=2)
    plt.title('K-Means Clustering Results')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_kmeans()