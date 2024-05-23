import numpy as np
from sklearn.datasets import load_digits
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 10)})
from tqdm import trange

class UMAP:
    def __init__(self, n_neighbors: int = 15, min_dist: float = 0.1, n_components: int = 2, n_epochs: int = 200):
        """
        UMAP class implementing a simplified version of the algorithm.

        :param n_neighbors: Number of nearest neighbors for graph construction.
        :param min_dist: Minimum distance between embedded points.
        :param n_components: Number of dimensions for the reduced space.
        :param n_epochs: Number of epochs for optimization.
        """
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_components = n_components
        self.n_epochs = n_epochs

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the UMAP model to the data and return the embedding.

        :param X: Input data of shape (n_samples, n_features)
        :return: Reduced data of shape (n_samples, n_components)
        """
        n_samples = X.shape[0]

        # Step 1: Find nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors).fit(X)
        distances, indices = nbrs.kneighbors(X)

        # Step 2: Construct high-dimensional graph (pairwise probabilities)
        graph = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(self.n_neighbors):
                graph[i, indices[i, j]] = np.exp(-distances[i, j] ** 2)

        # Normalize probabilities
        graph = graph / graph.sum(axis=1)[:, np.newaxis]

        # Step 3: Initialize embedding
        embedding = np.random.uniform(-1, 1, (n_samples, self.n_components))

        # Step 4: Optimize embedding using gradient descent
        learning_rate = 0.01
        for _ in trange(self.n_epochs):
            for i in range(n_samples):
                attraction = np.zeros(self.n_components)
                for j in range(n_samples):
                    if graph[i, j] > 0:
                        diff = embedding[i] - embedding[j]
                        dist = np.linalg.norm(diff)
                        weight = graph[i, j]
                        attraction += weight * diff / (dist + 1e-5)

                embedding[i] -= learning_rate * attraction

        return embedding

if __name__ == '__main__':
    # Load the digits dataset and apply UMAP projection
    digits = load_digits()
    X, y = digits.data, digits.target

    # Instantiate UMAP and project the digits data
    umap = UMAP(n_neighbors=10, min_dist=0.1, n_components=2, n_epochs=200)
    embedding = umap.fit_transform(X)

    # Visualize the UMAP projection
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap='Spectral', s=5)
    plt.colorbar(scatter)
    plt.title('UMAP Projection of Digits Dataset')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.savefig('figures/UMAP.png')
    plt.close()