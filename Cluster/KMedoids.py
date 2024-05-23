
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 生成数据集
def generate_data(n_samples: int = 300, n_features: int = 2, centers: int = 3, random_state: int = 42):
    X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, random_state=random_state)
    return X

X = generate_data()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


class KMedoids:
    def __init__(self, n_clusters: int = 3, max_iter: int = 300, tol: float = 1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.medoids = None

    def fit(self, X: np.ndarray):
        m, n = X.shape
        np.random.seed(42)
        initial_indices = np.random.permutation(m)[:self.n_clusters]
        self.medoids = X[initial_indices]

        for _ in range(self.max_iter):
            labels = self._assign_labels(X)
            new_medoids = np.array([self._compute_new_medoid(X[labels == i]) for i in range(self.n_clusters)])
            if np.all(new_medoids == self.medoids):
                break
            self.medoids = new_medoids

    def _assign_labels(self, X: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(X[:, np.newaxis] - self.medoids, axis=2)
        return np.argmin(distances, axis=1)

    def _compute_new_medoid(self, cluster: np.ndarray) -> np.ndarray:
        m = cluster.shape[0]
        if m == 0:
            return cluster
        distances = np.sum(np.linalg.norm(cluster[:, np.newaxis] - cluster, axis=2), axis=1)
        return cluster[np.argmin(distances)] # 返回距离最近的点作为新的中心点

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._assign_labels(X)

# 使用 K-medoids 进行聚类
kmedoids = KMedoids(n_clusters=3)
kmedoids.fit(X_pca)
labels_kmedoids = kmedoids.predict(X_pca)

# 可视化结果
def plot_clusters(X: np.ndarray, labels: np.ndarray, medoids: np.ndarray, title: str):
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
    plt.scatter(medoids[:, 0], medoids[:, 1], c='red', s=200, alpha=0.75)
    plt.title(title)
    plt.savefig(f'figures/{title}.png')
    plt.close()

plot_clusters(X_pca, labels_kmedoids, kmedoids.medoids, 'K-medoids Clustering')