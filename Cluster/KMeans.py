import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

centers = 5
# 生成数据集
def generate_data(n_samples: int = 300, n_features: int = 2, centers: int = 3, random_state: int = 42):
    X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, random_state=random_state)
    return X

X = generate_data(centers=centers)

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

class KMeans:
    def __init__(self, n_clusters: int = 3, max_iter: int = 300, tol: float = 1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centers = None

    def fit(self, X: np.ndarray):
        # 随机初始化中心点
        np.random.seed(42)
        initial_indices = np.random.permutation(X.shape[0])[:self.n_clusters]
        self.centers = X[initial_indices]

        for _ in range(self.max_iter):
            # 分配每个点到最近的中心点
            labels = self._assign_labels(X)
            new_centers = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
            if np.all(np.linalg.norm(new_centers - self.centers, axis=1) < self.tol):
                break
            self.centers = new_centers

    def _assign_labels(self, X: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(X[:, np.newaxis] - self.centers, axis=2)
        return np.argmin(distances, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._assign_labels(X)

class KMeansPlusPlus(KMeans):
    def _init_centers(self, X: np.ndarray):
        np.random.seed(42)
        centers = [X[np.random.randint(X.shape[0])]]
        for _ in range(1, self.n_clusters):
            distances = np.min(np.linalg.norm(X[:, np.newaxis] - np.array(centers), axis=2), axis=1)
            prob = distances / np.sum(distances)
            cumulative_prob = np.cumsum(prob)
            r = np.random.rand()
            next_center = X[np.where(cumulative_prob >= r)[0][0]]
            centers.append(next_center)
        return np.array(centers)

    def fit(self, X: np.ndarray):
        self.centers = self._init_centers(X)
        super().fit(X)

# 使用 K-means 进行聚类
kmeans = KMeans(n_clusters=centers)
kmeans.fit(X_pca)
labels_kmeans = kmeans.predict(X_pca)

# 使用 K-means++ 进行聚类
kmeans_pp = KMeansPlusPlus(n_clusters=centers)
kmeans_pp.fit(X_pca)
labels_kmeans_pp = kmeans_pp.predict(X_pca)

# 评估聚类结果
def calculate_sse(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    return np.sum((X - centers[labels]) ** 2)

sse_kmeans = calculate_sse(X_pca, labels_kmeans, kmeans.centers)
sse_kmeans_pp = calculate_sse(X_pca, labels_kmeans_pp, kmeans_pp.centers)

print(f'K-means SSE: {sse_kmeans}')
print(f'K-means++ SSE: {sse_kmeans_pp}')

# 可视化结果
def plot_clusters(X: np.ndarray, labels: np.ndarray, centers: np.ndarray, title: str):
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)
    plt.title(title)
    # plt.show()
    plt.savefig(f'figures/{title}.png')
    plt.close()

plot_clusters(X_pca, labels_kmeans, kmeans.centers, 'K-means Clustering')
plot_clusters(X_pca, labels_kmeans_pp, kmeans_pp.centers, 'K-means++ Clustering')
