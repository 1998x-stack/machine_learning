# 生成数据集和预处理
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def generate_data(n_samples: int = 300, n_features: int = 2, centers: int = 3, random_state: int = 42):
    X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, random_state=random_state)
    return X

X = generate_data()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

class KMeans:
    def __init__(self, n_clusters: int = 3, max_iter: int = 300, tol: float = 1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centers = None

    def fit(self, X: np.ndarray):
        np.random.seed(42)
        initial_indices = np.random.permutation(X.shape[0])[:self.n_clusters]
        self.centers = X[initial_indices]

        for _ in range(self.max_iter):
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

class BiKMeans:
    def __init__(self, n_clusters: int = 3, max_iter: int = 300, tol: float = 1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.clusters = []
        self.labels = None

    def fit(self, X: np.ndarray):
        # 初始化，将所有数据点作为一个簇
        self.labels = np.zeros(X.shape[0], dtype=int)
        self.clusters = [(X, np.zeros(X.shape[0], dtype=int))]
        
        while len(self.clusters) < self.n_clusters:
            max_sse_cluster_index = self._find_max_sse_cluster()
            max_sse_cluster, max_sse_labels = self.clusters.pop(max_sse_cluster_index)
            
            kmeans_pp = KMeansPlusPlus(n_clusters=2)
            kmeans_pp.fit(max_sse_cluster)
            new_labels = kmeans_pp.predict(max_sse_cluster)
            
            for i in range(2):
                sub_cluster = max_sse_cluster[new_labels == i]
                self.labels[max_sse_labels == max_sse_cluster_index][new_labels == i] = len(self.clusters)
                self.clusters.append((sub_cluster, np.where(new_labels == i, len(self.clusters), -1)))
        
    def _find_max_sse_cluster(self):
        max_sse = 0
        max_sse_index = 0
        for i, (cluster, labels) in enumerate(self.clusters):
            kmeans = KMeans(n_clusters=1)
            kmeans.fit(cluster)
            sse = np.sum((cluster - kmeans.centers) ** 2)# 求解簇内误差平方和
            if sse > max_sse:
                max_sse = sse
                max_sse_index = i # 记录簇内误差平方和最大的簇的索引
        return max_sse_index

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.labels

# 使用 Bi-Kmeans 进行聚类
bi_kmeans = BiKMeans(n_clusters=3)
bi_kmeans.fit(X_pca)
labels_bi_kmeans = bi_kmeans.predict(X_pca)

# 可视化结果
def plot_clusters(X: np.ndarray, labels: np.ndarray, title: str):
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
    plt.title(title)
    plt.savefig(f'figures/{title}.png')
    plt.close()

plot_clusters(X_pca, labels_bi_kmeans, 'Bi-Kmeans Clustering')