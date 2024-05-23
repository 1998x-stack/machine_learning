import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from ml_dataset import load_data


class PCA:
    """
    Principal Component Analysis (PCA) implementation using NumPy.
    
    Attributes:
        n_components (int): Number of principal components to keep.
        components (np.ndarray): Principal components.
        explained_variance (np.ndarray): Variance explained by each component.
        mean (np.ndarray): Mean of the data, used for centering.
    """
    
    def __init__(self, n_components: int):
        """
        Initialize the PCA instance.
        
        Args:
            n_components (int): Number of principal components to keep.
        """
        self.n_components = n_components
        self.components = None
        self.explained_variance = None
        self.mean = None
        
    def fit(self, X: np.ndarray) -> None:
        """
        Fit the PCA model to the given dataset.
        
        Args:
            X (np.ndarray): Data to be fitted, of shape (n_samples, n_features).
        """
        # Center the data by subtracting the mean
        self.mean = np.mean(X, axis=0) # 计算每一列的均值
        X_centered = X - self.mean # 每一列减去均值
        
        # Compute the covariance matrix
        cov_mat = np.cov(X_centered, rowvar=False) # rowvar=False表示每一列代表一个特征，计算协方差矩阵
        
        # Compute the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_mat) # 计算协方差矩阵的特征值和特征向量
        
        # Sort eigenvectors by descending eigenvalues
        sorted_indices = np.argsort(eigenvalues)[::-1] # 降序排列特征值的索引
        eigenvectors = eigenvectors[:, sorted_indices] # 重新排列特征向量
        eigenvalues = eigenvalues[sorted_indices] # 重新排列特征值
        
        # Keep only the top n_components eigenvectors
        self.components = eigenvectors[:, :self.n_components] # 取前n_components个特征向量
        self.explained_variance = eigenvalues[:self.n_components] # 取前n_components个特征值
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the data into the principal component space.
        
        Args:
            X (np.ndarray): Data to be transformed, of shape (n_samples, n_features).
        
        Returns:
            np.ndarray: Transformed data, of shape (n_samples, n_components).
        """
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

if __name__ == '__main__':
    digits = load_data('digits')
    X = digits.data
    y = digits.target
    pca_evaluator = PCA(n_components=2)
    pca_evaluator.fit(X)
    X_transformed = pca_evaluator.transform(X)
    plt.figure(figsize=(10, 6))
    for i in range(10):
        plt.scatter(X_transformed[y == i, 0], X_transformed[y == i, 1], label=str(i))
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Digits Dataset')
    plt.legend()
    plt.savefig('figures/PCA.png')
    plt.close()