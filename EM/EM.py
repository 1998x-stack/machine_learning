import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

def gaussian_pdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """计算多维高斯分布的概率密度函数值
    
    Args:
        x (np.ndarray): 观测数据点
        mean (np.ndarray): 高斯分布的均值向量
        cov (np.ndarray): 高斯分布的协方差矩阵
    
    Returns:
        float: 概率密度函数值
    """
    d = len(x)
    cov_det = np.linalg.det(cov)
    cov_inv = np.linalg.inv(cov)
    norm_const = 1.0 / (np.power((2 * np.pi), float(d) / 2) * np.sqrt(cov_det)) # 归一化常数
    x_mu = np.matrix(x - mean)
    result = np.power(np.e, -0.5 * (x_mu * cov_inv * x_mu.T)) # 指数项
    return norm_const * result # 返回概率密度函数值

def initialize_parameters(data: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """初始化高斯混合模型的参数
    
    Args:
        data (np.ndarray): 观测数据
        K (int): 高斯分模型的数量
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 初始化的模型参数 (权重, 均值, 协方差)
    """
    n, d = data.shape
    weights = np.ones(K) / K # 权重初始化为均匀分布
    means = data[np.random.choice(n, K, False), :] # 随机选择 K 个观测数据作为均值
    covariances = np.array([np.eye(d)] * K) # 协方差矩阵初始化为单位矩阵
    return weights, means, covariances

def e_step(data: np.ndarray, weights: np.ndarray, means: np.ndarray, covariances: np.ndarray) -> np.ndarray:
    """执行 EM 算法的 E 步
    
    Args:
        data (np.ndarray): 观测数据
        weights (np.ndarray): 各高斯分模型的权重
        means (np.ndarray): 各高斯分模型的均值向量
        covariances (np.ndarray): 各高斯分模型的协方差矩阵
    
    Returns:
        np.ndarray: 响应度矩阵
    """
    n, K = data.shape[0], len(weights)
    responsibilities = np.zeros((n, K))
    for i in range(n):
        for k in range(K):
            responsibilities[i, k] = weights[k] * gaussian_pdf(data[i], means[k], covariances[k]) # 计算响应度
        responsibilities[i, :] /= np.sum(responsibilities[i, :]) # 归一化
    return responsibilities

def m_step(data: np.ndarray, responsibilities: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """执行 EM 算法的 M 步
    
    Args:
        data (np.ndarray): 观测数据
        responsibilities (np.ndarray): 响应度矩阵
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 更新后的模型参数 (权重, 均值, 协方差)
    """
    n, d = data.shape
    K = responsibilities.shape[1]
    weights = np.zeros(K)
    means = np.zeros((K, d))
    covariances = np.zeros((K, d, d))
    
    for k in range(K):
        Nk = np.sum(responsibilities[:, k]) # 第 k 个高斯分模型的权重
        weights[k] = Nk / n # 更新权重
        means[k] = np.sum(responsibilities[:, k].reshape(-1, 1) * data, axis=0) / Nk # 更新均值
        diff = data - means[k] # 计算数据点与均值的差
        covariances[k] = np.dot(responsibilities[:, k] * diff.T, diff) / Nk # 更新协方差矩阵
        
    return weights, means, covariances

def em_algorithm(data: np.ndarray, K: int, max_iters: int = 100, tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """执行 EM 算法以估计高斯混合模型的参数
    
    Args:
        data (np.ndarray): 观测数据
        K (int): 高斯分模型的数量
        max_iters (int): 最大迭代次数
        tol (float): 收敛判定阈值
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 最终估计的模型参数 (权重, 均值, 协方差)
    """
    weights, means, covariances = initialize_parameters(data, K)
    log_likelihoods = []

    for i in range(max_iters):
        responsibilities = e_step(data, weights, means, covariances)
        weights, means, covariances = m_step(data, responsibilities)

        log_likelihood = np.sum([np.log(np.sum([weights[k] * gaussian_pdf(data[j], means[k], covariances[k]) for k in range(K)])) for j in range(data.shape[0])])
        log_likelihoods.append(log_likelihood)
        
        if i > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            break

    return weights, means, covariances, log_likelihoods

# 生成测试数据
np.random.seed(42)
n_samples = 300
C = np.array([[0., -0.1], [1.7, 0.4]])
X_train = np.r_[np.dot(np.random.randn(n_samples, 2), C),
                np.dot(np.random.randn(n_samples, 2), C) + np.array([3, 3])]

# 运行 EM 算法
K = 2
weights, means, covariances, log_likelihoods = em_algorithm(X_train, K)

# 可视化优化过程中的参数变化
plt.figure(figsize=(10, 6))
plt.plot(log_likelihoods)
plt.title('Log-Likelihood during EM Optimization')
plt.xlabel('Iteration')
plt.ylabel('Log-Likelihood')
plt.grid(True)
plt.savefig('figures/Log-Likelihood during EM Optimization.png')
plt.close()

# 可视化聚类结果
def plot_gmm(X, means, covariances, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], s=10, cmap='viridis')
    plt.title(title)
    
    for mean, cov in zip(means, covariances):
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        axis_length = 2 * np.sqrt(eigenvalues)
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        
        ellipse = plt.matplotlib.patches.Ellipse(mean, *axis_length, angle, edgecolor='red', facecolor='none')
        plt.gca().add_patch(ellipse)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    plt.savefig(f'figures/{title}.png')
    plt.close()

plot_gmm(X_train, means, covariances, 'GMM Clustering Result')
