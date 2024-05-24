import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
sns.set()


class GibbsSamplerBivariateNormal:
    """吉布斯采样器用于生成二元正态分布样本"""

    def __init__(self, mu: np.ndarray, rho: np.ndarray, proposal_variance: float, num_samples: int,
                 min_bounds: np.ndarray, max_bounds: np.ndarray):
        """
        初始化吉布斯采样器
        :param mu: 目标均值 (2维数组)
        :param rho: 每个维度的相关系数 (2维数组)
        :param proposal_variance: 提案方差
        :param num_samples: 样本数量
        :param min_bounds: 每个维度的最小边界 (2维数组)
        :param max_bounds: 每个维度的最大边界 (2维数组)
        """
        self.mu = mu
        self.rho = rho
        self.proposal_variance = proposal_variance
        self.num_samples = num_samples
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds
        self.samples = np.zeros((num_samples, 2))

        # 初始化第一组样本
        self.samples[0, 0] = np.random.uniform(min_bounds[0], max_bounds[0])
        self.samples[0, 1] = np.random.uniform(min_bounds[1], max_bounds[1])

    def sample(self):
        """运行吉布斯采样器生成样本"""
        dims = [0, 1]  # 索引用于两个维度

        for t in range(1, self.num_samples):
            for i in dims:
                # 当前维度以外的维度
                non_dim = dims[1 - i]
                # 条件均值
                mu_cond = self.mu[i] + self.rho[i] * (self.samples[t - 1, non_dim] - self.mu[non_dim])
                # 条件方差
                var_cond = np.sqrt(1 - self.rho[i] ** 2)
                # 根据条件分布生成新样本
                self.samples[t, i] = norm.rvs(mu_cond, var_cond)

    def visualize(self):
        """可视化采样结果及前50个样本的采样路径"""
        plt.figure(figsize=(8, 8))
        plt.scatter(self.samples[:, 0], self.samples[:, 1], color='red', marker='.', label='Samples')

        # 绘制前50个样本的路径
        for t in range(50):
            plt.plot([self.samples[t, 0], self.samples[t + 1, 0]], [self.samples[t, 1], self.samples[t, 1]], 'k-')
            plt.plot([self.samples[t + 1, 0], self.samples[t + 1, 0]], [self.samples[t, 1], self.samples[t + 1, 1]], 'k-')
            plt.plot(self.samples[t + 1, 0], self.samples[t + 1, 1], 'ko')

        plt.scatter(self.samples[0, 0], self.samples[0, 1], color='green', marker='o', linewidths=3, label='x(t=0)')

        plt.xlabel('x_1')
        plt.ylabel('x_2')
        plt.legend(loc='upper left')
        plt.title('Gibbs Sampler for Bivariate Normal Distribution')
        plt.axis('square')
        plt.savefig('figures/gibbs_bivariate_normal.png')
        plt.close()
        
# 参数配置
mu = np.array([0, 0])
rho = np.array([0.8, 0.8])
proposal_variance = 1.0
num_samples = 5000
min_bounds = np.array([-3, -3])
max_bounds = np.array([3, 3])

# 运行吉布斯采样器并进行可视化
sampler = GibbsSamplerBivariateNormal(mu, rho, proposal_variance, num_samples, min_bounds, max_bounds)
sampler.sample()
sampler.visualize()