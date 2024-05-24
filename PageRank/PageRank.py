import numpy as np
import matplotlib.pyplot as plt
from typing import List

class PageRank:
    """
    PageRank算法类，用于计算网页的PageRank值。

    属性:
        transition_matrix (np.ndarray): 转移矩阵
        damping_factor (float): 阻尼因子，默认值为0.85
        epsilon (float): 收敛阈值，默认值为1e-6
        max_iter (int): 最大迭代次数，默认值为100
    """
    
    def __init__(self, transition_matrix: np.ndarray, damping_factor: float = 0.85, 
                 epsilon: float = 1e-6, max_iter: int = 100):
        """
        初始化PageRank类。

        参数:
            transition_matrix (np.ndarray): 转移矩阵
            damping_factor (float): 阻尼因子，默认值为0.85
            epsilon (float): 收敛阈值，默认值为1e-6
            max_iter (int): 最大迭代次数，默认值为100
        """
        self.transition_matrix = transition_matrix
        self.damping_factor = damping_factor
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.n = transition_matrix.shape[0]
        self.ranks = np.ones(self.n) / self.n  # 初始化PageRank值为均匀分布
        self.history = []  # 记录迭代过程中每个节点的PageRank值

    def iterate(self):
        """
        执行PageRank迭代计算，直到收敛或达到最大迭代次数。
        """
        teleport = np.ones(self.n) / self.n
        
        for _ in range(self.max_iter):
            new_ranks = self.damping_factor * self.transition_matrix @ self.ranks + (1 - self.damping_factor) * teleport
            self.history.append(new_ranks.copy())
            if np.linalg.norm(new_ranks - self.ranks, 1) < self.epsilon:
                break
            self.ranks = new_ranks

    def visualize_iterations(self):
        """
        可视化PageRank值在迭代过程中的变化情况。
        """
        self.iterate()
        iterations = len(self.history)
        for i in range(self.n):
            plt.plot(range(iterations), [self.history[j][i] for j in range(iterations)], label=f'Node {i+1}')
        
        plt.xlabel('Iterations')
        plt.ylabel('PageRank Value')
        plt.title('PageRank Values Convergence')
        plt.legend()
        plt.savefig('figures/PageRank.png')
        plt.close()

    def get_ranks(self) -> np.ndarray:
        """
        获取最终的PageRank值。

        返回:
            np.ndarray: 最终的PageRank值向量
        """
        return self.ranks

# 示例使用
# 定义转移矩阵
M = np.array([
    [0, 0, 1/2, 0],
    [1/3, 0, 0, 1/2],
    [1/3, 1/2, 0, 1/2],
    [1/3, 1/2, 1/2, 0]
])

pr = PageRank(M)
pr.visualize_iterations()
final_ranks = pr.get_ranks()
print("最终的PageRank值: ", final_ranks)
