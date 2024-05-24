import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

class ConditionalRandomField:
    def __init__(self, num_states: int, num_features: int):
        """
        初始化条件随机场类

        Args:
            num_states (int): 隐藏状态的数量
            num_features (int): 特征的数量
        """
        self.num_states = num_states
        self.num_features = num_features
        self.transition_params = np.random.randn(num_states, num_states)  # 转移参数
        self.feature_weights = np.random.randn(num_states, num_features)  # 特征权重

    def compute_potential(self, features: np.ndarray) -> np.ndarray:
        """
        计算特征势函数

        Args:
            features (np.ndarray): 特征数组

        Returns:
            np.ndarray: 势函数数组
        """
        potentials = np.dot(features, self.feature_weights.T)
        return potentials

    def forward_backward(self, potentials: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        前向后向算法计算前向和后向概率

        Args:
            potentials (np.ndarray): 势函数数组

        Returns:
            Tuple[np.ndarray, np.ndarray]: 前向概率和后向概率
        """
        num_samples, num_states = potentials.shape
        forward = np.zeros((num_samples, num_states))
        backward = np.zeros((num_samples, num_states))

        # 初始化前向概率
        forward[0] = potentials[0]

        # 计算前向概率
        for t in range(1, num_samples):
            for s in range(num_states):
                forward[t, s] = np.log(np.sum(np.exp(forward[t-1] + self.transition_params[:, s]))) + potentials[t, s]

        # 初始化后向概率
        backward[-1] = 0

        # 计算后向概率
        for t in range(num_samples - 2, -1, -1):
            for s in range(num_states):
                backward[t, s] = np.log(np.sum(np.exp(backward[t+1] + self.transition_params[s, :] + potentials[t+1, :])))

        return forward, backward

    def viterbi_decode(self, features: np.ndarray) -> List[int]:
        """
        使用维特比算法进行解码

        Args:
            features (np.ndarray): 特征数组

        Returns:
            List[int]: 最优状态序列
        """
        potentials = self.compute_potential(features)
        num_samples, num_states = potentials.shape

        viterbi = np.zeros((num_samples, num_states))
        backpointer = np.zeros((num_samples, num_states), dtype=int)

        # 初始化维特比算法
        viterbi[0] = potentials[0]

        # 递推计算
        for t in range(1, num_samples):
            for s in range(num_states):
                trans_probs = viterbi[t-1] + self.transition_params[:, s]
                backpointer[t, s] = np.argmax(trans_probs)
                viterbi[t, s] = np.max(trans_probs) + potentials[t, s]

        # 回溯最优路径
        best_path = [np.argmax(viterbi[-1])]
        for t in range(num_samples - 1, 0, -1):
            best_path.insert(0, backpointer[t, best_path[0]])

        return best_path

    def compute_log_likelihood(self, features: np.ndarray, labels: np.ndarray) -> float:
        """
        计算给定标签的对数似然

        Args:
            features (np.ndarray): 特征数组
            labels (np.ndarray): 标签数组

        Returns:
            float: 对数似然
        """
        potentials = self.compute_potential(features)
        num_samples = len(labels)
        log_likelihood = 0.0

        for t in range(num_samples):
            log_likelihood += potentials[t, labels[t]]
            if t > 0:
                log_likelihood += self.transition_params[labels[t-1], labels[t]]

        # 计算规范化因子
        forward, _ = self.forward_backward(potentials)
        log_likelihood -= np.log(np.sum(np.exp(forward[-1])))

        return log_likelihood

# 生成样本特征数据
num_samples = 10
num_features = 5
num_states = 3

np.random.seed(0)
features = np.random.rand(num_samples, num_features)
labels = np.random.randint(num_states, size=num_samples)

# 实例化条件随机场类
crf = ConditionalRandomField(num_states=num_states, num_features=num_features)

# 计算势函数
potentials = crf.compute_potential(features)

# 前向后向算法
forward, backward = crf.forward_backward(potentials)

# 维特比解码
best_path = crf.viterbi_decode(features)

# 打印结果
potentials, forward, backward, best_path, crf.compute_log_likelihood(features, labels)

def visualize_potentials(potentials: np.ndarray):
    """
    可视化势函数

    Args:
        potentials (np.ndarray): 势函数数组
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(potentials, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Potential Matrix Visualization')
    plt.xlabel('States')
    plt.ylabel('Samples')
    plt.savefig('figures/potentials.png')
    plt.close()


def visualize_path(path: List[int]):
    """
    可视化最优路径

    Args:
        path (List[int]): 最优路径
    """
    plt.figure(figsize=(10, 6))
    plt.plot(path, marker='o')
    plt.title('Best State Path')
    plt.xlabel('Sample Index')
    plt.ylabel('State')
    plt.savefig('figures/best_path.png')
    plt.close()

# 可视化势函数
visualize_potentials(potentials)

# 可视化最优路径
visualize_path(best_path)

