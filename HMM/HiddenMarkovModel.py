# Run the updated code with corrections

import numpy as np
import matplotlib.pyplot as plt
class HiddenMarkovModel:
    def __init__(self, n_states: int, n_observations: int):
        """
        初始化隐马尔可夫模型

        参数:
            n_states (int): 状态的数量
            n_observations (int): 观测的数量
        """
        self.n_states = n_states # 状态的数量
        self.n_observations = n_observations # 观测的数量
        self.pi = np.ones(n_states) / n_states # 初始状态概率
        self.A = np.ones((n_states, n_states)) / n_states # 状态转移概率
        self.B = np.ones((n_states, n_observations)) / n_observations # 观测概率

    def forward_algorithm(self, observations: list[int]) -> float:
        """
        前向算法计算观测序列的概率

        参数:
            observations (list[int]): 观测序列

        返回:
            float: 观测序列的概率
        """
        T = len(observations)
        alpha = np.zeros((T, self.n_states)) # 前向概率
        alpha[0] = self.pi * self.B[:, observations[0]] # 初始值
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t-1] * self.A[:, j]) * self.B[j, observations[t]] # 前向递推公式
        return np.sum(alpha[T-1])

    def baum_welch_training(self, observations: list[int], n_iterations: int = 100):
        """
        Baum-Welch算法训练隐马尔可夫模型

        参数:
            observations (list[int]): 观测序列
            n_iterations (int): 迭代次数，默认为100
        """
        T = len(observations)
        for n in range(n_iterations):
            alpha = np.zeros((T, self.n_states))
            beta = np.zeros((T, self.n_states))
            gamma = np.zeros((T, self.n_states))
            xi = np.zeros((T-1, self.n_states, self.n_states))
            alpha[0] = self.pi * self.B[:, observations[0]]
            for t in range(1, T):
                for j in range(self.n_states):
                    alpha[t, j] = np.sum(alpha[t-1] * self.A[:, j]) * self.B[j, observations[t]]
            beta[T-1] = np.ones(self.n_states)
            for t in range(T-2, -1, -1):
                for i in range(self.n_states):
                    beta[t, i] = np.sum(self.A[i, :] * self.B[:, observations[t+1]] * beta[t+1])
            for t in range(T-1):
                denom = np.sum(alpha[t] * beta[t])
                for i in range(self.n_states):
                    gamma[t, i] = alpha[t, i] * beta[t, i] / denom
                    xi[t, i, :] = (alpha[t, i] * self.A[i, :] * self.B[:, observations[t+1]] * beta[t+1]) / denom
            gamma[T-1] = alpha[T-1] / np.sum(alpha[T-1])
            self.pi = gamma[0]
            self.A = np.sum(xi, axis=0) / np.sum(gamma[:-1], axis=0)[:, None]
            for k in range(self.n_observations):
                mask = (np.array(observations) == k)
                self.B[:, k] = np.sum(gamma[mask], axis=0) / np.sum(gamma, axis=0)

    def viterbi_algorithm(self, observations: list[int]) -> list[int]:
        """
        维特比算法计算最可能的状态序列

        参数:
            observations (list[int]): 观测序列

        返回:
            list[int]: 最可能的状态序列
        """
        T = len(observations)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        delta[0] = self.pi * self.B[:, observations[0]]
        for t in range(1, T):
            for j in range(self.n_states):
                max_val = delta[t-1] * self.A[:, j]
                delta[t, j] = np.max(max_val) * self.B[j, observations[t]]
                psi[t, j] = np.argmax(max_val)
        states = np.zeros(T, dtype=int)
        states[T-1] = np.argmax(delta[T-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        return states.tolist()

    def visualize(self, observations: list[int], states: list[int]):
        """
        可视化结果

        参数:
            observations (list[int]): 观测序列
            states (list[int]): 状态序列
        """
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        ax[0].plot(observations, 'bo-', label='观测序列')
        ax[0].set_title('观测序列')
        ax[0].set_xlabel('时间')
        ax[0].set_ylabel('观测值')
        ax[0].legend()
        ax[1].plot(states, 'rs-', label='状态序列')
        ax[1].set_title('状态序列')
        ax[1].set_xlabel('时间')
        ax[1].set_ylabel('状态')
        ax[1].legend()
        plt.tight_layout()
        plt.savefig(f"figures/observations_states.png")

# 创建HMM实例
hmm = HiddenMarkovModel(n_states=3, n_observations=3)

# 定义观测序列
observations = [0, 1, 2, 1, 0]

# 训练HMM
hmm.baum_welch_training(observations, n_iterations=10)

# 计算观测序列的概率
prob = hmm.forward_algorithm(observations)
print(f"观测序列的概率: {prob}")

# 使用维特比算法计算最可能的状态序列
states = hmm.viterbi_algorithm(observations)
print(f"最可能的状态序列: {states}")

# 可视化结果
hmm.visualize(observations, states)