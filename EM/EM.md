### 分析与详细讲解 EM 算法

#### 1. 概述

EM（Expectation-Maximization，期望最大化）算法是一种迭代优化算法，用于求解具有隐变量的概率模型的极大似然估计或极大后验估计。EM 算法常用于数据缺失、混合模型参数估计等场景。其核心思想是通过迭代步骤，不断更新参数，使得观测数据的似然函数逐步增加，直至收敛。

#### 2. 算法步骤

EM 算法主要分为两个步骤：期望步（E步）和最大化步（M步）。

1. **E步（Expectation Step）**：
   - 根据当前参数估计，计算隐变量的条件期望。
   - 计算公式：$$ Q(\theta|\theta^{(t)}) = E_{Z|Y,\theta^{(t)}}[\log P(Y,Z|\theta)] $$
     其中，$\theta^{(t)}$ 为第 t 次迭代的参数估计。

2. **M步（Maximization Step）**：
   - 最大化 Q 函数，更新参数。
   - 更新公式：$$ \theta^{(t+1)} = \arg\max_{\theta} Q(\theta|\theta^{(t)}) $$

#### 3. 算法收敛性

EM 算法通过以下两条定理保证收敛性：

- **定理 9.1**：EM 参数估计序列的似然函数值是单调递增的。
- **定理 9.2**：在一定条件下，EM 算法得到的参数估计序列收敛到观测数据对数似然函数的稳定点。

#### 4. 实例：三硬币模型

**模型设定**：

- 硬币 A、B、C 的正面概率分别为 $\pi, p, q$。
- 硬币 A 正面时选 B，反面选 C。
- 观测结果：$ Y = \{1, 1, 0, 1, 0, 0, 1, 0, 1, 1\} $。

**算法步骤**：

1. **E步**：
   - 计算隐变量 $ Z $ 的条件期望（即每次观测结果来自于硬币 B 的概率）。

2. **M步**：
   - 根据期望最大化 $ Q $ 函数，更新 $\pi, p, q$。

**迭代公式**：

- $ \pi = \frac{\sum_{i=1}^N E[z_i|y_i, \theta^{(t)}]}{N} $
- $ p = \frac{\sum_{i=1}^N y_i E[z_i|y_i, \theta^{(t)}]}{\sum_{i=1}^N E[z_i|y_i, \theta^{(t)}]} $
- $ q = \frac{\sum_{i=1}^N y_i (1 - E[z_i|y_i, \theta^{(t)}])}{\sum_{i=1}^N (1 - E[z_i|y_i, \theta^{(t)}])} $

#### 5. 高斯混合模型中的应用

**模型设定**：

- 假设观测数据 $ y_1, y_2, \ldots, y_N $ 由 K 个高斯分模型生成，每个分模型的概率密度函数为 $ \mathcal{N}(\mu_k, \Sigma_k) $。

**算法步骤**：

1. **E步**：
   - 计算每个观测数据属于第 k 个高斯分模型的概率（响应度）。
   - 公式：$$ \gamma_{ik} = \frac{\pi_k \mathcal{N}(y_i|\mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(y_i|\mu_j, \Sigma_j)} $$

2. **M步**：
   - 更新高斯混合模型的参数。
   - 更新公式：
     - $ \pi_k = \frac{1}{N} \sum_{i=1}^N \gamma_{ik} $
     - $ \mu_k = \frac{\sum_{i=1}^N \gamma_{ik} y_i}{\sum_{i=1}^N \gamma_{ik}} $
     - $ \Sigma_k = \frac{\sum_{i=1}^N \gamma_{ik} (y_i - \mu_k)(y_i - \mu_k)^T}{\sum_{i=1}^N \gamma_{ik}} $


#### 3. 公式证明

**1. E步中的响应度公式证明**

响应度 $\gamma_{ik}$ 表示观测数据 $ y_i $ 由第 $ k $ 个高斯分模型生成的概率。使用贝叶斯定理，我们有：
$$ \gamma_{ik} = P(z_i=k | y_i, \Theta) = \frac{P(y_i | z_i=k, \Theta) P(z_i=k | \Theta)}{P(y_i | \Theta)} $$

其中：
- $ P(z_i=k | \Theta) = \pi_k $
- $ P(y_i | z_i=k, \Theta) = \mathcal{N}(y_i|\mu_k, \Sigma_k) $
- $ P(y_i | \Theta) = \sum_{j=1}^{K} \pi_j \mathcal{N}(y_i|\mu_j, \Sigma_j) $

代入上述公式，得到：
$$ \gamma_{ik} = \frac{\pi_k \mathcal{N}(y_i|\mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(y_i|\mu_j, \Sigma_j)} $$

**2. M步中的参数更新公式证明**

在 M 步中，我们通过最大化对数似然函数来更新参数。

- 更新权重 $\pi_k$：
  $$ \pi_k = \frac{1}{N} \sum_{i=1}^{N} \gamma_{ik} $$
  这是因为 $\pi_k$ 表示第 $ k $ 个高斯分模型的权重，即观测数据点属于第 $ k $ 个高斯分模型的比例。

- 更新均值 $\mu_k$：
  $$ \mu_k = \frac{\sum_{i=1}^{N} \gamma_{ik} y_i}{\sum_{i=1}^{N} \gamma_{ik}} $$
  这是加权平均值，其中权重为响应度 $\gamma_{ik}$。

- 更新协方差矩阵 $\Sigma_k$：
  $$ \Sigma_k = \frac{\sum_{i=1}^{N} \gamma_{ik} (y_i - \mu_k)(y_i - \mu_k)^T}{\sum_{i=1}^{N} \gamma_{ik}} $$
  这是加权协方差矩阵，其中权重为响应度 $\gamma_{ik}$。