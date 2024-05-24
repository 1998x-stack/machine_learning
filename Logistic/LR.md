## Task_1: 解析 PPTX 文件内容并分解 Logistic 回归详细步骤


1. Logistic 回归的定义与应用场景
2. Logistic 分布和 Sigmoid 函数
3. 二项 Logistic 回归模型及其公式
4. 似然函数与极大似然估计
5. 模型参数估计与优化算法
6. 梯度下降法和牛顿法等最优化算法
7. 多项 Logistic 回归与最大熵模型

### 详细展开 Logistic 回归

#### Logistic 回归的定义与应用场景
Logistic 回归是一种广义线性模型，用于解决分类问题，特别是二分类问题。它预测事件发生的概率，可以处理连续值输入变量和二分类或多分类输出变量。

#### Logistic 分布和 Sigmoid 函数
Logistic 分布的累积分布函数 (CDF) 为：
$$ F(x) = \frac{1}{1 + e^{-\frac{x - \mu}{\gamma}}} $$

密度函数为：
$$ f(x) = \frac{e^{-\frac{x - \mu}{\gamma}}}{\gamma (1 + e^{-\frac{x - \mu}{\gamma}})^2} $$

Sigmoid 函数（或 Logistic 函数）是 Logistic 回归的核心：
$$ \sigma(x) = \frac{1}{1 + e^{-x}} $$

#### 二项 Logistic 回归模型
二项 Logistic 回归模型使用条件概率表示：
$$ P(Y|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \beta_2 X_2 + \ldots + \beta_n X_n)}} $$

事件发生的几率（odds）为：
$$ \text{odds} = \frac{P(Y=1|X)}{P(Y=0|X)} $$

对数几率（log-odds）：
$$ \log\left(\frac{P(Y=1|X)}{P(Y=0|X)}\right) = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \ldots + \beta_n X_n $$

#### 似然函数与极大似然估计
似然函数 L(θ|x) 是统计模型中参数 θ 的函数，表示为：
$$ L(\beta|X) = \prod_{i=1}^n P(Y_i|X_i;\beta) $$

为了简化计算，通常取对数得到对数似然函数：
$$ \log L(\beta|X) = \sum_{i=1}^n [Y_i \log P(Y_i|X_i;\beta) + (1 - Y_i) \log (1 - P(Y_i|X_i;\beta))] $$

#### 模型参数估计与优化算法
通过极大似然估计获取模型参数 β，目标是使对数似然函数最大化。常用的优化算法包括梯度下降法、牛顿法和拟牛顿法。

#### 梯度下降法
梯度下降法是一种迭代优化算法，通过不断更新参数，使得目标函数值逐渐减少。每次迭代的更新公式为：
$$ \beta = \beta - \alpha \nabla_{\beta} \log L(\beta|X) $$

其中，α 为学习率，∇ 为梯度算子。

#### 牛顿法
牛顿法利用目标函数的二阶泰勒展开进行优化，更新公式为：
$$ \beta = \beta - H^{-1} \nabla_{\beta} \log L(\beta|X) $$

其中，H 为海塞矩阵的逆矩阵。

### 多项 Logistic 回归与最大熵模型
多项 Logistic 回归用于多分类问题，最大熵模型通过最大化熵来选择最优模型。最大熵原理假设在所有可能的概率模型中，熵最大的模型是最好的。

#### 最大熵模型的学习
最大熵模型的学习可以形式化为约束最优化问题，通过求解对偶问题，使用迭代算法如改进的迭代尺度法进行优化。
