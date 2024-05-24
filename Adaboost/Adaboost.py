import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
"""
初始化训练数据的权值分布。
循环训练弱分类器：
    在权值分布下训练弱分类器。
    计算弱分类器的训练误差。
    更新训练数据的权值分布。
构建最终分类器。
"""
class DecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.sign = None
        self.prediction = None

    def predict(self, X: np.ndarray):
        """
        对输入的样本进行预测

        Parameters:
            X (np.ndarray): 输入的样本特征矩阵，shape为 (n_samples, n_features)

        Returns:
            np.ndarray: 预测结果，shape为 (n_samples,)
        """
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)
        negative_idx = (X[:, self.feature_index] * self.sign < self.threshold * self.sign) # 小于阈值的样本
        predictions[negative_idx] = -1
        return predictions

class AdaBoost:
    def __init__(self, n_estimators: int):
        """
        AdaBoost分类器初始化函数

        Parameters:
            n_estimators (int): 弱分类器的数量

        Returns:
            None
        """
        self.n_estimators = n_estimators # 弱分类器的数量
        self.alphas = [] # 弱分类器的权重
        self.models = [] # 弱分类器的列表

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        使用AdaBoost算法拟合训练数据

        Parameters:
            X (np.ndarray): 输入的样本特征矩阵，shape为 (n_samples, n_features)
            y (np.ndarray): 输入的样本标签，shape为 (n_samples,)

        Returns:
            None
        """
        n_samples, n_features = X.shape # 样本数量和特征数量
        w = np.ones(n_samples) / n_samples # 初始化样本权重，均匀分布

        for _ in range(self.n_estimators):
            stump = self._train_stump(X, y, w) # 训练一个决策树桩
            pred = stump.predict(X) # 使用决策树桩进行预测
            err = np.sum(w * (pred != y)) / np.sum(w) # 计算加权误差
            alpha = 0.5 * np.log((1 - err) / (err + 1e-10)) # 计算弱分类器的权重
            w = w * np.exp(-alpha * y * pred) # 更新样本权重
            w /= np.sum(w) # 归一化样本权重

            self.alphas.append(alpha) # 保存弱分类器的权重
            self.models.append(stump) # 保存弱分类器

    def _train_stump(self, X: np.ndarray, y: np.ndarray, w: np.ndarray):
        """
        使用加权数据训练一个决策树桩

        Parameters:
            X (np.ndarray): 输入的样本特征矩阵，shape为 (n_samples, n_features)
            y (np.ndarray): 输入的样本标签，shape为 (n_samples,)
            w (np.ndarray): 样本权重，shape为 (n_samples,)

        Returns:
            DecisionStump: 训练好的决策树桩
        """
        stump = DecisionStump()
        min_err = float('inf')

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature]) # 特征的唯一值
            for threshold in thresholds:
                for sign in [-1, 1]:
                    pred = np.ones(y.shape)
                    pred[X[:, feature] * sign < threshold * sign] = -1 # 根据阈值和符号进行预测
                    err = np.sum(w * (pred != y)) # 计算加权误差

                    if err < min_err: # 保存最小误差的决策树桩
                        min_err = err
                        stump.feature_index = feature
                        stump.threshold = threshold
                        stump.sign = sign
                        stump.prediction = pred

        return stump

    def predict(self, X: np.ndarray):
        """
        对输入的样本进行预测

        Parameters:
            X (np.ndarray): 输入的样本特征矩阵，shape为 (n_samples, n_features)

        Returns:
            np.ndarray: 预测结果，shape为 (n_samples,)
        """
        final_pred = np.zeros(X.shape[0])
        for alpha, stump in zip(self.alphas, self.models):
            final_pred += alpha * stump.predict(X)
        return np.sign(final_pred)

# 测试AdaBoost算法
X, y = make_classification(n_samples=500, n_features=10, n_informative=5, n_redundant=2, random_state=42)
y = np.where(y == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = AdaBoost(n_estimators=50)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')

# 可视化决策边界
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.savefig('figures/decision_boundary.png')
    plt.close()

# 仅使用前两个特征进行可视化
X_vis = X_train[:, :2]
y_vis = y_train

model_vis = AdaBoost(n_estimators=50)
model_vis.fit(X_vis, y_vis)
plot_decision_boundary(model_vis, X_vis, y_vis)