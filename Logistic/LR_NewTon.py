import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 仅选择前两类进行二分类
X = X[y != 2]
y = y[y != 2]

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 训练集和测试集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义逻辑斯蒂回归类并使用梯度下降法进行训练
class LogisticRegressionGD:
    """逻辑斯蒂回归模型，使用梯度下降法进行训练。
    
    Attributes:
        learning_rate (float): 学习率
        n_iterations (int): 迭代次数
        weights (np.ndarray): 模型权重
        bias (float): 模型偏差
    """

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """计算sigmoid函数值。
        
        Args:
            z (np.ndarray): 输入值
        
        Returns:
            np.ndarray: Sigmoid函数值
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """训练逻辑斯蒂回归模型。
        
        Args:
            X (np.ndarray): 训练数据
            y (np.ndarray): 训练标签
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测输入数据的标签。
        
        Args:
            X (np.ndarray): 输入数据
        
        Returns:
            np.ndarray: 预测标签
        """
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return np.where(y_predicted >= 0.5, 1, 0)

# 实例化并训练模型
model_gd = LogisticRegressionGD(learning_rate=0.1, n_iterations=1000)
model_gd.fit(X_train, y_train)

# 预测并评估模型
y_pred_gd = model_gd.predict(X_test)
accuracy_gd = np.mean(y_pred_gd == y_test)
print(f"Gradient Descent Logistic Regression Accuracy: {accuracy_gd:.4f}")

# 定义逻辑斯蒂回归类并使用牛顿法进行训练
class LogisticRegressionNewton:
    """逻辑斯蒂回归模型，使用牛顿法进行训练。
    
    Attributes:
        n_iterations (int): 迭代次数
        weights (np.ndarray): 模型权重
        bias (float): 模型偏差
    """

    def __init__(self, n_iterations: int = 100):
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """计算sigmoid函数值。
        
        Args:
            z (np.ndarray): 输入值
        
        Returns:
            np.ndarray: Sigmoid函数值
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """训练逻辑斯蒂回归模型。
        
        Args:
            X (np.ndarray): 训练数据
            y (np.ndarray): 训练标签
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # 计算Hessian矩阵
            diag_gradient = y_predicted * (1 - y_predicted)
            H = (1 / n_samples) * np.dot(X.T, diag_gradient[:, np.newaxis] * X)

            # 更新权重和偏差
            H_inv = np.linalg.inv(H)
            update = np.dot(H_inv, dw)
            self.weights -= update
            self.bias -= db

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测输入数据的标签。
        
        Args:
            X (np.ndarray): 输入数据
        
        Returns:
            np.ndarray: 预测标签
        """
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return np.where(y_predicted >= 0.5, 1, 0)

# 修改可视化函数以仅使用前两个特征
def plot_decision_boundary(model, X, y, title):
    """绘制决策边界。
    
    Args:
        model: 逻辑斯蒂回归模型
        X (np.ndarray): 输入数据
        y (np.ndarray): 标签
    """
    # 仅使用前两个特征进行可视化
    X = X[:, :2]
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
    plt.title(f'Decision Boundary {title}')
    plt.savefig(f"figures/decision_boundary{title}.png")
    plt.close()
    
# 实例化并训练模型
model_newton = LogisticRegressionNewton(n_iterations=10)
model_newton.fit(X_train, y_train)


# 修改训练数据和测试数据以仅使用前两个特征
X_train_2d = X_train[:, :2]
X_test_2d = X_test[:, :2]

# 重新训练模型
model_gd.fit(X_train_2d, y_train)
model_newton.fit(X_train_2d, y_train)

# 重新预测
y_pred_gd = model_gd.predict(X_test_2d)
y_pred_newton = model_newton.predict(X_test_2d)

# 重新计算准确率
accuracy_gd = np.mean(y_pred_gd == y_test)
accuracy_newton = np.mean(y_pred_newton == y_test)
print(f"Gradient Descent Logistic Regression Accuracy: {accuracy_gd:.4f}")
print(f"Newton's Method Logistic Regression Accuracy: {accuracy_newton:.4f}")

# 重新可视化梯度下降法和牛顿法结果
plot_decision_boundary(model_gd, X_test_2d, y_test, "Gradient Descent")
plot_decision_boundary(model_newton, X_test_2d, y_test, "Newton's Method")
