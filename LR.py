# Import necessary libraries
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class LinearRegressionModelRegularized:
    """带正则化的线性回归模型
    包含线性回归模型的训练和预测功能，并引入L2正则化避免过拟合
    """
    def __init__(self, l2_penalty: float = 0.1, l1_penalty: float = 0.0) -> None:
        """初始化模型
        参数:
        l2_penalty -- L2正则化惩罚系数
        """
        self.weights = None
        self.bias = None
        self.l2_penalty = l2_penalty
        self.l1_penalty = l1_penalty

    def fit(self, X: np.ndarray, y: np.ndarray, learning_rate: float = 0.01, n_iters: int = 1000) -> None:
        """利用梯度下降算法训练线性回归模型，并添加L2正则化惩罚
        参数:
        X -- 特征矩阵
        y -- 目标变量
        learning_rate -- 学习率
        n_iters -- 迭代次数
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y)) + (self.l2_penalty / n_samples) * self.weights + (self.l1_penalty / n_samples) * np.sign(self.weights)
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        """利用模型进行预测
        参数:
        X -- 特征矩阵
        返回:
        模型预测的目标变量
        """
        return np.dot(X, self.weights) + self.bias

if __name__ == '__main__':
    # Generate synthetic data
    X, y = make_blobs(n_samples=200, centers=1, n_features=2, random_state=42)
    y = y.astype(np.float64)
    X_single_feature = X[:, [0]]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_single_feature, y, test_size=0.3, random_state=42)

    # Initialize and train the Linear Regression model with L2 regularization
    l2_penalty_value = 0.1
    linear_regression_regularized = LinearRegressionModelRegularized(l2_penalty=l2_penalty_value)
    linear_regression_regularized.fit(X_train, y_train, learning_rate=0.01, n_iters=1000)

    # Predict on the test data
    y_test_pred = linear_regression_regularized.predict(X_test)

    # Calculate mean squared error as evaluation
    test_mse = mean_squared_error(y_test, y_test_pred)

    # Visualize the predictions and the test data
    plt.scatter(X_test, y_test, color='blue', marker='o', label='Test Data Points')
    plt.plot(X_test, y_test_pred, color='red', linewidth=2, label='Regression Line with L2 Regularization')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.legend()
    plt.title(f'Linear Regression Model with L2 Regularization\nTest MSE: {test_mse:.2f}')
    plt.savefig('figures/linear_regression_l2_regularization.png')
    plt.close()

    # Return the MSE for external review
    print(f'Test MSE with L2 Regularization: {test_mse:.2f}')