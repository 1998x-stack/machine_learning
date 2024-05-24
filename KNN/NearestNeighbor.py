import numpy as np
from typing import List, Tuple

class NearestNeighbor:
    """
    K-Nearest Neighbor (KNN) Classifier implementation using numpy.

    Methods:
        - fit: Trains the model using training data.
        - predict: Predicts labels for test data based on the nearest neighbors.

    Example usage:
    ```
    knn = NearestNeighbor()
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    ```
    """

    def __init__(self):
        # 初始化训练数据和标签
        self.X_train: np.ndarray = np.array([])
        self.y_train: np.ndarray = np.array([])

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the model by memorizing the training data.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test: np.ndarray) -> List:
        """
        Predict the labels for the provided test data.
        
        Args:
            X_test (np.ndarray): Testing features
        
        Returns:
            List: Predicted labels for the test data
        """
        predictions = []
        for test_point in X_test:
            # 计算每个测试样本与所有训练样本的L2距离
            distances = np.linalg.norm(self.X_train - test_point, axis=1)
            # 找到最接近的训练样本的索引
            closest_index = np.argmin(distances)
            # 根据最接近的训练样本的标签进行预测
            predictions.append(self.y_train[closest_index])
        return predictions

# 创建一个示例对象并显示其方法的文档字符串
knn = NearestNeighbor()
print(knn.__doc__)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加载鸢尾花数据集
iris = load_iris()
X, y = iris.data, iris.target

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 检查数据集的大小和维度
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# 实例化KNN模型并进行训练
knn.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = knn.predict(X_test)

# 计算测试集的准确率
accuracy = np.mean(y_pred == y_test)
accuracy

import matplotlib.pyplot as plt

# 选择两个特征进行可视化
feature_x = 0
feature_y = 1

# 使用测试集的两个特征进行绘图
plt.figure(figsize=(8, 6))

# 原始标签
plt.scatter(X_test[:, feature_x], X_test[:, feature_y], c=y_test, marker='o', label='Actual', alpha=0.5)

# 预测标签
plt.scatter(X_test[:, feature_x], X_test[:, feature_y], c=y_pred, marker='x', label='Predicted', alpha=0.7)

# 图例和标签
plt.legend()
plt.xlabel(iris.feature_names[feature_x])
plt.ylabel(iris.feature_names[feature_y])
plt.title('KNN Classification Results on Iris Dataset')

plt.show()
