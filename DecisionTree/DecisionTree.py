import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from typing import Optional, Dict, Any


class DecisionTreeNode:
    """
    树的节点类，用于表示一个决策树中的单个节点。
    A class representing a single node in a Decision Tree.

    Attributes:
        feature_index (Optional[int]): 用于分割节点的特征索引, 如果是叶子节点则为None.
                                      Index of the feature used for splitting this node (None if leaf node).
        threshold (Optional[float]): 用于分割的阈值， 如果是叶子节点则为None.
                                    Threshold value used for splitting (None if leaf node).
        left (Optional['DecisionTreeNode']): 左子节点.
                                            Left child node.
        right (Optional['DecisionTreeNode']): 右子节点.
                                             Right child node.
        is_leaf (bool): 标识节点是否为叶子节点.
                       Indicates if this node is a leaf.
        value (Optional[Dict[str, Any]]): 如果为叶子节点，保存预测值或统计信息.
                                        Holds prediction value/statistics if this is a leaf node.
    """

    def __init__(self, feature_index: Optional[int] = None, threshold: Optional[float] = None, left: Optional['DecisionTreeNode'] = None,
                 right: Optional['DecisionTreeNode'] = None, is_leaf: bool = False, value: Optional[Dict[str, Any]] = None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.value = value


class DecisionTree:
    """
    决策树类，用于从头实现决策树模型。
    A class to implement a Decision Tree model from scratch.

    Attributes:
        root (Optional[DecisionTreeNode]): 决策树的根节点.
                                          The root node of the decision tree.
        max_depth (Optional[int]): 树的最大深度.
                                  The maximum depth of the tree.
        min_samples_split (int): 节点分割时需要的最小样本数.
                                Minimum samples required to split a node.
        criterion (str): 选择分割的准则，可以是'gini'或'entropy'.
                         Criterion for splitting ('gini' or 'entropy').
    """

    def __init__(self, max_depth: Optional[int] = None, min_samples_split: int = 2, criterion: str = 'gini'):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion

    def _gini(self, y: np.ndarray) -> float:
        """
        计算Gini系数，用于评估分割的质量。
        Compute Gini coefficient for evaluating split quality.

        Args:
            y (np.ndarray): 样本的目标值.
                           Target values of the samples.

        Returns:
            float: Gini系数.
                   Gini coefficient.
        """
        _, counts = np.unique(y, return_counts=True)
        prob_sq = (counts / len(y)) ** 2
        return 1 - np.sum(prob_sq)

    def _entropy(self, y: np.ndarray) -> float:
        """
        计算信息熵，用于评估分割的质量。
        Compute entropy for evaluating split quality.

        Args:
            y (np.ndarray): 样本的目标值.
                           Target values of the samples.

        Returns:
            float: 信息熵.
                   Entropy value.
        """
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs))

    def _criterion_function(self, y: np.ndarray) -> float:
        """
        根据选择的准则返回分割评估函数。
        Returns the split evaluation function based on the chosen criterion.

        Args:
            y (np.ndarray): 样本的目标值.
                           Target values of the samples.

        Returns:
            float: 准则的结果值.
                   Resulting value of the criterion.
        """
        if self.criterion == 'gini':
            return self._gini(y)
        elif self.criterion == 'entropy':
            return self._entropy(y)
        raise ValueError("Unsupported criterion. Choose 'gini' or 'entropy'.")

    def _split(self, X: np.ndarray, y: np.ndarray, feature_index: int, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        根据特定的特征索引和阈值划分数据。
        Split data based on a specific feature index and threshold.

        Args:
            X (np.ndarray): 输入特征.
                           Input features.
            y (np.ndarray): 样本的目标值.
                           Target values of the samples.
            feature_index (int): 要用于分割的特征索引.
                                Index of the feature to use for splitting.
            threshold (float): 分割的阈值.
                              Threshold value for splitting.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 划分后的数据集.
                                                                  Split datasets.
        """
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask
        return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[Optional[int], Optional[float]]:
        """
        找到在所有特征中最优的分割索引和阈值。
        Find the best feature index and threshold across all features.

        Args:
            X (np.ndarray): 输入特征.
                           Input features.
            y (np.ndarray): 样本的目标值.
                           Target values of the samples.

        Returns:
            tuple[Optional[int], Optional[float]]: 最佳分割的特征索引和阈值.
                                                  Best feature index and threshold for splitting.
        """
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        current_criterion = self._criterion_function(y)

        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                _, _, y_left, y_right = self._split(X, y, feature_index, threshold)
                if len(y_left) < self.min_samples_split or len(y_right) < self.min_samples_split:
                    continue

                p_left = len(y_left) / len(y)
                gain = current_criterion - (p_left * self._criterion_function(y_left) + (1 - p_left) * self._criterion_function(y_right))

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> DecisionTreeNode:
        """
        递归地构建决策树。
        Recursively build the decision tree.

        Args:
            X (np.ndarray): 输入特征.
                           Input features.
            y (np.ndarray): 样本的目标值.
                           Target values of the samples.
            depth (int): 当前的树深度.
                        Current depth of the tree.

        Returns:
            DecisionTreeNode: 构建好的决策树节点.
                             The constructed decision tree node.
        """
        if len(np.unique(y)) == 1 or depth == self.max_depth or len(X) < self.min_samples_split:
            # 终止条件
            # Stop condition
            return DecisionTreeNode(is_leaf=True, value={"label": np.argmax(np.bincount(y))})

        feature_index, threshold = self._best_split(X, y)
        if feature_index is None:
            # 如果无法进一步分割，创建叶子节点
            # If no further splitting can be done, create a leaf node
            return DecisionTreeNode(is_leaf=True, value={"label": np.argmax(np.bincount(y))})

        left_X, right_X, left_y, right_y = self._split(X, y, feature_index, threshold)
        left_child = self._build_tree(left_X, left_y, depth + 1)
        right_child = self._build_tree(right_X, right_y, depth + 1)

        return DecisionTreeNode(feature_index=feature_index, threshold=threshold, left=left_child, right=right_child, is_leaf=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        训练决策树模型。
        Train the Decision Tree model.

        Args:
            X (np.ndarray): 输入特征.
                           Input features.
            y (np.ndarray): 样本的目标值.
                           Target values of the samples.
        """
        self.root = self._build_tree(X, y, depth=0)

    def _predict_node(self, node: DecisionTreeNode, X: np.ndarray) -> Any:
        """
        对单个数据点进行预测。
        Predict a single data point based on the given node.

        Args:
            node (DecisionTreeNode): 当前节点.
                                     Current node.
            X (np.ndarray): 单个数据点的特征.
                           Features of a single data point.

        Returns:
            Any: 节点的预测结果.
                 Node's prediction result.
        """
        if node.is_leaf:
            return node.value['label']

        # 根据特征索引和阈值分割数据点
        # Split data point based on feature index and threshold
        if X[node.feature_index] <= node.threshold:
            return self._predict_node(node.left, X)
        else:
            return self._predict_node(node.right, X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        对整个数据集进行预测。
        Predict an entire dataset.

        Args:
            X (np.ndarray): 输入特征.
                           Input features.

        Returns:
            np.ndarray: 预测结果数组.
                        Array of prediction results.
        """
        return np.array([self._predict_node(self.root, x) for x in X])

if __name__ == '__main__':
    # 加载数据集并分割为训练和测试集
    # Load dataset and split into training and test sets
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

    # 实例化并训练自定义决策树模型
    # Instantiate and train the custom Decision Tree model
    tree = DecisionTree(max_depth=5, criterion='gini')
    tree.fit(X_train, y_train)

    # 对测试集进行预测并打印准确性
    # Predict on the test set and print accuracy
    y_pred = tree.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy of custom Decision Tree on Iris dataset: {accuracy * 100:.2f}%")