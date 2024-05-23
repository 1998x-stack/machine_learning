import numpy as np
from typing import List, Tuple, Any, Union

class DecisionTree:
    def __init__(self, criterion: str = 'gini'):
        """
        初始化决策树
        :param criterion: 划分标准，支持'gini', 'entropy', 'information_gain_ratio'
        """
        self.criterion = criterion
        self.tree = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        训练决策树模型
        :param X: 特征数据集
        :param y: 标签数据集
        """
        if self.criterion == 'gini':
            self.tree = self._build_tree_CART(X, y)
        elif self.criterion == 'entropy':
            self.tree = self._build_tree_ID3(X, y)
        elif self.criterion == 'information_gain_ratio':
            self.tree = self._build_tree_C45(X, y)
        else:
            raise ValueError("不支持的划分标准")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测新数据
        :param X: 新的特征数据集
        :return: 预测结果
        """
        return np.array([self._predict_single(sample, self.tree) for sample in X])

    def _predict_single(self, sample: np.ndarray, tree: dict) -> Any:
        """
        预测单个样本
        :param sample: 单个样本
        :param tree: 决策树
        :return: 预测结果
        """
        if not isinstance(tree, dict):
            return tree
        root = next(iter(tree))
        branches = tree[root]
        value = sample[root]
        return self._predict_single(sample, branches[value])

    def _calculate_entropy(self, y: np.ndarray) -> float:
        """
        计算熵
        :param y: 标签数据集
        :return: 熵值
        """
        unique, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities))

    def _calculate_gini(self, y: np.ndarray) -> float:
        """
        计算基尼指数
        :param y: 标签数据集
        :return: 基尼指数值
        """
        unique, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1.0 - np.sum(probabilities ** 2)

    def _calculate_information_gain(self, X: np.ndarray, y: np.ndarray, feature: int) -> float:
        """
        计算信息增益
        :param X: 特征数据集
        :param y: 标签数据集
        :param feature: 特征索引
        :return: 信息增益
        """
        base_entropy = self._calculate_entropy(y)
        values, counts = np.unique(X[:, feature], return_counts=True)
        weighted_entropy = np.sum((counts[i] / len(X) * self._calculate_entropy(y[X[:, feature] == value]) for i, value in enumerate(values)))
        return base_entropy - weighted_entropy

    def _calculate_information_gain_ratio(self, X: np.ndarray, y: np.ndarray, feature: int) -> float:
        """
        计算信息增益比
        :param X: 特征数据集
        :param y: 标签数据集
        :param feature: 特征索引
        :return: 信息增益比
        """
        information_gain = self._calculate_information_gain(X, y, feature)
        split_info = self._calculate_entropy(X[:, feature])
        return information_gain / split_info if split_info != 0 else 0

    def _build_tree_CART(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        构建CART决策树
        :param X: 特征数据集
        :param y: 标签数据集
        :return: 决策树
        """
        if len(np.unique(y)) == 1:
            return y[0]
        if X.shape[1] == 0:
            return np.bincount(y).argmax()
        best_feature = self._choose_best_feature_CART(X, y)
        tree = {best_feature: {}}
        for value in np.unique(X[:, best_feature]):
            sub_X, sub_y = self._split_dataset(X, y, best_feature, value)
            tree[best_feature][value] = self._build_tree_CART(sub_X, sub_y)
        return tree

    def _choose_best_feature_CART(self, X: np.ndarray, y: np.ndarray) -> int:
        """
        选择最佳特征（基于CART）
        :param X: 特征数据集
        :param y: 标签数据集
        :return: 最佳特征索引
        """
        best_gini = float('inf')
        best_feature = -1
        for feature in range(X.shape[1]):
            gini = self._calculate_gini_index(X, y, feature)
            if gini < best_gini:
                best_gini = gini
                best_feature = feature
        return best_feature

    def _calculate_gini_index(self, X: np.ndarray, y: np.ndarray, feature: int) -> float:
        """
        计算基尼指数（基于CART）
        :param X: 特征数据集
        :param y: 标签数据集
        :param feature: 特征索引
        :return: 基尼指数
        """
        values, counts = np.unique(X[:, feature], return_counts=True)
        gini_index = 0.0
        for value in values:
            sub_y = y[X[:, feature] == value]
            gini = self._calculate_gini(sub_y)
            gini_index += (len(sub_y) / len(y)) * gini
        return gini_index

    def _build_tree_ID3(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        构建ID3决策树
        :param X: 特征数据集
        :param y: 标签数据集
        :return: 决策树
        """
        if len(np.unique(y)) == 1:
            return y[0]
        if X.shape[1] == 0:
            return np.bincount(y).argmax()
        best_feature = self._choose_best_feature_ID3(X, y)
        tree = {best_feature: {}}
        for value in np.unique(X[:, best_feature]):
            sub_X, sub_y = self._split_dataset(X, y, best_feature, value)
            tree[best_feature][value] = self._build_tree_ID3(sub_X, sub_y)
        return tree

    def _choose_best_feature_ID3(self, X: np.ndarray, y: np.ndarray) -> int:
        """
        选择最佳特征（基于ID3）
        :param X: 特征数据集
        :param y: 标签数据集
        :return: 最佳特征索引
        """
        best_info_gain = -1
        best_feature = -1
        for feature in range(X.shape[1]):
            info_gain = self._calculate_information_gain(X, y, feature)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = feature
        return best_feature

    def _build_tree_C45(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        构建C4.5决策树
        :param X: 特征数据集
        :param y: 标签数据集
        :return: 决策树
        """
        if len(np.unique(y)) == 1:
            return y[0]
        if X.shape[1] == 0:
            return np.bincount(y).argmax()
        best_feature = self._choose_best_feature_C45(X, y)
        tree = {best_feature: {}}
        for value in np.unique(X[:, best_feature]):
            sub_X, sub_y = self._split_dataset(X, y, best_feature, value)
            tree[best_feature][value] = self._build_tree_C45(sub_X, sub_y)
        return tree

    def _choose_best_feature_C45(self, X: np.ndarray, y: np.ndarray) -> int:
        """
        选择最佳特征（基于C4.5）
        :param X: 特征数据集
        :param y: 标签数据集
        :return: 最佳特征索引
        """
        best_info_gain_ratio = -1
        best_feature = -1
        for feature in range(X.shape[1]):
            info_gain_ratio = self._calculate_information_gain_ratio(X, y, feature)
            if info_gain_ratio > best_info_gain_ratio:
                best_info_gain_ratio = info_gain_ratio
                best_feature = feature
        return best_feature

    def _split_dataset(self, X: np.ndarray, y: np.ndarray, feature: int, value: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        按照给定特征划分数据集
        :param X: 特征数据集
        :param y: 标签数据集
        :param feature: 特征索引
        :param value: 特征值
        :return: 划分后的特征数据集和标签数据集
        """
        mask = X[:, feature] == value
        return X[mask], y[mask]

    def visualize(self):
        """
        可视化决策树（仅在需要时使用）
        """
        if self.tree is None:
            raise ValueError("决策树尚未构建，请先调用 fit 方法")
        # 使用networkx和matplotlib绘制决策树
        import matplotlib.pyplot as plt
        import networkx as nx

        def add_edges(graph, tree, parent=None):
            if isinstance(tree, dict):
                root = next(iter(tree))
                for k, v in tree[root].items():
                    child = f"{root}={k}"
                    graph.add_edge(parent, child)
                    add_edges(graph, v, child)
            else:
                graph.add_edge(parent, tree)

        graph = nx.DiGraph()
        add_edges(graph, self.tree, "Root")
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, with_labels=True, arrows=True)
        plt.savefig(f'figures/decision_tree_{self.criterion}.png')
        plt.close()

# 测试代码
if __name__ == "__main__":
    # 示例数据集
    data_set = np.array([
        ['young', 'high', 'no', 'good', 'no'],
        ['young', 'high', 'no', 'excellent', 'no'],
        ['middle-aged', 'high', 'no', 'good', 'yes'],
        ['senior', 'medium', 'no', 'good', 'yes'],
        ['senior', 'low', 'yes', 'good', 'yes'],
        ['senior', 'low', 'yes', 'excellent', 'no'],
        ['middle-aged', 'low', 'yes', 'excellent', 'yes'],
        ['young', 'medium', 'no', 'good', 'no'],
        ['young', 'low', 'yes', 'good', 'yes'],
        ['senior', 'medium', 'yes', 'good', 'yes'],
        ['young', 'medium', 'yes', 'excellent', 'yes'],
        ['middle-aged', 'medium', 'no', 'excellent', 'yes'],
        ['middle-aged', 'high', 'yes', 'good', 'yes'],
        ['senior', 'medium', 'no', 'excellent', 'no']
    ])
    labels = np.array(['age', 'income', 'student', 'credit'])
    X = data_set[:, :-1]
    y = data_set[:, -1]

    # 构建决策树
    dt = DecisionTree(criterion='gini')
    dt.fit(X, y)
    print("决策树（CART）：", dt.tree)

    dt_entropy = DecisionTree(criterion='entropy')
    dt_entropy.fit(X, y)
    print("决策树（ID3）：", dt_entropy.tree)

    dt_c45 = DecisionTree(criterion='information_gain_ratio')
    dt_c45.fit(X, y)
    print("决策树（C4.5）：", dt_c45.tree)

    # 可视化决策树
    dt.visualize()
    dt_entropy.visualize()
    dt_c45.visualize()