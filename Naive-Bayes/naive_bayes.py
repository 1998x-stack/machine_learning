import numpy as np
from typing import List, Tuple

class NaiveBayesClassifier:
    """
    朴素贝叶斯分类器

    属性:
        vocab_list: 词汇表
        p0_vect: 类别0的词向量概率
        p1_vect: 类别1的词向量概率
        p_class1: 类别1的先验概率
    """

    def __init__(self):
        self.vocab_list = []
        self.p0_vect = None
        self.p1_vect = None
        self.p_class1 = 0.0

    def fit(self, data_set: List[List[str]], class_labels: List[int]) -> None:
        """
        训练朴素贝叶斯分类器

        参数:
            data_set: 训练数据集
            class_labels: 类别标签
        """
        self.vocab_list = self._create_vocab_list(data_set)
        train_matrix = [self._set_of_words2vec(self.vocab_list, doc) for doc in data_set]
        self.p0_vect, self.p1_vect, self.p_class1 = self._train(np.array(train_matrix), np.array(class_labels))

    def predict(self, input_data: List[str]) -> int:
        """
        预测输入数据的类别

        参数:
            input_data: 待分类的文本数据

        返回:
            类别标签 (0 或 1)
        """
        input_vec = self._set_of_words2vec(self.vocab_list, input_data)
        return self._classify(np.array(input_vec), self.p0_vect, self.p1_vect, self.p_class1)

    def _create_vocab_list(self, data_set: List[List[str]]) -> List[str]:
        """
        创建词汇表

        参数:
            data_set: 训练数据集

        返回:
            词汇表
        """
        vocab_set = set()
        for document in data_set:
            vocab_set = vocab_set | set(document)
        return list(vocab_set)

    def _set_of_words2vec(self, vocab_list: List[str], input_set: List[str]) -> List[int]:
        """
        将输入文档转换为词向量

        参数:
            vocab_list: 词汇表
            input_set: 输入文档

        返回:
            词向量
        """
        return_vec = [0] * len(vocab_list)
        for word in input_set:
            if word in vocab_list:
                return_vec[vocab_list.index(word)] = 1
        return return_vec

    def _train(self, train_matrix: np.ndarray, train_category: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        训练朴素贝叶斯分类器

        参数:
            train_matrix: 训练数据集的词向量矩阵
            train_category: 训练数据集的类别标签

        返回:
            类别0的词向量概率，类别1的词向量概率，类别1的先验概率
        """
        num_train_docs = len(train_matrix)
        num_words = len(train_matrix[0])
        p_abusive = sum(train_category) / float(num_train_docs)
        p0_num = np.ones(num_words)
        p1_num = np.ones(num_words)
        p0_denom = 2.0
        p1_denom = 2.0
        for i in range(num_train_docs):
            if train_category[i] == 1:
                p1_num += train_matrix[i]
                p1_denom += sum(train_matrix[i])
            else:
                p0_num += train_matrix[i]
                p0_denom += sum(train_matrix[i])
        p1_vect = np.log(p1_num / p1_denom)
        p0_vect = np.log(p0_num / p0_denom)
        return p0_vect, p1_vect, p_abusive

    def _classify(self, vec2classify: np.ndarray, p0_vec: np.ndarray, p1_vec: np.ndarray, p_class1: float) -> int:
        """
        使用朴素贝叶斯分类器进行分类

        参数:
            vec2classify: 待分类的词向量
            p0_vec: 类别0的词向量概率
            p1_vec: 类别1的词向量概率
            p_class1: 类别1的先验概率

        返回:
            分类结果 (0 或 1)
        """
        p1 = sum(vec2classify * p1_vec) + np.log(p_class1)
        p0 = sum(vec2classify * p0_vec) + np.log(1.0 - p_class1)
        return 1 if p1 > p0 else 0

def load_data_set() -> Tuple[List[List[str]], List[int]]:
    """
    加载数据集

    返回:
        文档列表和类别标签
    """
    posting_list = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    class_vec = [0, 1, 0, 1, 0, 1]  # 1 代表侮辱性文字，0 代表正常言论
    return posting_list, class_vec

def main():
    """
    主函数，用于测试朴素贝叶斯分类器
    """
    data_set, class_labels = load_data_set()
    nb_classifier = NaiveBayesClassifier()
    nb_classifier.fit(data_set, class_labels)
    
    test_data_1 = ['love', 'my', 'dalmation']
    test_result_1 = nb_classifier.predict(test_data_1)
    print(f"Test data: {test_data_1} classified as: {test_result_1}")
    
    test_data_2 = ['stupid', 'garbage']
    test_result_2 = nb_classifier.predict(test_data_2)
    print(f"Test data: {test_data_2} classified as: {test_result_2}")

if __name__ == "__main__":
    main()